from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, RemoveMessage

from config import SYSTEM_PROMPT_BASE, WINDOW_SIZE, _ollama_complete
from memory import LongTermMemory

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str
    user_id: str
    session_id: str
    lt_context: str
    turn_count: int

def build_system_prompt(lt_context: str) -> str:
    if lt_context:
        lt_section = (
            "RETURNING CUSTOMER — facts recalled from previous sessions:\n"
            f"{lt_context}\n\n"
            "Use this information naturally. Do not say you 'looked it up'."
        )
    else:
        lt_section = "(No previous sessions found for this customer.)"
    return SYSTEM_PROMPT_BASE.format(long_term_context=lt_section)

def node_load_ltm(state: AgentState, ltm: LongTermMemory) -> dict:
    try:
        user_id = state["user_id"]
        if ltm.user_exists(user_id):
            ctx = ltm.retrieve(user_id)
            return {"lt_context": ctx}
        else:
            return {"lt_context": ""}
    except Exception as e:
        print(f"Error loading LTM: {e}")
        return {"lt_context": ""}

def node_chat(state: AgentState) -> dict:
    try:
        messages = state["messages"]
        summary = state.get("summary", "")

        window_msgs: list[AnyMessage] = []
        if summary:
            window_msgs.append(HumanMessage(content="[System note: earlier conversation summary]"))
            window_msgs.append(AIMessage(content=f"[Earlier summary]: {summary}"))

        window_msgs.extend(messages[-WINDOW_SIZE:])
        system = build_system_prompt(state.get("lt_context", ""))
        reply = _ollama_complete(system, window_msgs)

        return {
            "messages": [AIMessage(content=reply)],
            "turn_count": state.get("turn_count", 0) + 1,
        }
    except Exception as e:
        print(f"Error in chat node: {e}")
        return {
            "messages": [AIMessage(content="Sorry, I encountered an error processing your message.")],
        }

def node_summarise(state: AgentState) -> dict:
    try:
        messages = state["messages"]
        summary = state.get("summary", "")
        
        overflow_count = len(messages) - WINDOW_SIZE
        to_compress = messages[:overflow_count]

        def fmt(m: AnyMessage) -> str:
            label = "Customer" if isinstance(m, HumanMessage) else "MannKiBot"
            return f"[{label}]: {m.content}"

        excerpt = "\n".join(fmt(m) for m in to_compress)
        print(f"\n  ⚡ [Summary Memory] Compressing {overflow_count} messages...")

        summarise_system = (
            "You are a precise summariser. Output ONLY a 2-3 sentence summary. "
            "No preamble, no bullet points. Preserve: customer name, books ordered, "
            "reading preferences, any complaints."
        )
        new_piece = _ollama_complete(
            system=summarise_system,
            history=[HumanMessage(content=f"Summarise this conversation excerpt:\n\n{excerpt}")]
        )

        accumulated = f"{summary}\n{new_piece}".strip() if summary else new_piece
        print(f" Summary: {new_piece[:120]}...")

        removals = [RemoveMessage(id=m.id) for m in to_compress]
        print(f"Removed {overflow_count} messages; window restored to {WINDOW_SIZE}\n")

        return {
            "summary": accumulated,
            "messages": removals,
        }
    except Exception as e:
        print(f"Error in summarisation node: {e}")
        return {}

def router_should_summarise(state: AgentState) -> Literal["summarise", "__end__"]:
    try:
        if len(state["messages"]) > WINDOW_SIZE:
            return "summarise"
        return "__end__"
    except Exception as e:
        print(f"Error in router evaluation: {e}")
        return "__end__"

def extract_facts(messages: list[AnyMessage], summary: str) -> str:
    """Used by session.end() to extract facts for long-term storage"""
    try:
        def fmt(m: AnyMessage) -> str:
            label = "Customer" if isinstance(m, HumanMessage) else "MannKiBot"
            return f"[{label}]: {m.content}"

        body = ""
        if summary:
            body += f"[Earlier summary]: {summary}\n\n"
        body += "\n".join(fmt(m) for m in messages)

        system = (
            "You are a precise information extractor. Output ONLY the extracted facts "
            "as plain English sentences. No preamble, no headings, no bullet points. "
            "Cover: customer name, books ordered (exact titles + authors), reading preferences, "
            "favourite authors, complaints, any other relevant personal details. "
            "If something was not mentioned, omit it entirely."
        )
        return _ollama_complete(
            system=system,
            history=[HumanMessage(content=f"Extract all customer facts from this conversation:\n\n{body}")]
        )
    except Exception as e:
        print(f"Error extracting facts: {e}")
        return ""

def build_graph(ltm: LongTermMemory):
    """
    Topology:  load_ltm → chat → [router] → summarise → END
                                          └───────────► END
    """
    def _load_ltm(state: AgentState) -> dict:
        if state.get("turn_count", 0) > 0:
            return {}
        return node_load_ltm(state, ltm)

    try:
        g = StateGraph(AgentState)
        g.add_node("load_ltm", _load_ltm)
        g.add_node("chat", node_chat)
        g.add_node("summarise", node_summarise)

        g.set_entry_point("load_ltm")
        g.add_edge("load_ltm", "chat")
        g.add_conditional_edges(
            "chat",
            router_should_summarise,
            {"summarise": "summarise", "__end__": END}
        )
        g.add_edge("summarise", END)

        return g.compile(checkpointer=MemorySaver())
    except Exception as e:
        print(f"Failed to build LangGraph: {e}")
        raise
