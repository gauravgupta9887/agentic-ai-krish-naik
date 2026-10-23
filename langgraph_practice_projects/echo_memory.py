import sqlite3
from dataclasses import dataclass, field
from typing import Annotated
import operator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END


@dataclass
class State:
    messages: Annotated[list[str], operator.add] = field(default_factory=list)
    turn_count: int = 0          # no reducer: each node run overwrites it


def echo_bot(state: State) -> dict:
    """Reads the newest message, echoes it back, and counts the turn."""
    turn = state.turn_count + 1
    return {
        "messages": [f"echo (turn {turn}): {state.messages[-1]}"],
        "turn_count": turn,
    }


def make_saver(kind: str = "memory", db_path: str = "chat_memory.db"):
    """'memory' forgets when the process ends. 'sqlite' keeps everything in a file."""
    if kind == "memory":
        return InMemorySaver()
    if kind == "sqlite":
        return SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))
    raise ValueError(f"unknown saver kind: {kind!r}")


def build_graph(checkpointer=None):
    builder = StateGraph(State)
    builder.add_node("echo_bot", echo_bot)
    builder.add_edge(START, "echo_bot")
    builder.add_edge("echo_bot", END)
    return builder.compile(checkpointer=checkpointer)
