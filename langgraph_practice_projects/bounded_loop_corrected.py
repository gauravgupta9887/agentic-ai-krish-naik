from typing import TypedDict, Literal, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

MAX_TRIES = 3      # the bound: no more than this many attempts
PASS_SCORE = 50    # a score at or above this passes
STEP = 10          # stand-in for "the model improves a bit each time"


class State(TypedDict):
    try_count: int
    score: int
    messages: Annotated[list[str], operator.add]


# ---- nodes: take the state, return only the changes ----
def attempt(state: State) -> dict:
    n = state["try_count"] + 1
    score = state["score"] + STEP
    return {
        "try_count": n,
        "score": score,
        "messages": [f"Attempt {n}: score {score}"],
    }


def passed(state: State) -> dict:
    return {"messages": [f"Passed on attempt {state['try_count']}"]}


def human_review(state: State) -> dict:
    answer = interrupt({"question": f"Gave up after {state['try_count']} attempts. Approve? yes or no"})
    approved = str(answer).strip().lower() == "yes"
    return {"messages": [f"Manual verdict: {'approved' if approved else 'not approved'}"]}


# ---- the router: a PURE function. Pass wins; then the bound; else go round again ----
def route_after_attempt(state: State) -> Literal["passed", "attempt", "human_review"]:
    if state["score"] >= PASS_SCORE:
        return "passed"
    if state["try_count"] >= MAX_TRIES:
        return "human_review"
    return "attempt"


def build_graph(checkpointer=None, router=route_after_attempt):
    """`router` can be swapped, so we can build a broken, unbounded graph on purpose."""
    builder = StateGraph(State)
    builder.add_node("attempt", attempt)
    builder.add_node("passed", passed)
    builder.add_node("human_review", human_review)

    builder.add_edge(START, "attempt")
    builder.add_conditional_edges(
        "attempt",
        router,
        {"attempt": "attempt", "passed": "passed", "human_review": "human_review"},
    )
    builder.add_edge("passed", END)
    builder.add_edge("human_review", END)
    return builder.compile(checkpointer=checkpointer)
