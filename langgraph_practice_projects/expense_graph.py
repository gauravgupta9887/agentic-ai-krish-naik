from typing import TypedDict, Literal, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

AUTO_LIMIT = 50        # below this: approved automatically
MANAGER_LIMIT = 500    # below this: a manager decides; at or above: a director


class State(TypedDict):
    expense: int
    comment: Annotated[list[str], operator.add]


# ---- the router: a PURE function. No printing, no changing state. ----
def route_by_amount(state: State) -> Literal["auto_approve", "manager", "director"]:
    amount = state["expense"]
    if amount < 0:
        raise ValueError(f"expense cannot be negative: {amount}")
    if amount < AUTO_LIMIT:
        return "auto_approve"
    if amount < MANAGER_LIMIT:
        return "manager"
    return "director"


# ---- nodes: take the state, return only the changes ----
def start_check(state: State) -> dict:
    return {"comment": [f"Checking expense of {state['expense']}"]}


def auto_approve(state: State) -> dict:
    return {"comment": ["Expense auto approved"]}


def manager(state: State) -> dict:
    answer = interrupt({"question": f"Manager: approve {state['expense']}? yes or no"})
    verdict = "approved" if answer == "yes" else "not approved"
    return {"comment": [f"Expense {verdict} by manager"]}


def director(state: State) -> dict:
    answer = interrupt({"question": f"Director: approve {state['expense']}? yes or no"})
    verdict = "approved" if answer == "yes" else "not approved"
    return {"comment": [f"Expense {verdict} by director"]}


def build_graph(checkpointer=None):
    builder = StateGraph(State)
    builder.add_node("start_check", start_check)
    builder.add_node("auto_approve", auto_approve)
    builder.add_node("manager", manager)
    builder.add_node("director", director)

    builder.add_edge(START, "start_check")
    builder.add_conditional_edges(
        "start_check",
        route_by_amount,
        {"auto_approve": "auto_approve", "manager": "manager", "director": "director"},
    )
    builder.add_edge("auto_approve", END)
    builder.add_edge("manager", END)
    builder.add_edge("director", END)
    return builder.compile(checkpointer=checkpointer)
