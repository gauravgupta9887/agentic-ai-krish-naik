from typing import TypedDict, Literal, Annotated, List
import operator as op
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt


class State(TypedDict):
    expense: int
    comment: Annotated[List[str], op.add]

 
def manager_node(state: State):
    answer = interrupt({"question": f"Shall the amount {state['expense']} be approved Manager? say yes or no"})
    return {"comment": ["Expense approved by manager"] if answer == "yes" else ["Expense not approved by manager"]}


def director_node(state: State):
    answer = interrupt({"question":f"Shall the amount {state['expense']} be approved Director? say yes or no"})
    return {"comment": ["Expense approved by director"] if answer=="yes" else ["Expense not approved by director"]}


def judge(state: State):
    return {"comment": [f"Starting the check for expense amount {state['expense']}"]}  


def auto_approve_node(state: State):
    return {"comment": ["Expense auto approved"]}


def after_judge(state: State) -> Literal["manager_node", "director_node", "auto_approve_node"]:
    if state["expense"] < 50:
        print("auto-approved")
        return "auto_approve_node"
    elif state["expense"] < 500:
        print("to be handled by manager")
        return "manager_node"
    else:
        print("to be handled by director")
        return "director_node"


def build_graph(checkpointer=None):
    builder = StateGraph(State)
    builder.add_node("judge", judge)
    builder.add_node("manager_node", manager_node)
    builder.add_node("director_node", director_node)
    builder.add_node("auto_approve_node", auto_approve_node)
    builder.add_edge(START, "judge")
    builder.add_conditional_edges("judge", after_judge)
    builder.add_edge("auto_approve_node", END)

    graph = builder.compile(checkpointer=checkpointer)
    return graph
