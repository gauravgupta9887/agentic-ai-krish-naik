from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, START, END
import operator as op
from langgraph.types import interrupt

MAX_TRIES = 3


class State(TypedDict):
    try_count: int
    score: int
    messages: Annotated[List[str], op.add]


def start_flow(state: State):
    return {
        "messages": [f"Start function called with present try value of {state['try_count']}"],
        "try_count": state["try_count"] + 1,
        "score": state["score"] + 10
    }


def normal_flow(state):
    return {
            "messages": [f"It's a normal pass in try no. {state['try_count']}"],
        }


def human_flow(state):
    answer = interrupt({"question":"Tried so many times. Do you want to pass [YES/NO]"})
    verdict = "approved" if answer == "YES" else "not approved"
    return {"messages": [f"Manual verdict : {verdict}"]}

def judge(state)->Literal["normal_flow","start_flow","human_flow"]:
    if state["score"] >= 50:
        return "normal_flow"
    else:
        if state["try_count"] < MAX_TRIES:
            return "start_flow"
        else:
            return "human_flow"
    
def build_graph(checkpointer=None):
    builder = StateGraph(State)
    
    builder.add_node("start_flow", start_flow)
    builder.add_node("normal_flow", normal_flow)
    builder.add_node("human_flow", human_flow)
    
    builder.add_edge(START, "start_flow")      
    builder.add_conditional_edges("start_flow", judge)
    builder.add_edge("normal_flow", END)
    builder.add_edge("human_flow", END)  

    graph = builder.compile(checkpointer=checkpointer)

    return graph
