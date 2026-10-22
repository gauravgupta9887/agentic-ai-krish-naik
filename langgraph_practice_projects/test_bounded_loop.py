import copy
import uuid 

import pytest
from bounded_loop import build_graph, judge
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver


@pytest.fixture
def graph():
    """A fresh graph with its own memory for every test."""
    return build_graph(InMemorySaver())


@pytest.fixture
def config():
    """A fresh new config"""
    return {"configurable":{"thread_id":str(uuid.uuid4())}}


def run(graph, config, score):
    return graph.invoke({"score":score, "messages":[], "try_count": 0}, config)


def visited(graph, config, score):
    """Names of the nodes that ran, in order."""
    names = []
    for chunk in graph.stream({"score": score, "messages": [], "try_count":0}, config, stream_mode="updates"):
        names.extend(k for k in chunk if k != "__interrupt__")
    return names


def test_router_does_not_change_the_state():
    state = {"score": 120, "try_count":1, "messages": ["earlier"]}
    before = copy.deepcopy(state)
    judge(state)
    assert state == before


# ------------------------------------------------- 1. the router (no graph)

@pytest.mark.parametrize(
    "score, try_count, expected",
    [
        (10, 1, "start_flow"),
        (10, 4, "human_flow"),
        (50, 2, "normal_flow"),
        (60, 2, "normal_flow"),
    ],
)
def test_router_picks_the_right_branch(score, try_count, expected):
    assert judge({"score": score, "try_count": try_count, "messages": []}) == expected

# ------------------------------------------ 2. the whole graph, three paths
def test_small_score_is_auto_approved_without_pausing(graph, config):
    result = run(graph, config, 49)
    assert "__interrupt__" not in result
    assert result["messages"] == ["Start function called with present try value of 0", "It's a normal pass in try no. 1"]

def test_path_pauses_then_approves(graph, config):
    paused = run(graph, config, 10)
    assert "Tried so many times." in paused["__interrupt__"][0].value["question"]

    final = graph.invoke(Command(resume="YES"), config)
    assert final["messages"][-1] == "Manual verdict : approved"

def test_path_can_reject(graph, config):
    run(graph, config, 10)
    final = graph.invoke(Command(resume="no"), config)
    assert final["messages"][-1] == "Manual verdict : not approved"

def test_large_score_goes_to_the_director(graph, config):
    result = run(graph, config, 10)
    assert result["messages"] == ["Start function called with present try value of 0", "Start function called with present try value of 1","Start function called with present try value of 2"]


def test_path_taken_is_exactly_what_we_expect(graph, config):
    assert visited(graph, config, 59) == ["start_flow", "normal_flow"]

# ------------------------------------------------- 3. threads and memory
def test_different_threads_do_not_share_messagess(graph):
    a = {"configurable": {"thread_id": "A"}}
    b = {"configurable": {"thread_id": "B"}}
    run(graph, a, 49)
    result_b = run(graph, b, 49)
    assert result_b["messages"] == ["Start function called with present try value of 0", "It's a normal pass in try no. 1"]


def test_reusing_a_thread_keeps_the_old_messagess(graph, config):
    """Documents a behaviour that surprised us: the same thread_id continues the old state."""
    run(graph, config, 49)
    second = run(graph, config, 49)
    assert len(second["messages"]) == 4
