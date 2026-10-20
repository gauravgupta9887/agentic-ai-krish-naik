import copy
import uuid
import pytest
from router_with_tests import build_graph, after_judge
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


@pytest.fixture
def graph():
    """A fresh graph with its own memory for every test."""
    return build_graph(InMemorySaver())


@pytest.fixture
def config():
    """A fresh new config"""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def run(graph, config, amount):
    return graph.invoke({"expense": amount, "comment": []}, config)


def visited(graph, config, amount):
    """Names of the nodes that ran, in order."""
    names = []
    for chunk in graph.stream({"expense": amount, "comment": []}, config,
                              stream_mode="updates"):
        names.extend(k for k in chunk if k != "__interrupt__")
    return names


def test_router_does_not_change_the_state():
    state = {"expense": 120, "comment": ["earlier"]}
    before = copy.deepcopy(state)
    after_judge(state)
    assert state == before


# ------------------------------------------------- 1. the router (no graph)

@pytest.mark.parametrize(
    "amount, expected",
    [
        (0, "auto_approve_node"),
        (49, "auto_approve_node"),
        (50, "manager_node"),
        (499, "manager_node"),
        (500, "director_node"),
        (501, "director_node"),
    ],
)
def test_router_picks_the_right_branch(amount, expected):
    assert after_judge({"expense": amount, "comment": []}) == expected


# ------------------------------------------ 2. the whole graph, three paths
def test_small_expense_is_auto_approved_without_pausing(graph, config):
    result = run(graph, config, 49)
    assert "__interrupt__" not in result
    assert result["comment"] == ["Starting the check for expense amount 49",
                                 "Expense auto approved"]


def test_manager_path_pauses_then_approves(graph, config):
    paused = run(graph, config, 50)
    assert "Manager" in paused["__interrupt__"][0].value["question"]

    final = graph.invoke(Command(resume="yes"), config)
    assert final["comment"][-1] == "Expense approved by manager"


def test_manager_path_can_reject(graph, config):
    run(graph, config, 200)
    final = graph.invoke(Command(resume="no"), config)
    assert final["comment"][-1] == "Expense not approved by manager"


def test_large_expense_goes_to_the_director(graph, config):
    paused = run(graph, config, 500)
    assert "Director" in paused["__interrupt__"][0].value["question"]
    final = graph.invoke(Command(resume="yes"), config)
    assert final["comment"][-1] == "Expense approved by director"


def test_path_taken_is_exactly_what_we_expect(graph, config):
    assert visited(graph, config, 49) == ["judge", "auto_approve_node"]


# ------------------------------------------------- 3. threads and memory
def test_different_threads_do_not_share_comments(graph):
    a = {"configurable": {"thread_id": "A"}}
    b = {"configurable": {"thread_id": "B"}}
    run(graph, a, 49)
    result_b = run(graph, b, 49)
    assert result_b["comment"] == ["Starting the check for expense amount 49",
                                   "Expense auto approved"]


def test_reusing_a_thread_keeps_the_old_comments(graph, config):
    """Documents a behaviour that surprised us: the same thread_id continues
    the old state."""
    run(graph, config, 49)
    second = run(graph, config, 49)
    assert len(second["comment"]) == 4
