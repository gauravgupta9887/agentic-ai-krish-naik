import copy
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from bounded_loop import MAX_TRIES, build_graph, route_after_attempt


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def graph():
    """A fresh graph with its own memory for every test."""
    return build_graph(InMemorySaver())      # note the return!


@pytest.fixture
def config():
    """A brand-new thread id for every test."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def start(score):
    return {"try_count": 0, "score": score, "messages": []}


def visited(graph, config, score):
    """Names of the nodes that ran, in order."""
    names = []
    for chunk in graph.stream(start(score), config, stream_mode="updates"):
        names.extend(k for k in chunk if k != "__interrupt__")
    return names


# ------------------------------------------------- 1. the router (no graph)
@pytest.mark.parametrize(
    "score, try_count, expected",
    [
        (49, 1, "attempt"),         # just below the pass mark
        (50, 1, "passed"),          # exactly the pass mark
        (49, MAX_TRIES - 1, "attempt"),      # last attempt that still allows another
        (49, MAX_TRIES, "human_review"),     # the bound is reached
        (50, MAX_TRIES, "passed"),           # passing on the last allowed try still passes
        (0, MAX_TRIES + 5, "human_review"),  # past the bound still stops
    ],
)
def test_router_picks_the_right_branch(score, try_count, expected):
    state = {"score": score, "try_count": try_count, "messages": []}
    assert route_after_attempt(state) == expected


def test_router_does_not_change_the_state():
    state = {"score": 20, "try_count": 1, "messages": ["earlier"]}
    before = copy.deepcopy(state)
    route_after_attempt(state)
    assert state == before


# --------------------------------- 2. the three cases the exercise asks for
def test_passes_on_the_second_try(graph, config):
    result = graph.invoke(start(30), config)          # 40, then 50
    assert "__interrupt__" not in result
    assert result["try_count"] == 2
    assert result["messages"] == ["Attempt 1: score 40", "Attempt 2: score 50", "Passed on attempt 2"]


def test_passes_exactly_on_the_last_allowed_try(graph, config):
    result = graph.invoke(start(20), config)          # 30, 40, then 50 on try 3
    assert "__interrupt__" not in result
    assert result["try_count"] == MAX_TRIES
    assert result["messages"][-1] == f"Passed on attempt {MAX_TRIES}"


def test_never_passes_ends_in_human_review_after_exactly_max_tries(graph, config):
    """THE worst-case test: a check that never passes must not loop forever."""
    graph.invoke(start(-1000), config)
    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",)                 # paused at the escalation node
    assert snapshot.values["try_count"] == MAX_TRIES
    attempts = [m for m in snapshot.values["messages"] if m.startswith("Attempt")]
    assert len(attempts) == MAX_TRIES


def test_path_taken_is_exactly_what_we_expect(graph, config):
    assert visited(graph, config, 30) == ["attempt", "attempt", "passed"]


# -------------------------------------------- 3. the human step: pause, resume
@pytest.mark.parametrize("answer, verdict", [("yes", "approved"), (" YES ", "approved"), ("no", "not approved"), ("maybe", "not approved")])
def test_human_verdict(graph, config, answer, verdict):
    paused = graph.invoke(start(-1000), config)
    assert "Gave up after 3 attempts" in paused["__interrupt__"][0].value["question"]
    final = graph.invoke(Command(resume=answer), config)
    assert final["messages"][-1] == f"Manual verdict: {verdict}"


# ------------------------------------------------ 4. the bound really holds
@pytest.mark.parametrize("score", [-1000, -50, 0, 10, 19, 20, 30, 40, 49, 50, 90])
def test_attempts_never_exceed_the_bound(graph, score):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke(start(score), config)
    assert result["try_count"] <= MAX_TRIES


# ------------------------------------- 5. break it on purpose: no bound at all
def always_again(state):
    return "attempt"


def test_without_a_bound_only_the_recursion_limit_saves_us():
    broken = build_graph(InMemorySaver(), router=always_again)
    config = {"configurable": {"thread_id": "broken"}, "recursion_limit": 10}   # top level!
    with pytest.raises(GraphRecursionError):
        broken.invoke(start(0), config)


def test_recursion_limit_inside_configurable_is_silently_ignored(graph):
    """Documents a stumble: put recursion_limit at the top level of the config, not inside 'configurable'."""
    wrong = {"configurable": {"thread_id": "w", "recursion_limit": 2}}
    result = graph.invoke(start(10), wrong)                     # needs 4 steps, limit ignored: runs fine
    assert result["try_count"] == MAX_TRIES

    right = {"configurable": {"thread_id": "r"}, "recursion_limit": 2}
    with pytest.raises(GraphRecursionError):
        graph.invoke(start(10), right)
