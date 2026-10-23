import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from echo_memory import build_graph, make_saver

HERE = Path(__file__).parent


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def graph():
    return build_graph(make_saver("memory"))        # fresh memory per test


def thread(name=None):
    return {"configurable": {"thread_id": name or str(uuid.uuid4())}}


def say(graph, config, text):
    return graph.invoke({"messages": [text]}, config)


def cli(*args):
    """Run chat_cli.py as a separate process: this is a real 'restart'."""
    out = subprocess.run(
        [sys.executable, "chat_cli.py", *args],
        cwd=HERE, capture_output=True, text=True, check=True,
    )
    return out.stdout


# ------------------------------------------- 1. one thread keeps its memory
def test_first_turn_echoes_the_message(graph):
    result = say(graph, thread(), "hello")
    assert result["messages"] == ["hello", "echo (turn 1): hello"]
    assert result["turn_count"] == 1


def test_second_turn_on_same_thread_continues(graph):
    cfg = thread()
    say(graph, cfg, "one")
    result = say(graph, cfg, "two")
    assert result["messages"] == ["one", "echo (turn 1): one", "two", "echo (turn 2): two"]
    assert result["turn_count"] == 2


# ----------------------------------------------- 2. threads are isolated
def test_threads_do_not_share_messages_or_counts(graph):
    a, b = thread("a"), thread("b")
    say(graph, a, "a1")
    say(graph, a, "a2")
    result_b = say(graph, b, "b1")
    assert result_b["messages"] == ["b1", "echo (turn 1): b1"]
    assert result_b["turn_count"] == 1
    assert graph.get_state(a).values["turn_count"] == 2


def test_a_new_thread_id_starts_fresh(graph):
    say(graph, thread("old"), "hi")
    assert say(graph, thread("new"), "hi")["turn_count"] == 1


# ------------------------------------------------------- 3. the history
def test_history_has_one_checkpoint_per_step(graph):
    cfg = thread()
    say(graph, cfg, "one")
    say(graph, cfg, "two")
    history = list(graph.get_state_history(cfg))
    assert [s.metadata["step"] for s in history] == [4, 3, 2, 1, 0, -1]    # newest first
    assert [len(s.values.get("messages", [])) for s in history] == [4, 3, 2, 2, 1, 0]
    assert graph.get_state(cfg).next == ()                                  # run finished


# -------------------------------- 4. restart: the memory saver forgets...
def test_memory_saver_forgets_after_a_restart():
    first = cli("--saver", "memory", "--thread", "t", "hello")
    second = cli("--saver", "memory", "--thread", "t", "again")
    assert "turn_count: 1" in first
    assert "turn_count: 1" in second            # a new process: the first chat is gone


# ------------------------------------ ...and the sqlite saver remembers
def test_sqlite_saver_survives_a_restart(tmp_path):
    db = str(tmp_path / "chat.db")
    first = cli("--saver", "sqlite", "--db", db, "--thread", "t", "hello")
    second = cli("--saver", "sqlite", "--db", db, "--thread", "t", "again")   # a new process
    assert "turn_count: 1" in first
    assert "turn_count: 2" in second
    assert "echo (turn 1): hello" in second     # the first process's message is still there


def test_sqlite_state_can_be_read_by_a_third_process(tmp_path):
    db = str(tmp_path / "chat.db")
    cli("--saver", "sqlite", "--db", db, "--thread", "t", "hello")
    shown = cli("--saver", "sqlite", "--db", db, "--thread", "t", "--show")
    assert "saved turn_count: 1" in shown


def test_sqlite_threads_stay_isolated_across_restarts(tmp_path):
    db = str(tmp_path / "chat.db")
    cli("--saver", "sqlite", "--db", db, "--thread", "a", "hello")
    other = cli("--saver", "sqlite", "--db", db, "--thread", "b", "hello")
    assert "turn_count: 1" in other


def test_in_memory_sqlite_path_is_a_trap():
    """':memory:' looks like sqlite but lives only inside one process."""
    cli("--saver", "sqlite", "--db", ":memory:", "--thread", "t", "hello")
    second = cli("--saver", "sqlite", "--db", ":memory:", "--thread", "t", "again")
    assert "turn_count: 1" in second


def test_unknown_saver_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown saver"):
        make_saver("postgres")
