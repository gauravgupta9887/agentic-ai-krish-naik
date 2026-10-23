"""One run of this script = one process. Run it twice to test a 'restart'.

    python chat_cli.py --saver sqlite --db chat.db --thread alice "hello"
    python chat_cli.py --saver sqlite --db chat.db --thread alice "and again"
    python chat_cli.py --saver sqlite --db chat.db --thread alice --show     # read only
"""
import argparse

from echo_memory import build_graph, make_saver

parser = argparse.ArgumentParser()
parser.add_argument("message", nargs="?", default=None)
parser.add_argument("--saver", choices=["memory", "sqlite"], default="memory")
parser.add_argument("--db", default="chat_memory.db")
parser.add_argument("--thread", default="demo")
parser.add_argument("--show", action="store_true", help="only print the saved state, send nothing")
args = parser.parse_args()

graph = build_graph(make_saver(args.saver, args.db))
config = {"configurable": {"thread_id": args.thread}}

if args.show:
    values = graph.get_state(config).values
    print("saved messages:", values.get("messages", []))
    print("saved turn_count:", values.get("turn_count", 0))
else:
    result = graph.invoke({"messages": [args.message]}, config)
    print("messages:", result["messages"])
    print("turn_count:", result["turn_count"])
