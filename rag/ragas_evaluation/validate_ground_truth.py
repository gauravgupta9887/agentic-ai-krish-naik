"""
Sanity-check the hand-written QA set BEFORE spending any LLM calls:
  * every evidence span must literally occur in the (normalised) corpus
  * ids unique, fields present, exactly the source file claimed contains the span
A ground-truth set with a typo in it silently turns into a permanent recall ceiling.
"""
import json
import sys
from collections import Counter

import config
from corpus import load_corpus

squash = lambda s: " ".join(s.split()).lower()


def main() -> int:
    qa = json.loads(config.GROUND_TRUTH.read_text())
    pages = load_corpus()
    by_source: dict[str, str] = {}
    for p in pages:
        by_source[p.source] = by_source.get(p.source, "") + "\n" + p.text
    by_source = {k: squash(v) for k, v in by_source.items()}

    bad = 0
    ids = Counter(q["id"] for q in qa)
    for q in qa:
        problems = []
        if ids[q["id"]] > 1:
            problems.append("duplicate id")
        for f in ("question", "ground_truth", "evidence", "source", "type"):
            if not q.get(f):
                problems.append(f"missing {f}")
        needle = squash(q["evidence"])
        if q["source"] not in by_source:
            problems.append(f"unknown source {q['source']}")
        elif needle not in by_source[q["source"]]:
            problems.append("evidence NOT found in claimed source")
        others = [s for s, t in by_source.items() if s != q["source"] and needle in t]
        if others:
            problems.append(f"evidence also appears in {others} (ambiguous)")
        print(f"{'OK ' if not problems else 'BAD'} {q['id']} [{q['type']:<16}] {', '.join(problems)}")
        bad += bool(problems)
    print(f"\n{len(qa)} questions, {bad} problems; types: {dict(Counter(q['type'] for q in qa))}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
