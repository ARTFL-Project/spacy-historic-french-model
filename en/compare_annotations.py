#!/usr/bin/env python3
"""
Compare annotation runs on the same sentences: pairwise agreement, accuracy against a gold
set when there is one, and an adjudication sheet listing every sentence with each run's
labels side by side (runs are anonymized as A, B, C... so that the adjudicator doesn't
know which model produced what).

    python compare_annotations.py annotations/pilot --gold gold/gold_1.jsonl
    python compare_annotations.py annotations/pilot --sheet gold/adjudication

Run files are <dir>/<run>.jsonl as written by annotate.py; usage files give cost and time.
"""

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

FIELDS = ("pos", "tag", "lemma")


def load(path):
    with open(path, encoding="utf8") as f:
        return {s["id"]: s for s in map(json.loads, f)}


def labels(token, field):
    value = token[field]
    return value.lower() if field == "lemma" else value


def score(pred, ref, ids):
    """Token accuracy of pred against ref, per field and for all three fields together."""
    counts = defaultdict(int)
    for i in ids:
        for p, r in zip(pred[i]["annotations"]["tokens"], ref[i]["annotations"]["tokens"]):
            counts["n"] += 1
            correct = [labels(p, f) == labels(r, f) for f in FIELDS]
            for field, ok in zip(FIELDS, correct):
                counts[field] += ok
            counts["all"] += all(correct)
    return {k: counts[k] / counts["n"] for k in (*FIELDS, "all")} if counts["n"] else {}


def table(rows, headers):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]
    lines = [" | ".join(str(h).ljust(w) for h, w in zip(headers, widths)), "-+-".join("-" * w for w in widths)]
    lines += [" | ".join(str(x).ljust(w) for x, w in zip(row, widths)) for row in rows]
    return "\n".join(lines)


def pct(x):
    return f"{100 * x:.1f}" if x is not None else "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir")
    parser.add_argument("--gold")
    parser.add_argument("--sheet", help="write an anonymized adjudication sheet to this directory")
    args = parser.parse_args()

    runs = {p.stem: load(p) for p in sorted(Path(args.runs_dir).glob("*.jsonl"))}
    usage = {p.name[: -len(".usage.json")]: json.loads(p.read_text()) for p in Path(args.runs_dir).glob("*.usage.json")}
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    periods = sorted({runs[next(iter(runs))][i]["period"] for i in ids})
    print(f"{len(runs)} runs, {len(ids)} sentences annotated by all of them\n")

    print("Cost and time")
    rows = []
    for name in runs:
        u = usage.get(name, {})
        n = u.get("annotated") or 1
        rows.append([name, u.get("annotated"), len(u.get("failures", [])), u.get("prompt_tokens"), u.get("completion_tokens"),
                     f"{u.get('cost', 0):.3f}", f"{1000 * u.get('cost', 0) / n:.2f}", u.get("seconds")])
    print(table(rows, ["run", "sentences", "failed", "input tok", "output tok", "cost $", "$ per 1K sent.", "seconds"]))

    print("\nPairwise agreement (POS / TAG / lemma / all three), %")
    rows = []
    for a, b in itertools.combinations(runs, 2):
        s = score(runs[a], runs[b], ids)
        rows.append([a, b] + [pct(s[k]) for k in (*FIELDS, "all")])
    print(table(rows, ["run", "run", "POS", "TAG", "lemma", "all"]))

    if args.gold:
        gold = load(args.gold)
        gold_ids = [i for i in ids if i in gold]
        print(f"\nAccuracy against {args.gold} ({len(gold_ids)} sentences), %")
        rows = []
        for name, run in runs.items():
            s = score(run, gold, gold_ids)
            rows.append([name, "all"] + [pct(s[k]) for k in (*FIELDS, "all")])
            for period in periods:
                s = score(run, gold, [i for i in gold_ids if gold[i].get("period", run[i]["period"]) == period])
                rows.append(["", period] + [pct(s.get(k)) for k in (*FIELDS, "all")])
        print(table(rows, ["run", "period", "POS", "TAG", "lemma", "all"]))

    if args.sheet:
        names = list(runs)
        random.Random(0).shuffle(names)
        letters = {name: chr(ord("A") + i) for i, name in enumerate(names)}
        out = Path(args.sheet)
        out.mkdir(parents=True, exist_ok=True)
        # the key stays outside the sheet directory, out of the adjudicator's sight
        key = out.parent / f"{out.name}_key.json"
        key.write_text(json.dumps(letters, indent=2))
        for period in periods:
            lines = []
            for i in ids:
                first = runs[names[0]][i]
                if first["period"] != period:
                    continue
                lines.append(f"# sentence {i}\n# text: {first['text']}")
                for k, token in enumerate(first["annotations"]["tokens"]):
                    readings = defaultdict(list)
                    for n in names:
                        readings[tuple(runs[n][i]["annotations"]["tokens"][k][f] for f in FIELDS)].append(letters[n])
                    if len(readings) == 1:
                        lines.append(f"{k}\t{token['text']}\t{' '.join(next(iter(readings)))}")
                    else:
                        ranked = sorted(readings.items(), key=lambda item: -len(item[1]))
                        cells = " | ".join(f"{' '.join(r)} [{','.join(sorted(v))}]" for r, v in ranked)
                        lines.append(f"{k}\t{token['text']}\t{cells}  <-- differ")
                lines.append("")
            (out / f"sheet_{period}.txt").write_text("\n".join(lines), encoding="utf8")
        print(f"\nAdjudication sheets written to {out}/ (key in {key})")


if __name__ == "__main__":
    main()
