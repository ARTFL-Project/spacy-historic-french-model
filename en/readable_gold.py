#!/usr/bin/env python3
"""
Convert gold sentences between JSONL and a readable column format for hand correction,
the same layout as the French gold sets:

    ================================================================================
    SENTENCE 12 (1600-1700)
    ================================================================================
    TEXT: ...

    IDX   TOKEN                POS        TAG        LEMMA
    ----- -------------------- ---------- ---------- --------------------
    0     WHat                 PRON       WP         what
    7     Abbots               NOUN       NNS        abbot                 # ? NOUN NN abbot | note: ...

    python readable_gold.py to-txt gold/gold_1.jsonl gold/readable/gold_1.txt --runs annotations/gold_1
    python readable_gold.py to-jsonl gold/readable/gold_1.txt gold/gold_1.jsonl

Edit only the POS, TAG and LEMMA columns; token text and order must stay as they are.
With --runs, tokens worth checking are marked after a "#": "?" where the annotation runs
disagreed (with the other readings), "!" where the gold overrules a reading all runs agreed
on, and the adjudicator's note if there is one (read from <gold>.notes.tsv). Everything
after "#" is ignored when converting back.
"""

import argparse
import csv
import json
import re
from pathlib import Path

SEPARATOR = "=" * 80
FIELDS = ("pos", "tag", "lemma")


def reading(token):
    return (token["pos"], token["tag"], token["lemma"])


def load_runs(runs_dir):
    runs = []
    for path in sorted(Path(runs_dir).glob("*.jsonl")):
        with open(path, encoding="utf8") as f:
            runs.append({s["id"]: s["annotations"]["tokens"] for s in map(json.loads, f)})
    return runs


def load_notes(jsonl_path):
    path = Path(jsonl_path).with_suffix(".notes.tsv")
    notes = {}
    if path.exists():
        with open(path, encoding="utf8") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                notes.setdefault((int(row["id"]), int(row["idx"])), []).append(row["note"])
    return notes


def flags(sentence, i, token, runs, notes):
    marks = []
    # runs tokenized differently from the gold (e.g. before a tokenizer fix) can't be compared token by token
    words = [t["text"] for t in sentence["annotations"]["tokens"]]
    readings = [run[sentence["id"]][i] for run in runs
                if sentence["id"] in run and [t["text"] for t in run[sentence["id"]]] == words]
    if readings:
        distinct = {tuple(x.lower() if f == "lemma" else x for f, x in zip(FIELDS, reading(r))) for r in readings}
        gold = tuple(x.lower() if f == "lemma" else x for f, x in zip(FIELDS, reading(token)))
        others = [" ".join(r) for r in sorted(distinct) if r != gold]
        if len(distinct) > 1:
            marks.append("? " + " | ".join(others))
        elif others:
            marks.append("! runs: " + others[0])
    for note in notes.get((sentence["id"], i), []):
        marks.append("note: " + note)
    return "  # " + " | ".join(marks) if marks else ""


def to_txt(jsonl_path, txt_path, runs_dir=None):
    runs = load_runs(runs_dir) if runs_dir else []
    notes = load_notes(jsonl_path)
    Path(txt_path).parent.mkdir(parents=True, exist_ok=True)
    flagged = 0
    with open(jsonl_path, encoding="utf8") as f, open(txt_path, "w", encoding="utf8") as out:
        for sentence in map(json.loads, f):
            out.write(f"{SEPARATOR}\nSENTENCE {sentence['id']} ({sentence.get('period', '')})\n{SEPARATOR}\n")
            out.write(f"TEXT: {sentence['text']}\n\n")
            out.write(f"{'IDX':<5} {'TOKEN':<20} {'POS':<10} {'TAG':<10} LEMMA\n")
            out.write(f"{'-' * 5} {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 20}\n")
            for i, token in enumerate(sentence["annotations"]["tokens"]):
                mark = flags(sentence, i, token, runs, notes)
                flagged += bool(mark)
                line = f"{i:<5} {token['text']:<20} {token['pos']:<10} {token['tag']:<10} {token['lemma']}"
                out.write(f"{line:<70}{mark}\n" if mark else line + "\n")
            out.write("\n")
    print(f"{txt_path}: {flagged} tokens marked for checking")


def to_jsonl(txt_path, jsonl_path):
    sentences, current = [], None
    with open(txt_path, encoding="utf8") as f:
        for line in f:
            line = line.rstrip("\n")
            header = re.fullmatch(r"SENTENCE (\d+) \((.*)\)", line)
            if header:
                current = {"id": int(header[1]), "period": header[2], "text": None, "annotations": {"tokens": []}}
                sentences.append(current)
            elif line.startswith("TEXT: ") and current is not None:
                current["text"] = line[6:]
            elif current is not None and re.match(r"\d+\s", line):
                line = re.split(r"\s+#\s", line, maxsplit=1)[0]
                # tokens never contain spaces; the lemma is the last column
                idx, text, pos, tag, lemma = line.split(None, 4)
                if int(idx) != len(current["annotations"]["tokens"]):
                    raise ValueError(f"sentence {current['id']}: token {idx} out of order")
                current["annotations"]["tokens"].append({"text": text, "pos": pos, "tag": tag, "lemma": lemma.strip()})
    with open(jsonl_path, "w", encoding="utf8") as out:
        for sentence in sentences:
            out.write(json.dumps(sentence, ensure_ascii=False) + "\n")
    print(f"{len(sentences)} sentences written to {jsonl_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("direction", choices=["to-txt", "to-jsonl"])
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--runs", help="directory of annotation runs, to mark the tokens worth checking (to-txt)")
    args = parser.parse_args()
    if args.direction == "to-txt":
        to_txt(args.source, args.target, args.runs)
    else:
        to_jsonl(args.source, args.target)
