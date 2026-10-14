#!/usr/bin/env python3
"""
Assemble adjudicated gold annotations into a JSONL gold set.

The adjudicated files (gold/adjudicated/<period>.tsv) use the annotation output format:
a line "# sentence <id>", then one line per token: token, POS, TAG, lemma, and optionally a
note explaining a doubtful decision. Tokens are checked against the tokenized sentences,
tags against the tagsets.

    python assemble_gold.py gold/pilot_sentences.jsonl gold/adjudicated gold/gold_1.jsonl
    python assemble_gold.py gold/pilot_sentences.jsonl gold/adjudicated /tmp/check.jsonl --period 1500-1600

Notes are written next to the gold set (<gold>.notes.tsv) for review.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from annotate import POS_TAGS, PTB_TAGS, nfc
from historic_tokenizer import make_nlp


def read_adjudicated(directory):
    sentences, current = {}, None
    for path in sorted(Path(directory).glob("*.tsv")):
        for n, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
            header = re.fullmatch(r"#\s*sentence\s+(\d+)\s*", line.strip())
            if header:
                current = sentences.setdefault(int(header[1]), [])
            elif line.strip() and not line.startswith("#"):
                fields = line.split("\t")
                if len(fields) not in (4, 5):
                    raise ValueError(f"{path}:{n}: expected 4 or 5 tab-separated fields: {line!r}")
                current.append(fields + [""] * (5 - len(fields)))
    return sentences


def main(sentences_path, adjudicated_dir, gold_path, period=None):
    nlp = make_nlp()
    adjudicated = read_adjudicated(adjudicated_dir)
    errors, notes, gold = [], [], []
    with open(sentences_path, encoding="utf8") as f:
        sentences = [s for s in map(json.loads, f) if period is None or s["period"] == period]
    for sentence in sentences:
        tokens = [t.text for t in nlp.tokenizer(nfc(sentence["text"]))]
        lines = adjudicated.get(sentence["id"])
        if lines is None:
            errors.append(f"sentence {sentence['id']}: not adjudicated")
            continue
        if [nfc(w) for w, *_ in lines] != [nfc(t) for t in tokens]:
            errors.append(f"sentence {sentence['id']}: tokens differ from the tokenizer's")
            continue
        for i, (token, (_, pos, tag, lemma, note)) in enumerate(zip(tokens, lines)):
            if pos not in POS_TAGS or tag not in PTB_TAGS or not lemma.strip() or " " in lemma.strip():
                errors.append(f"sentence {sentence['id']}: bad labels for {token!r}: {pos} {tag} {lemma!r}")
            if note.strip():
                notes.append(f"{sentence['id']}\t{i}\t{token}\t{pos}\t{tag}\t{lemma}\t{note.strip()}")
        gold.append({"id": sentence["id"], "period": sentence["period"], "text": sentence["text"],
                     "annotations": {"tokens": [{"text": t, "pos": p, "tag": g, "lemma": l.strip()}
                                                for t, (_, p, g, l, _) in zip(tokens, lines)]}})
    if errors:
        print("\n".join(errors))
        sys.exit(1)
    with open(gold_path, "w", encoding="utf8") as out:
        for sentence in gold:
            out.write(json.dumps(sentence, ensure_ascii=False) + "\n")
    Path(gold_path).with_suffix(".notes.tsv").write_text(
        "id\tidx\ttoken\tPOS\tTAG\tlemma\tnote\n" + "\n".join(notes) + "\n", encoding="utf8")
    print(f"{len(gold)} gold sentences written to {gold_path}, {len(notes)} notes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sentences")
    parser.add_argument("adjudicated_dir")
    parser.add_argument("gold")
    parser.add_argument("--period", help="only check and write the sentences of this period")
    args = parser.parse_args()
    main(args.sentences, args.adjudicated_dir, args.gold, args.period)
