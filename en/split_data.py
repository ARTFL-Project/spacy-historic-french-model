#!/usr/bin/env python3
"""
Split each century's sentence pool into the sentences to annotate for training and for
testing: 10,000 and 1,000 per century. The split is by document, so that no document has
sentences on both sides, and the documents of the gold and challenge sets are left out
altogether.

Ids are integers, unique across all files (annotate.py needs them): training sentences are
numbered from 1,000,000, test sentences from 2,000,000.

Writes data/sentences_train_<period>.jsonl and data/sentences_test_<period>.jsonl.
"""

import json
from collections import defaultdict
from pathlib import Path

PERIODS = ["1500-1600", "1600-1700", "1700-1800", "1800-1900", "1900-2000"]
TRAIN, TEST = 10000, 1000
GOLD_SETS = ["gold/gold_1_sentences.jsonl", "gold/gold_2_sentences.jsonl", "gold/challenge_sentences.jsonl"]
OUTPUT = Path("data")


def read(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f]


def main():
    gold_docs = {s["doc"] for path in GOLD_SETS for s in read(path)}
    OUTPUT.mkdir(exist_ok=True)
    next_id = {"train": 1_000_000, "test": 2_000_000}
    for period in PERIODS:
        pool = [s for s in read(f"sentences/sentences_{period}.jsonl") if s["doc"] not in gold_docs]
        by_doc = defaultdict(list)
        for sentence in pool:
            by_doc[sentence["doc"]].append(sentence)
        # documents in order of first appearance in the shuffled pool: test documents first
        test, train = [], []
        for sentences in by_doc.values():
            (test if len(test) < TEST else train).extend(sentences)
        test, train = test[:TEST], train[:TRAIN]
        if len(train) < TRAIN:
            print(f"warning: {period} has only {len(train)} training sentences")
        for split, sentences in (("train", train), ("test", test)):
            with open(OUTPUT / f"sentences_{split}_{period}.jsonl", "w", encoding="utf8") as f:
                for sentence in sentences:
                    f.write(json.dumps({"id": next_id[split], **sentence, "period": period}, ensure_ascii=False) + "\n")
                    next_id[split] += 1
        docs = lambda xs: len({s["doc"] for s in xs})
        print(f"{period}: {len(train)} training sentences from {docs(train)} documents, "
              f"{len(test)} test sentences from {docs(test)} documents")


if __name__ == "__main__":
    main()
