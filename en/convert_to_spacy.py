#!/usr/bin/env python3
"""
Write the spaCy training files from the corrected annotations.

Each sentence is tokenized again with the historic tokenizer, which gives the exact
whitespace; the annotated tokens must match its tokens. 5% of each century's training
sentences, taken by document, are held out for dev.

A third of the 1500-1800 training sentences printed with the long s are rewritten with s
(text and tokens; the labels don't change), so that the model handles both spellings: many
editions and OCR of early printed books print s for ſ. Each sentence still appears once.
Dev, test and gold sentences keep their printed spelling.

Reads training_data/annotations_<period>.jsonl, test_data/annotations_<period>_test_set.jsonl
and the gold sets; writes training_space/data/: train.spacy, dev.spacy, test.spacy,
<period>_test.spacy, gold_1.spacy, gold_2.spacy, challenge.spacy.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from spacy.tokens import DocBin

from annotate import nfc
from historic_tokenizer import make_nlp

PERIODS = ["1500-1600", "1600-1700", "1700-1800", "1800-1900", "1900-2000"]
LONG_S_PERIODS = {"1500-1600", "1600-1700", "1700-1800"}
LONG_S_SHARE = 1 / 3
NLP = make_nlp()


def read(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f]


def to_doc(sentence, skipped):
    doc = NLP.make_doc(nfc(sentence["text"]))
    tokens = sentence["annotations"]["tokens"]
    if [nfc(t.text) for t in doc] != [nfc(t["text"]) for t in tokens]:
        skipped["tokens differ from the tokenizer's"] += 1
        return None
    for token, labels in zip(doc, tokens):
        token.pos_, token.tag_, token.lemma_ = labels["pos"], labels["tag"], labels["lemma"]
    return doc


def without_long_s(sentence):
    tokens = [{**t, "text": t["text"].replace("ſ", "s")} for t in sentence["annotations"]["tokens"]]
    return {**sentence, "text": sentence["text"].replace("ſ", "s"), "annotations": {"tokens": tokens}}


def write(sentences, path):
    skipped = Counter()
    db = DocBin(attrs=["POS", "TAG", "LEMMA"])
    for sentence in sentences:
        doc = to_doc(sentence, skipped)
        if doc is not None:
            db.add(doc)
    db.to_disk(path)
    note = f", skipped {dict(skipped)}" if skipped else ""
    print(f"{path}: {len(db)} docs{note}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-fraction", type=float, default=0.05)
    parser.add_argument("--output-dir", default="training_space/data")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(0)
    train, dev, test = [], [], []
    for period in PERIODS:
        sentences = read(f"training_data/annotations_{period}.jsonl")
        by_doc = defaultdict(list)
        for sentence in sentences:
            by_doc[sentence.get("doc")].append(sentence)
        docs = sorted(by_doc, key=str)
        rng.shuffle(docs)
        held_out = []
        for doc in docs:
            if len(held_out) >= args.dev_fraction * len(sentences):
                break
            held_out.extend(by_doc[doc])
        held_ids = {s["id"] for s in held_out}
        dev += held_out
        train += [s for s in sentences if s["id"] not in held_ids]
        period_test = read(f"test_data/annotations_{period}_test_set.jsonl")
        write(period_test, out / f"{period}_test.spacy")
        test += period_test
    rewritten = 0
    for k, sentence in enumerate(train):
        if sentence.get("period") in LONG_S_PERIODS and "ſ" in sentence["text"] and rng.random() < LONG_S_SHARE:
            train[k] = without_long_s(sentence)
            rewritten += 1
    print(f"{rewritten} training sentences rewritten without the long s")
    rng.shuffle(train)
    write(train, out / "train.spacy")
    write(dev, out / "dev.spacy")
    write(test, out / "test.spacy")
    for name in ("gold_1", "gold_2", "challenge"):
        write(read(f"gold/{name}.jsonl"), out / f"{name}.spacy")


if __name__ == "__main__":
    main()
