#!/usr/bin/env python3
"""
Draw the sentences of the evaluation sets from the sentence pools:

- gold set 1: 20 sentences per century, used to choose the annotation model and to tune
  label fixes;
- gold set 2: 20 sentences per century, kept for the final evaluation;
- challenge set: sentences with rare historic constructions (thou + -est verbs, 'tis,
  ye as the article, fused forms such as Ile, an meaning if, Latin phrases...), which a
  random sample barely contains.

Every sentence comes from a different document, and these documents are kept out of the
training data (see their "doc" field). Ids are unique across the three sets.

Writes gold/gold_1_sentences.jsonl, gold/gold_2_sentences.jsonl and gold/challenge_sentences.jsonl.
"""

import json
import random
import re
from pathlib import Path

PERIODS = ["1500-1600", "1600-1700", "1700-1800", "1800-1900", "1900-2000"]
PER_PERIOD = 20
# (name, pattern, number of sentences): rare constructions, searched in the pools
CHALLENGES = [
    ("thou + 2nd person verb", r"\b[Tt]hou\s+(?:\w+(?:e?st|'st)|art|wilt|shalt|wert|wast)\b", 6),
    ("'tis, 'twas", r"['’][Tt](?:is|was|were|will|would)\b", 5),
    ("ye as the article", r"\byͤ|\b[Yy]e (?:same|said|Lord|King|which|other|first|whole)\b", 4),
    ("fused forms", r"\b(?:Ile|tis|twas|thabuse|thother|tone)\b", 4),
    ("an meaning if", r"(?:^|[,;:] )[Aa]n (?:it|thou|he|you|I|ye|she|they) ", 3),
    ("thorn abbreviations", r"\by[tu]\b|\bwt\b", 4),
    ("Latin inside English", r"\b(?:quod|ergo|enim|autem|sicut|nobis|dominus|ibidem|scilicet|videlicet|anno domini)\b", 5),
    ("-eth / -est in 18th-19th c.", r"\b\w{3,}eth\b", 3),
    ("dialect", r"\b\w+in['’](?=[\s,.;!?])|\b'em\b|\bwi['’]\b|\bo['’] th['’]", 6),
]
OUTPUT = Path("gold")


def load(period):
    with open(f"sentences/sentences_{period}.jsonl", encoding="utf8") as f:
        return [{**json.loads(line), "period": period} for line in f]


def main():
    pools = {period: load(period) for period in PERIODS}
    used_docs = set()

    def take(sentences, n):
        chosen = []
        for sentence in sentences:
            if sentence["doc"] not in used_docs:
                chosen.append(sentence)
                used_docs.add(sentence["doc"])
            if len(chosen) == n:
                break
        return chosen

    # the pools are shuffled: the first sentences from distinct documents are a random sample
    gold_1 = [s for period in PERIODS for s in take(pools[period], PER_PERIOD)]
    gold_2 = [s for period in PERIODS for s in take(pools[period][1000:], PER_PERIOD)]

    rng = random.Random(0)
    challenge = []
    for name, pattern, n in CHALLENGES:
        regex = re.compile(pattern)
        periods = PERIODS[3:] if name == "dialect" else PERIODS[2:4] if name.startswith("-eth") else PERIODS[:3]
        matches = [s for period in periods for s in pools[period][2000:] if regex.search(s["text"])]
        rng.shuffle(matches)
        chosen = take(matches, n)
        print(f"challenge: {name}: {len(matches)} candidates, {len(chosen)} taken")
        challenge += [{**s, "challenge": name} for s in chosen]

    next_id = 0
    OUTPUT.mkdir(exist_ok=True)
    for name, sentences in (("gold_1", gold_1), ("gold_2", gold_2), ("challenge", challenge)):
        with open(OUTPUT / f"{name}_sentences.jsonl", "w", encoding="utf8") as f:
            for sentence in sentences:
                f.write(json.dumps({"id": next_id, **sentence}, ensure_ascii=False) + "\n")
                next_id += 1
        print(f"{name}: {len(sentences)} sentences")


if __name__ == "__main__":
    main()
