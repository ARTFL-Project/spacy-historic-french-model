#!/usr/bin/env python3
"""
Correct systematic slips in the LLM annotations and write the training and test files.

The rules come from ANNOTATION_GUIDELINES.md and from the departures that
`ud_conventions.py check` finds in the annotations; each is scored against the gold sets
(--score) before use, and only rules that don't lower accuracy there are kept.

    python fix_annotations.py --score     # effect of each rule on the gold sets
    python fix_annotations.py             # write training_data/ and test_data/

Reads annotations/{train,train2,test}_<period>/gemini-3.8-flash_low.jsonl (train2: the sentences
added for the earliest centuries, see extend_data.py); writes
training_data/annotations_<period>.jsonl and test_data/annotations_<period>_test_set.jsonl.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

PERIODS = ["1500-1600", "1600-1700", "1700-1800", "1800-1900", "1900-2000"]
RUN = "gemini-3.8-flash_low.jsonl"

# POS allowed with each TAG (see the tagset table of the guidelines); anything else takes the TAG's usual POS
ALLOWED = {
    "NN": {"NOUN", "PRON"}, "NNS": {"NOUN"}, "NNP": {"PROPN"}, "NNPS": {"PROPN"},
    "VB": {"VERB", "AUX"}, "VBD": {"VERB", "AUX"}, "VBG": {"VERB", "AUX"}, "VBN": {"VERB", "AUX"},
    "VBP": {"VERB", "AUX"}, "VBZ": {"VERB", "AUX"}, "MD": {"AUX"}, "JJ": {"ADJ"}, "JJR": {"ADJ"}, "JJS": {"ADJ"},
    "RB": {"ADV", "PART"}, "RBR": {"ADV"}, "RBS": {"ADV"}, "RP": {"ADP"}, "IN": {"ADP", "SCONJ"}, "TO": {"PART"},
    "CC": {"CCONJ"}, "DT": {"DET", "PRON"}, "PDT": {"DET"}, "PRP": {"PRON"}, "PRP$": {"PRON"}, "WDT": {"DET", "PRON"},
    "WP": {"PRON"}, "WP$": {"PRON"}, "WRB": {"ADV"}, "EX": {"PRON"}, "CD": {"NUM", "PRON"}, "POS": {"PART"},
    "UH": {"INTJ"}, "FW": {"X"}, "SYM": {"SYM"},
}
QUANTIFIERS = {"some", "any", "all", "each", "both", "either", "neither", "another"}
# prepositions that are SCONJ only before a gerund (before, after, since, till... can also introduce a clause)
PREPOSITIONS = {"of", "in", "from", "on", "by", "with", "at", "into", "upon", "without", "about", "through",
                "against", "towards", "toward", "among", "amongst", "within", "beyond", "besides", "instead", "unto"}
CLOSED_CLASS = {"ADP", "DET", "PRON", "CCONJ", "SCONJ", "AUX", "PART", "ADV", "VERB", "NUM"}


def norm(word):
    return word.lower().replace("ſ", "s").replace("’", "'")


def next_word(tokens, i):
    """The next token that isn't an adverb ("of not knowing")."""
    for j in range(i + 1, min(len(tokens), i + 3)):
        if tokens[j]["tag"] != "RB":
            return tokens[j]
    return None


def pair(tokens, i):
    t = tokens[i]
    if t["tag"] in ALLOWED and t["pos"] not in ALLOWED[t["tag"]]:
        from annotate import TAG_TO_POS
        return {"pos": TAG_TO_POS[t["tag"]]}


def quantifier(tokens, i):
    t = tokens[i]
    if t["pos"] == "PRON" and t["tag"] == "DT" and t["lemma"].lower() in QUANTIFIERS:
        return {"pos": "DET"}


def preposition_gerund(tokens, i):
    t = tokens[i]
    if t["tag"] != "IN" or norm(t["text"]) not in PREPOSITIONS or t["lemma"].lower() not in PREPOSITIONS:
        return None
    following = next_word(tokens, i)
    gerund = following is not None and following["tag"] == "VBG"
    if gerund and t["pos"] != "SCONJ":
        return {"pos": "SCONJ"}
    if not gerund and t["pos"] == "SCONJ":
        return {"pos": "ADP"}


def causal_for(tokens, i):
    t = tokens[i]
    if norm(t["text"]) == "for" and t["tag"] == "CC":
        return {"pos": "SCONJ", "tag": "IN", "lemma": "for"}


def according(tokens, i):
    t = tokens[i]
    if norm(t["text"]) == "according" and t["pos"] == "ADJ":
        return {"pos": "VERB", "tag": "VBG", "lemma": "accord"}


def people(tokens, i):
    t = tokens[i]
    if norm(t["text"]) == "people" and t["tag"] == "NN":
        return {"tag": "NNS", "lemma": "people"}


def the_adverb(tokens, i):
    t = tokens[i]
    if norm(t["text"]) in ("the", "th'") and t["pos"] == "ADV":
        return {"pos": "DET", "tag": "DT", "lemma": "the"}


def closed_class_lemma(tokens, i):
    t = tokens[i]
    lemma = t["lemma"].lower()
    if t["pos"] in ("DET", "PRON") and lemma in ("these", "those"):
        return {"lemma": {"these": "this", "those": "that"}[lemma]}
    if t["tag"] in ("JJR", "JJS", "RBR", "RBS") and lemma in ("much", "many", "little"):
        w = norm(t["text"])
        own = "more" if w.startswith(("mor", "moe")) else "most" if w.startswith(("mos", "moo")) else \
              "less" if w.startswith("les") else "least" if w.startswith("lea") else None
        if own:
            return {"lemma": own}
    if t["pos"] == "PRON" and lemma == "whom":
        return {"lemma": "who"}
    if t["pos"] == "DET" and norm(t["text"]) == "an" and lemma == "an":
        return {"lemma": "a"}
    if t["tag"] == "PRP" and lemma in ("mine", "thine"):
        return {"lemma": {"mine": "my", "thine": "thy"}[lemma]}


def long_s_lemma(tokens, i):
    # lemmas are in modern spelling: never a long s
    if "ſ" in tokens[i]["lemma"]:
        return {"lemma": tokens[i]["lemma"].replace("ſ", "s")}


def quote_lemma(tokens, i):
    t = tokens[i]
    if t["tag"] in ("``", "''") and t["lemma"] in ("“", "”", "‘", "’"):
        return {"lemma": '"' if t["lemma"] in "“”" else "'"}


def lemma_case(tokens, i):
    t = tokens[i]
    lemma = t["lemma"]
    if t["pos"] in CLOSED_CLASS and len(lemma) > 1 and lemma[0].isupper() and lemma not in ("I", "O"):
        return {"lemma": lemma.lower()}
    # common nouns: lowercase lemma, whatever the printed capital (all-capital abbreviations aside).
    # A proper noun with a lowercase lemma is left alone: on the gold sets it is more often a
    # capitalized common noun mistagged PROPN (Apoſtle, Strumpet) than a lemma to capitalize.
    if t["pos"] == "NOUN" and lemma[:1].isupper() and not lemma.isupper():
        return {"lemma": lemma.lower()}


RULES = [
    ("POS/TAG pair outside the tagset table -> POS of the TAG", pair),
    ("quantifier without a noun -> DET", quantifier),
    ("preposition: SCONJ before a gerund, ADP otherwise", preposition_gerund),
    ("for tagged CC -> SCONJ IN", causal_for),
    ("according -> VERB VBG accord", according),
    ("people -> NNS", people),
    ("the as ADV -> DET", the_adverb),
    ("closed-class lemmas (this/that, more/most, who, a, my/thy)", closed_class_lemma),
    ("quotation mark lemma -> straight", quote_lemma),
    ("long s in a lemma -> s", long_s_lemma),
    ("lemma capitalization", lemma_case),
]


def fix(sentence, stats, rules=RULES):
    tokens = [dict(t) for t in sentence["annotations"]["tokens"]]
    for name, rule in rules:
        for i in range(len(tokens)):
            change = rule(tokens, i)
            if change and any(tokens[i][k] != v for k, v in change.items()):
                tokens[i].update(change)
                stats[name] += 1
    return {**sentence, "annotations": {"tokens": tokens}}


def read(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f]


def accuracy(run, gold):
    n = correct = 0
    for sentence in run:
        reference = gold.get(sentence["id"])
        if not reference:
            continue
        for p, g in zip(sentence["annotations"]["tokens"], reference["annotations"]["tokens"]):
            n += 1
            correct += p["pos"] == g["pos"] and p["tag"] == g["tag"] and p["lemma"].lower() == g["lemma"].lower()
    return correct / n


def score():
    """Accuracy (all three labels) of Flash on the gold sets without the rules, with all of them, and without each."""
    for name in ("gold_1", "gold_2", "challenge"):
        gold = {s["id"]: s for s in read(f"gold/{name}.jsonl")}
        run = read(f"annotations/{name}/guidelines_v2/{RUN}")
        stats = Counter()
        fixed = [fix(s, stats) for s in run]
        print(f"{name}: {100 * accuracy(run, gold):.2f} -> {100 * accuracy(fixed, gold):.2f} with all rules")
        for rule_name, rule in RULES:
            if stats[rule_name]:
                without = [fix(s, Counter(), [r for r in RULES if r[1] is not rule]) for s in run]
                delta = 100 * (accuracy(fixed, gold) - accuracy(without, gold))
                print(f"   {stats[rule_name]:4} changes, {delta:+.2f} points  {rule_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true", help="score the rules on the gold sets instead")
    args = parser.parse_args()
    if args.score:
        return score()
    total = Counter()
    for period in PERIODS:
        for splits, target in ((("train", "train2"), f"training_data/annotations_{period}.jsonl"),
                               (("test",), f"test_data/annotations_{period}_test_set.jsonl")):
            stats = Counter()
            sentences = [s for split in splits if Path(f"annotations/{split}_{period}/{RUN}").exists()
                         for s in read(f"annotations/{split}_{period}/{RUN}")]
            fixed = [fix(s, stats) for s in sentences]
            Path(target).parent.mkdir(exist_ok=True)
            with open(target, "w", encoding="utf8") as f:
                for sentence in fixed:
                    f.write(json.dumps(sentence, ensure_ascii=False) + "\n")
            print(f"{target}: {len(fixed)} sentences, {sum(stats.values())} labels changed")
            total.update(stats)
    for name, n in total.most_common():
        print(f"  {n:6}  {name}")


if __name__ == "__main__":
    main()
