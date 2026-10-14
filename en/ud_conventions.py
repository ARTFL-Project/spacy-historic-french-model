#!/usr/bin/env python3
"""
Label conventions of the UD English treebanks EWT and GUM, the reference for everything in
ANNOTATION_GUIDELINES.md that is not specific to historic English.

    python ud_conventions.py lookup following "one another"
    python ud_conventions.py check gold/gold_1.jsonl
    python ud_conventions.py check annotations/gold_1/gemini-3.8-flash_low.jsonl --details

`lookup` shows how the treebanks label a word or a phrase, with an example of each reading.
`check` lists the tokens of an annotation file (annotate.py or gold format) whose labels
depart from a dominant convention: a POS, TAG or lemma that the treebanks give a word in at
least 90% of at least 20 occurrences, a POS/TAG pair they (almost) never use, or a
preposition before a gerund that is not SCONJ. Context-dependent words (that, as, but...)
have no dominant reading, so they are never flagged: those need judgement.

The treebanks are expected in raw/ud:
    git clone --depth 1 https://github.com/UniversalDependencies/UD_English-EWT raw/ud/UD_English-EWT
    git clone --depth 1 https://github.com/UniversalDependencies/UD_English-GUM raw/ud/UD_English-GUM
"""

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

UD_DIR = Path("raw/ud")
FIELDS = ("pos", "tag", "lemma")
MIN_COUNT = 20
MIN_SHARE = 0.9
RARE_PAIR = 0.005  # a POS/TAG pair given to less than 0.5% of a TAG's tokens


def normalize(form):
    return unicodedata.normalize("NFC", form).lower().replace("ſ", "s").replace("’", "'")


def load_treebanks():
    sentences = []
    for path in sorted(UD_DIR.glob("UD_English-*/*.conllu")):
        sentence, text = [], ""
        for line in open(path, encoding="utf8"):
            line = line.rstrip("\n")
            if line.startswith("# text = "):
                text = line[9:]
            elif not line:
                if sentence:
                    sentences.append((text, sentence))
                sentence = []
            elif not line.startswith("#"):
                fields = line.split("\t")
                if "-" in fields[0] or "." in fields[0]:
                    continue  # multiword tokens and empty nodes
                sentence.append({"text": fields[1], "lemma": fields[2], "pos": fields[3], "tag": fields[4]})
        if sentence:
            sentences.append((text, sentence))
    if not sentences:
        raise SystemExit(f"no treebank found in {UD_DIR}: see the instructions at the top of this file")
    return sentences


class Conventions:
    def __init__(self):
        self.sentences = load_treebanks()
        self.by_form = defaultdict(lambda: {field: Counter() for field in FIELDS})
        self.pairs = defaultdict(Counter)
        for _, sentence in self.sentences:
            for token in sentence:
                counts = self.by_form[normalize(token["text"])]
                for field in FIELDS:
                    counts[field][token[field].lower() if field == "lemma" else token[field]] += 1
                self.pairs[token["tag"]][token["pos"]] += 1

    def dominant(self, form, field):
        """The value the treebanks give this form in at least MIN_SHARE of MIN_COUNT occurrences, if any."""
        counts = self.by_form.get(normalize(form), {}).get(field)
        if not counts:
            return None
        value, n = counts.most_common(1)[0]
        total = sum(counts.values())
        return (value, n, total) if total >= MIN_COUNT and n / total >= MIN_SHARE else None

    def rare_pair(self, pos, tag):
        counts = self.pairs.get(tag)
        return bool(counts) and counts[pos] / sum(counts.values()) < RARE_PAIR

    def lookup(self, query, examples=1):
        words = [normalize(w) for w in query.split()]
        readings = defaultdict(list)
        for text, sentence in self.sentences:
            forms = [normalize(t["text"]) for t in sentence]
            for i in range(len(forms) - len(words) + 1):
                if forms[i : i + len(words)] == words:
                    key = " + ".join(f"{t['pos']} {t['tag']} {t['lemma']}" for t in sentence[i : i + len(words)])
                    readings[key].append(text)
        total = sum(len(v) for v in readings.values())
        print(f"== {query}: {total} occurrences")
        for key, texts in sorted(readings.items(), key=lambda item: -len(item[1]))[:12]:
            print(f"  {len(texts):6} ({100 * len(texts) / total:4.1f}%)  {key}")
            for text in texts[:examples]:
                print(f"           e.g. {text[:110]}")


def check(conventions, path, details=False):
    flags = Counter()
    examples = defaultdict(list)
    n_tokens = 0
    with open(path, encoding="utf8") as f:
        for sentence in map(json.loads, f):
            tokens = sentence["annotations"]["tokens"]
            for i, token in enumerate(tokens):
                n_tokens += 1
                found = []
                for field in FIELDS:
                    dominant = conventions.dominant(token["text"], field)
                    ours = token[field].lower() if field == "lemma" else token[field]
                    if dominant and ours != dominant[0]:
                        found.append((field, ours, f"{dominant[0]} ({dominant[1]}/{dominant[2]})"))
                if conventions.rare_pair(token["pos"], token["tag"]):
                    found.append(("POS/TAG", f"{token['pos']} {token['tag']}", "pair rare in UD"))
                if (token["tag"] == "IN" and token["pos"] != "SCONJ" and i + 1 < len(tokens)
                        and tokens[i + 1]["tag"] == "VBG"):
                    found.append(("pos", token["pos"], "SCONJ (preposition before a gerund)"))
                for field, ours, ud in found:
                    key = (normalize(token["text"]), field, ours, ud)
                    flags[key] += 1
                    context = " ".join(t["text"] for t in tokens[max(0, i - 4) : i + 4])
                    examples[key].append(f"{sentence.get('id')}: {context}")
    print(f"{path}: {n_tokens} tokens, {sum(flags.values())} departures from UD conventions")
    for (form, field, ours, ud), n in flags.most_common():
        print(f"  {n:4}  {form:15} {field:8} ours {ours:12} UD {ud}")
        if details:
            for example in examples[(form, field, ours, ud)][:3]:
                print(f"          {example}")
    return flags


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    lookup = sub.add_parser("lookup")
    lookup.add_argument("queries", nargs="+", help="words or phrases")
    lookup.add_argument("--examples", type=int, default=1)
    checker = sub.add_parser("check")
    checker.add_argument("files", nargs="+")
    checker.add_argument("--details", action="store_true")
    args = parser.parse_args()
    conventions = Conventions()
    if args.command == "lookup":
        for query in args.queries:
            conventions.lookup(query, args.examples)
    else:
        for path in args.files:
            check(conventions, path, args.details)


if __name__ == "__main__":
    main()
