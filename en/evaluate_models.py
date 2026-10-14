#!/usr/bin/env python3
"""
Score pipelines on POS, TAG and lemma: the three gold sets, the test set of each century and
the UD English EWT test set (modern web English, as a check on modern text).

    python evaluate_models.py                                  # the default models below
    python evaluate_models.py roberta=training_space/models_roberta/model-best en_core_web_trf

Missing model directories are skipped. Each dataset is tokenized by the pipeline being
scored and aligned with the gold tokens, so pipelines with other tokenizers (en_core_web_trf)
can be compared.
"""

import gc
import sys
from pathlib import Path

import spacy
from spacy.tokens import Doc, DocBin
from spacy.training import Example

DATA = Path("training_space/data")
EWT_TEST = Path("raw/ud/UD_English-EWT/en_ewt-ud-test.conllu")
PERIODS = ["1500-1600", "1600-1700", "1700-1800", "1800-1900", "1900-2000"]
DEFAULT_MODELS = {
    "en_core_web_trf": "en_core_web_trf",
    "roberta-large": "training_space/models_roberta/model-best",
    "macberth": "training_space/models_macberth/model-best",
}
DATASETS = {"Gold 1": DATA / "gold_1.spacy", "Gold 2": DATA / "gold_2.spacy", "Challenge": DATA / "challenge.spacy",
            "Test 1500-2000": DATA / "test.spacy", **{f"Test {p}": DATA / f"{p}_test.spacy" for p in PERIODS}}


def load_docbin(vocab, path):
    return list(DocBin().from_disk(path).get_docs(vocab))


def load_conllu(vocab, path):
    docs = []
    words, spaces, pos, tags, lemmas = [], [], [], [], []
    for line in open(path, encoding="utf8"):
        line = line.rstrip("\n")
        if not line:
            if words:
                docs.append(Doc(vocab, words=words, spaces=spaces, pos=pos, tags=tags, lemmas=lemmas))
            words, spaces, pos, tags, lemmas = [], [], [], [], []
        elif not line.startswith("#"):
            f = line.split("\t")
            if "-" in f[0] or "." in f[0]:
                continue
            words.append(f[1])
            spaces.append("SpaceAfter=No" not in f[9])
            pos.append(f[3])
            tags.append(f[4])
            lemmas.append(f[2])
    return docs


def evaluate(nlp, docs):
    predicted = nlp.pipe((doc.text for doc in docs), batch_size=64)
    examples = [Example(pred, gold) for pred, gold in zip(predicted, docs)]
    scores = nlp.evaluate(examples)
    return {key: scores.get(key) for key in ("pos_acc", "tag_acc", "lemma_acc")}


def table(results):
    headers = ["Dataset", "Model", "POS", "TAG", "Lemma"]
    rows = [[dataset, model] + [f"{100 * s[k]:.2f}" if s.get(k) is not None else "-" for k in ("pos_acc", "tag_acc", "lemma_acc")]
            for dataset, by_model in results.items() for model, s in by_model.items()]
    widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(line)
    print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    print(line.replace("-", "="))
    for row in rows:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |")
    print(line)


def main():
    spacy.require_gpu()
    models = dict(arg.split("=", 1) if "=" in arg else (arg, arg) for arg in sys.argv[1:]) or DEFAULT_MODELS
    vocab = spacy.blank("en").vocab
    datasets = {name: load_docbin(vocab, path) for name, path in DATASETS.items() if path.exists()}
    if EWT_TEST.exists():
        datasets["UD EWT test (modern)"] = load_conllu(vocab, EWT_TEST)
    results = {name: {} for name in datasets}
    for model, path in models.items():
        if model != path and not Path(path).exists():
            print(f"skipping {model}: {path} not found")
            continue
        nlp = spacy.load(path)
        for name, docs in datasets.items():
            results[name][model] = evaluate(nlp, docs)
            print(f"{model} on {name}: {results[name][model]}", file=sys.stderr)
        del nlp
        gc.collect()
    print("\n(en_core_web_trf lemmatizes by rule, without modernizing spellings; its POS come from its tags)")
    table(results)


if __name__ == "__main__":
    main()
