#!/usr/bin/env python3
"""
Rebuild the corrected annotation files from the original GPT-4 annotations.

The originals are read from git (ORIGINAL_COMMIT) so that every run starts from the
same untouched data. Two passes are applied to each sentence:

1. Retokenization: tokens that the spaCy French tokenizer splits (elisions such as
   "c'est" or "qu'il", inverted clitics such as "dit-il") are split into the same
   pieces, keeping the surface form of each piece. Fixed expressions whose lemma is
   a single unit ("d'abord", "peut-être", "aujourd'hui") are left whole.
2. Label fixes. Each rule was scored against the hand-validated sentences in
   validated_test_data/validated_jsonl; rules that made the data worse are disabled.

Writes training_data/annotations_*.jsonl and test_data/annotations_*_test_set.jsonl.
"""

import json
import re
import subprocess
from collections import Counter, defaultdict

import spacy
from spacy.tokens import Doc
from spacy.training import Alignment

ORIGINAL_COMMIT = "6df8d70"
PERIODS = ["1500-1600", "1600-1700", "1700-1800", "1800-1900", "1900-2000"]
TOKENIZER_PATH = "historic-french-model/tokenizer"

# Contracted preposition + article: lemmatize as the preposition, like fr_dep_news_trf
CONTRACTED_LEMMAS = {
    "de le": "de",
    "de les": "de",
    "de+le": "de",
    "de+les": "de",
    "à le": "à",
    "à les": "à",
    "à+le": "à",
    "à+les": "à",
    "en les": "en",
}

# Rules tried on 2026-10-12 and disabled after scoring them against the hand-validated
# sentences (fixes / breaks):
#   être/avoir VERB -> AUX everywhere:      22 / 29  (main-verb avoir and être are VERB)
#   que SCONJ -> PRON before a pronoun:     15 / 24  (mostly real conjunctions)
#   pas NOUN -> ADV:                        all real nouns ("un faux pas", "à pas lents")
AUX_ONLY_WHEN_AUX_DEP = True

# Pieces other than the head of a split token must be function words
CLOSED_CLASS = {"PRON", "DET", "ADP", "SCONJ", "CCONJ", "AUX", "PART"}


def read_original(path):
    output = subprocess.run(
        ["git", "show", f"{ORIGINAL_COMMIT}:{path}"], capture_output=True, text=True, check=True
    ).stdout
    return [json.loads(line) for line in output.splitlines()]


def should_have_space_after(text: str, next_text: str = None) -> bool:
    """Determine if a token should have a space after it in French."""
    if next_text is None:
        return False

    # No space before these characters
    if next_text in [",", ".", ")", "]", "}", "»", ";", ":", "!", "?"]:
        return False

    # No space after these characters
    if text in ["(", "[", "{", "«"]:
        return False

    # No space around apostrophes
    if text.endswith(("'", "’")) or next_text.startswith(("'", "’")):
        return False

    # No space around hyphens
    if text.endswith("-") or next_text.startswith("-"):
        return False

    # Default: include space
    return True


def token_spaces(sentence_text, words):
    """Whether each token is followed by a space: read from the sentence text when the
    tokens match it, otherwise guessed with French spacing rules."""
    spaces = []
    position = 0
    for word in words:
        while position < len(sentence_text) and sentence_text[position].isspace():
            position += 1
        if not sentence_text.startswith(word, position):
            break
        position += len(word)
        spaces.append(position < len(sentence_text) and sentence_text[position].isspace())
    else:
        return spaces
    return [should_have_space_after(word, words[i + 1] if i + 1 < len(words) else None) for i, word in enumerate(words)]


def norm(text):
    return text.lower().replace("’", "'")


def build_label_table(sentences):
    """Label counts of each token form, used to label the pieces of split tokens."""
    by_text_lemma = defaultdict(Counter)
    by_text = defaultdict(Counter)
    for sentence in sentences:
        for token in sentence["annotations"]["tokens"]:
            if "text" not in token:
                continue
            text = norm(token["text"])
            by_text_lemma[(text, token["lemma"].lower())][(token["pos"], token["tag"], token["dep"])] += 1
            by_text[text][(token["pos"], token["tag"], token["lemma"], token["dep"])] += 1
    return by_text_lemma, by_text


def lookup_keys(piece):
    # "-t" is usually annotated as "-t-", and "-il" as "il"
    text = norm(piece)
    return (text, text + "-", text.strip("-"))


def most_common(counts, prefer_pronoun):
    if prefer_pronoun:
        pronouns = Counter({labels: n for labels, n in counts.items() if labels[0] == "PRON"})
        counts = pronouns or counts
    return counts.most_common(1)[0][0]


def prefers_pronoun(index, piece):
    # Clitics after the verb or after an elided pronoun: "dites-le", "s'en", "n'y"
    return index > 0 and ((piece.startswith("-") and len(piece) > 1) or norm(piece) in ("en", "y"))


def piece_labels(piece, table, lemma=None, prefer_pronoun=False):
    by_text_lemma, by_text = table
    for key in lookup_keys(piece):
        if lemma is not None and (key, lemma.lower()) in by_text_lemma:
            pos, tag, dep = most_common(by_text_lemma[(key, lemma.lower())], prefer_pronoun)
            return {"pos": pos, "tag": tag, "lemma": lemma, "dep": dep}
        if key in by_text:
            pos, tag, usual_lemma, dep = most_common(by_text[key], prefer_pronoun)
            return {"pos": pos, "tag": tag, "lemma": lemma if lemma is not None else usual_lemma, "dep": dep}
    return None


def label_share(piece, lemma, token, table):
    """How often this piece, with this lemma, carries the token's POS and tag."""
    by_text_lemma, _ = table
    for key in lookup_keys(piece):
        counts = by_text_lemma.get((key, lemma.lower()))
        if counts:
            matching = sum(n for (pos, tag, _), n in counts.items() if (pos, tag) == (token["pos"], token["tag"]))
            return matching / sum(counts.values())
    return 0.0


def split_token(token, pieces, table):
    """Return the tokens replacing `token`, or None to keep it whole."""
    lemma_parts = [part for part in re.split(r"\s+|\+", token["lemma"].strip()) if part]

    if len(lemma_parts) == len(pieces):
        # The lemma names each word ("ce être"): give each piece its lemma. The token's
        # own labels go to the piece that most often carries them ("qu'" in a relative
        # "qu'elle"); the other pieces get their usual labels.
        shares = [label_share(piece, lemma, token, table) for piece, lemma in zip(pieces, lemma_parts)]
        head = shares.index(max(shares)) if max(shares) > 0 else None
        new_tokens = []
        for i, (piece, lemma) in enumerate(zip(pieces, lemma_parts)):
            guess = piece_labels(piece, table, lemma, prefers_pronoun(i, piece))
            if i == head or guess is None:
                new_tokens.append({**token, "text": piece, "lemma": lemma})
            else:
                new_tokens.append({**token, **guess, "text": piece})
        return new_tokens, "lemma_per_piece"

    # The lemma is that of one piece ("dit-il" -> "dire"): that piece keeps the token's
    # labels, as long as the other pieces are function words.
    guesses = [piece_labels(piece, table, prefer_pronoun=prefers_pronoun(i, piece)) for i, piece in enumerate(pieces)]
    heads = [i for i, guess in enumerate(guesses) if guess and guess["lemma"].lower() == token["lemma"].lower()]
    if len(heads) != 1:
        return None, None
    head = heads[0]
    if any(guess is None or guess["pos"] not in CLOSED_CLASS for i, guess in enumerate(guesses) if i != head):
        return None, None
    new_tokens = [
        {**token, "text": piece} if i == head else {**token, **guess, "text": piece}
        for i, (piece, guess) in enumerate(zip(pieces, guesses))
    ]
    return new_tokens, "head_piece"


def make_tokenizer():
    nlp = spacy.blank("fr")
    nlp.tokenizer.from_disk(TOKENIZER_PATH)
    return nlp.tokenizer


def retokenize(sentence, tokenizer, table, stats):
    """Split tokens the way the tokenizer splits them in the sentence, as during training."""
    tokens = sentence["annotations"]["tokens"]
    words = [t["text"] for t in tokens]
    doc = Doc(tokenizer.vocab, words=words, spaces=token_spaces(sentence["text"], words))
    predicted = [t.text for t in tokenizer(doc.text)]
    alignment = Alignment.from_strings([t.text for t in doc], predicted)
    new_tokens = []
    for i, token in enumerate(tokens):
        targets = [int(j) for j in alignment.x2y[i]]
        if len(targets) > 1 and all(list(alignment.y2x[j]) == [i] for j in targets):
            pieces = [predicted[j] for j in targets]
            replacement, how = split_token(token, pieces, table)
            if replacement is not None:
                new_tokens.extend(replacement)
                stats[f"split ({how})"] += 1
                stats[f"split: {norm(token['text'])}"] += 1
                continue
            stats["kept whole"] += 1
        new_tokens.append(token)
    return new_tokens


def is_aux_dep(dep):
    dep = dep.lower()
    return dep.startswith("aux") or dep == "cop"


def fix_labels(token, stats):
    token = dict(token)
    if token["lemma"] in CONTRACTED_LEMMAS:
        token["lemma"] = CONTRACTED_LEMMAS[token["lemma"]]
        stats["lemma: contracted article -> preposition"] += 1
    if token["lemma"] == "pas" and token["pos"] == "PART":
        token["pos"] = "ADV"
        stats["pos: pas PART -> ADV"] += 1
    if token["lemma"] in ("être", "avoir") and token["pos"] == "VERB":
        if is_aux_dep(token["dep"]) or not AUX_ONLY_WHEN_AUX_DEP:
            token["pos"] = "AUX"
            stats["pos: être/avoir VERB -> AUX (aux/cop dep)"] += 1
    return token


def process_sentence(sentence, tokenizer, table, stats):
    tokens = sentence["annotations"]["tokens"]
    if not tokens or not all("text" in t and "lemma" in t for t in tokens):
        stats["malformed sentences left as-is"] += 1
        return sentence
    tokens = retokenize(sentence, tokenizer, table, stats)
    tokens = [fix_labels(t, stats) for t in tokens]
    return {**sentence, "annotations": {**sentence["annotations"], "tokens": tokens}}


def main():
    tokenizer = make_tokenizer()
    files = [(f"training_data/annotations_{p}.jsonl", read_original(f"training_data/annotations_{p}.jsonl")) for p in PERIODS]
    table = build_label_table(sentence for _, sentences in files for sentence in sentences)
    files += [
        (f"test_data/annotations_{p}_test_set.jsonl", read_original(f"test_data/annotations_{p}_test_set.jsonl"))
        for p in PERIODS
    ]

    total = Counter()
    for path, sentences in files:
        stats = Counter()
        fixed = [process_sentence(sentence, tokenizer, table, stats) for sentence in sentences]
        with open(path, "w", encoding="utf-8") as f:
            for sentence in fixed:
                f.write(json.dumps(sentence, ensure_ascii=False) + "\n")
        summary = {k: v for k, v in stats.items() if not k.startswith("split: ")}
        print(f"{path}: {len(fixed)} sentences, {summary}")
        total.update(stats)

    print("\nMost frequent splits:")
    for key, count in Counter({k: v for k, v in total.items() if k.startswith("split: ")}).most_common(25):
        print(f"  {key[7:]:20s} {count}")


if __name__ == "__main__":
    main()
