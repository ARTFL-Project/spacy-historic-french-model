import argparse
import random
from collections import Counter
from pathlib import Path

import jsonlines
import spacy
from spacy.tokens import Doc, DocBin
from tqdm import tqdm

from prepare_annotations import token_spaces

TRAIN_DB = DocBin()
DEV_DB = DocBin()
TEST_DB = DocBin()
VALIDATED_DB = DocBin()
PERIOD_DB = {f"{i}-{i + 100}": DocBin() for i in range(1500, 2000, 100)}
NLP = spacy.load("fr_dep_news_trf")


def augment_rare_categories(examples, rare_pos=["INTJ"], rare_tags=["VIMP"]):
    """Duplicate examples containing rare POS tags"""
    augmented = []
    added = 0
    total = len(examples)
    for example in examples:
        # Add original example
        augmented.append(example)

        # We only want to add at most 1.5% of the total examples
        if added > total * 0.015:
            continue

        # Check if example contains rare POS
        tokens = example["annotations"]["tokens"]
        if any(token["pos"] in rare_pos for token in tokens):
            # Add it multiple times for rare categories
            augmented.extend([example] * 2)
            added += 2
            continue
        if any(token["tag"] in rare_tags for token in tokens):
            # Add it multiple times for rare categories
            augmented.extend([example] * 2)
            added += 2
            continue
    print(f"Added {added} examples for rare categories")
    return augmented


def standardize_dep_label(dep: str) -> str:
    """Standardize dependency labels to Universal Dependencies format."""
    # First handle ROOT/root case
    if dep.lower() == "root":
        return "ROOT"

    if dep == "VERB" or dep == "adj" or dep == "main":
        return "ROOT"

    # Convert to lowercase for consistency
    dep = dep.lower()

    dep_mapping = {
        # Core arguments
        "dobj": "obj",
        "pobj": "obj",
        "nsubjpass": "nsubj:pass",
        # Auxiliaries
        "auxpass": "aux:pass",
        "aux:caus": "aux",
        "aux:clitic": "aux",
        "aux:refl": "aux",
        "aux:reflex": "aux",
        "aux:tense": "aux",
        "aux:part": "aux",
        # Expletives and reflexives
        "expl:pv": "expl",
        "expl:pass": "expl",
        "expl:comp": "expl",
        "expl:subj": "expl",
        "refl": "expl",
        "reflex": "expl",
        "reflexive": "expl",
        # Compounds and flat structures
        "compound:prt": "compound",
        "flat:title": "flat",
        "foreign": "flat",
        # Obliques
        "obl:temp": "obl:tmod",
        "obl:npmod": "obl:mod",
        "obl:arg": "obl",
        # Cleanup rare/invalid
        "pcomp": "xcomp",
        "acomp": "xcomp",
        "xcomp:pass": "xcomp",
        "xcomp:pred": "xcomp",
        "relcl": "acl:relcl",
        "prep": "case",
        "poss": "nmod:poss",
        "npadvmod": "advmod",
        "prefix": "dep",
        "ref": "dep",
        "list": "dep",
    }

    return dep_mapping.get(dep, dep)


# Fine-grained tags that sometimes appear in the pos field instead of a UD tag
TAG_TO_POS = {
    "DETWH": "DET",
    "ADJWH": "DET",
    "ADVWH": "ADV",
    "PROREL": "PRON",
    "PROWH": "PRON",
    "PRO": "PRON",
    "CLS": "PRON",
    "CLO": "PRON",
    "CLR": "PRON",
    "PONCT": "PUNCT",
    "P": "ADP",
    "P+D": "ADP",
    "P+PRO": "ADP",
    "V": "VERB",
    "VPP": "VERB",
    "VPR": "VERB",
    "VINF": "VERB",
    "VS": "VERB",
    "VIMP": "VERB",
    "VER": "VERB",  # typo in the hand-validated 1700-1800 sentences
}


def fix_pos(pos: str) -> str:
    if pos == "SYM":
        return "X"
    return TAG_TO_POS.get(pos, pos)


def fix_tag(tag: str, lemma) -> str:
    if tag == "ADJWH":
        return "DETWH"
    elif tag == "NUM":
        return "DET"
    elif tag == "PREF":
        if lemma == "de" or lemma == "à":
            return "P"
        elif lemma in ("aussi", "ne", "ni", "non", "pas", "plus", "si"):
            return "ADV"
        elif lemma in ("-", "-t", "t", "-in"):
            return "PONCT"
        else:
            return "P"
    elif tag == "PRON":
        return "CLR"
    elif tag == "PART":
        return "ADV"
    return tag


def convert_to_spacy(examples, sent_type="train", period=None):
    skipped = Counter()
    for example in tqdm(examples, total=len(examples)):
        tokens = example["annotations"]["tokens"]
        try:
            words = [t["text"] for t in tokens]
        except KeyError:
            skipped["token without text"] += 1
            continue

        spaces = token_spaces(example["text"], words)

        # try:
        #     ents = [t["ent"] for t in tokens]
        # except KeyError:
        #     continue

        try:
            doc = Doc(
                NLP.vocab,
                words=words,
                spaces=spaces,
                tags=[fix_tag(t["tag"], t["lemma"]) for t in tokens],
                pos=[fix_pos(t["pos"]) for t in tokens],
                lemmas=[t["lemma"] for t in tokens],
                deps=[standardize_dep_label(t["dep"]) for t in tokens],
                # ents=ents,
            )
        except Exception as e:
            skipped[f"{type(e).__name__}: {str(e)[:70]}"] += 1
            continue

        if sent_type == "train":
            TRAIN_DB.add(doc)
        elif sent_type == "dev":
            DEV_DB.add(doc)
        elif sent_type == "test":
            TEST_DB.add(doc)
        elif sent_type == "validated":
            VALIDATED_DB.add(doc)

        if period is not None:
            PERIOD_DB[period].add(doc)

    if skipped:
        print(f"Skipped {sum(skipped.values())} {sent_type} sentences: {dict(skipped)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-fraction", type=float, default=0.05, help="share of each period held out for dev")
    parser.add_argument("--output-dir", default="training_space/data")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_fraction = 1 - args.dev_fraction

    random.seed(0)
    training_sentences = []
    dev_sentences = []
    for period in [(1500, 1600), (1600, 1700), (1700, 1800), (1800, 1900), (1900, 2000)]:
        with jsonlines.open(f"training_data/annotations_{period[0]}-{period[1]}.jsonl") as reader:
            sentences = list(reader)
            split = round(train_fraction * len(sentences))
            training_sents = augment_rare_categories(sentences[:split])
            training_sentences.extend(training_sents)
            dev_sents = sentences[split:]
            dev_sentences.extend(dev_sents)

    random.shuffle(training_sentences)
    random.shuffle(dev_sentences)

    convert_to_spacy(training_sentences, sent_type="train")
    convert_to_spacy(dev_sentences, sent_type="dev")
    TRAIN_DB.to_disk(output_dir / "train.spacy")
    DEV_DB.to_disk(output_dir / "dev.spacy")
    for period in [(1500, 1600), (1600, 1700), (1700, 1800), (1800, 1900), (1900, 2000)]:
        test_file = f"test_data/annotations_{period[0]}-{period[1]}_test_set.jsonl"
        if not Path(test_file).exists():
            print(f"Warning: {test_file} not found, skipping")
            continue
        with jsonlines.open(test_file) as reader:
            convert_to_spacy(list(reader), sent_type="test", period=f"{period[0]}-{period[1]}")
    TEST_DB.to_disk(output_dir / "test.spacy")
    for period, db in PERIOD_DB.items():
        db.to_disk(output_dir / f"{period}_test.spacy")

    # Hand-validated sentences, kept as annotated (no automatic label fixes)
    for validated_file in sorted(Path("validated_test_data/validated_jsonl").glob("*.jsonl")):
        with jsonlines.open(validated_file) as reader:
            convert_to_spacy(list(reader), sent_type="validated")
    VALIDATED_DB.to_disk(output_dir / "validated_test.spacy")