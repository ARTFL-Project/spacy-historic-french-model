import random

import jsonlines
import spacy
from spacy.tokens import Doc, DocBin
from tqdm import tqdm

TRAIN_DB = DocBin()
DEV_DB = DocBin()
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
    if text.endswith("'") or next_text.startswith("'"):
        return False

    # No space around hyphens
    if text.endswith("-") or next_text.startswith("-"):
        return False

    # Default: include space
    return True


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


def fix_pos(pos: str) -> str:
    if pos == "SYM":
        return "X"
    return pos


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


def fix_lemma(lemma: str) -> str:
    if lemma == "a+le":
        return "au"
    elif lemma == "a+les":
        return "aux"
    elif lemma == "de+le" or lemma == "de + le":
        return "du"
    elif lemma == "de+les":
        return "des"
    return lemma


def convert_to_spacy(examples, sent_type="train"):
    for example in tqdm(examples, total=len(examples)):
        tokens = example["annotations"]["tokens"]
        try:
            words = [t["text"] for t in tokens]
        except KeyError:
            continue

        # Calculate spaces
        spaces = []
        for i in range(len(words)):
            if i < len(words) - 1:
                spaces.append(should_have_space_after(words[i], words[i + 1]))
            else:
                spaces.append(False)  # No space after last token

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
        except:
            continue

        if sent_type == "train":
            TRAIN_DB.add(doc)
        else:
            DEV_DB.add(doc)


if __name__ == "__main__":
    training_sentences = []
    dev_sentences = []
    for period in [(1500, 1600), (1600, 1700), (1700, 1800), (1800, 1900), (1900, 2000)]:
        with jsonlines.open(f"annotations_{period[0]}-{period[1]}.jsonl") as reader:
            sentences = list(reader)
            training_sents = augment_rare_categories(sentences[: int(0.85 * len(sentences))])
            training_sentences.extend(training_sents)
            dev_sents = sentences[int(0.85 * len(sentences)) :]
            dev_sentences.extend(dev_sents)

    random.shuffle(training_sentences)
    random.shuffle(dev_sentences)

    convert_to_spacy(training_sentences, sent_type="train")
    convert_to_spacy(dev_sentences, sent_type="dev")
    TRAIN_DB.to_disk("data/train.spacy")
    DEV_DB.to_disk("data/dev.spacy")
