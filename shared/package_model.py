#!/usr/bin/env python3
"""
Make a trained pipeline loadable without this project's code: the historic tokenizer's rules
are already saved in the pipeline's tokenizer, so the training-time callback that set them
up is removed from its config. English pipelines also get a rule for *cannot*, and the
transformer is left to choose its attention implementation when the pipeline is loaded.

    cd fr && python ../shared/package_model.py training_space/models/model-best historic-french-model
"""

import shutil
import sys
from pathlib import Path

import spacy
import srsly
from spacy.util import load_config

# The tokenizer splits cannot into can + not. Where the transformer has a single subword for
# cannot, both tokens get the same vector and the same tag, so they are set by rule.
CANNOT = [{"LOWER": "can", "SPACY": False}, {"LOWER": "not"}]
CANNOT_ATTRS = [{"POS": "AUX", "TAG": "MD", "LEMMA": "can"}, {"POS": "PART", "TAG": "RB", "LEMMA": "not"}]


def add_cannot_rule(target):
    nlp = spacy.load(target)
    ruler = nlp.add_pipe("attribute_ruler", name="cannot_rule", last=True)
    for index, attrs in enumerate(CANNOT_ATTRS):
        ruler.add([CANNOT], attrs, index=index)
    nlp.to_disk(target)


def unset_attention(target):
    """The saved transformer config marks its attention implementation as chosen, which makes
    transformers fall back to the slow eager attention when the pipeline is loaded. Without the
    mark, it uses flash-attention if installed, otherwise PyTorch's SDPA."""
    path = target / "transformer" / "model"
    if not path.exists():
        return
    msg = srsly.msgpack_loads(path.read_bytes())
    for shims in msg["shims"]:
        for i, shim_bytes in enumerate(shims):
            shim = srsly.msgpack_loads(shim_bytes)
            shim["config"].pop("_attn_implementation_autoset", None)
            shims[i] = srsly.msgpack_dumps(shim)
    path.write_bytes(srsly.msgpack_dumps(msg))


def main(source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    config = load_config(target / "config.cfg", interpolate=False)
    config["nlp"]["after_creation"] = None
    config.to_disk(target / "config.cfg")
    if config["nlp"]["lang"] == "en":
        add_cannot_rule(target)
    unset_attention(target)
    print(f"{target}: ready")


if __name__ == "__main__":
    main(*sys.argv[1:3])
