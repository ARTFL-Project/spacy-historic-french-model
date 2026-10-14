#!/usr/bin/env python3
"""
Make a trained pipeline loadable without this project's code: the historic tokenizer's rules
are already saved in the pipeline's tokenizer, so the training-time callback that set them
up is removed from its config.

    python package_model.py training_space/models/model-best historic-french-model
"""

import shutil
import sys
from pathlib import Path

from spacy.util import load_config


def main(source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    config = load_config(target / "config.cfg", interpolate=False)
    config["nlp"]["after_creation"] = None
    config.to_disk(target / "config.cfg")
    print(f"{target}: ready")


if __name__ == "__main__":
    main(*sys.argv[1:3])
