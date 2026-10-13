# spacy-historic-french-model
A SpaCy French model for 16th-20th century French

This model was trained on a corpus of 50,000 sentences derived from ARTFL-Frantext (10,000 per century), automatically annotated by GPT4. The annotations were then corrected: contractions are split the way spaCy's tokenizer splits them (`c'est` → `c'` + `est`, `dit-il` → `dit` + `-il`), and label fixes checked against 100 hand-validated sentences were applied.

The model only supports lemmatization, and coarse and fine-grained POS tagging (the pos_ and tag_ attributes in SpaCy). Although GPT4 provided NER tagging, these were inconsistent, and we did not have a large enough sample to train SpaCy on it.

## Performance

| Test set | POS | TAG | Lemma |
|----------|-----|-----|-------|
| Hand-validated (100 sentences) | 95.88% | 97.55% | 97.64% |
| Held-out test set (5,000 sentences) | 97.60% | 96.85% | 96.87% |
| 1500-1600 | 96.65% | 95.95% | 94.46% |
| 1600-1700 | 97.59% | 96.54% | 96.81% |
| 1700-1800 | 97.94% | 97.12% | 97.65% |
| 1800-1900 | 97.96% | 97.43% | 97.88% |
| 1900-2000 | 97.90% | 97.31% | 97.69% |

## Installation

1. Download the model from the [latest release](https://github.com/ARTFL-Project/spacy-historic-french-model/releases/latest)

2. Extract the model:
```bash
tar -xzf historic-french-model.tar.gz
```

Then load the model (ensure you have the correct path to the historic-french-model folder):
```python
import spacy

# Load the model from the extracted directory
nlp = spacy.load("historic-french-model")

# Process text
text = "Votre texte en français"
doc = nlp(text)
```

## Rebuilding the model

1. `python prepare_annotations.py` rebuilds `training_data/` and `test_data/` from the original GPT4 annotations (commit `6df8d70`), splitting contractions and applying the label fixes.
2. `python convert_annot_to_spacy_bin.py` writes the spaCy training files to `training_space/data/`, holding out 5% of each century for dev.
3. `cd training_space && python -m spacy train config.cfg --output <output_dir> --gpu-id 0`
4. `python spacy_model_evaluator.py` scores the model and `fr_dep_news_trf` on the Sequoia treebank, the hand-validated sentences and the test sets.

Hand-validated sentences are edited in `validated_test_data_readable/txt/` and converted with `txt_to_jsonl.py` into `validated_test_data/validated_jsonl/`.
