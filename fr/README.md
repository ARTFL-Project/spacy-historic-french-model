# Historic French model
A SpaCy French model for 16th-20th century French

This model was trained on a corpus of 50,000 sentences derived from ARTFL-Frantext (10,000 per century), automatically annotated by GPT4. The annotations were then corrected:
- contractions are split the way spaCy's tokenizer splits them (`c'est` → `c'` + `est`, `dit-il` → `dit` + `-il`);
- about 40 label rules bring the annotations in line with UD French conventions for POS and lemmas and with the French Treebank for TAG. Each rule was checked against hand-corrected sentences. The conventions are documented in [ANNOTATION_GUIDELINES.md](ANNOTATION_GUIDELINES.md).

The model only supports lemmatization, and coarse and fine-grained POS tagging (the pos_ and tag_ attributes in SpaCy). Although GPT4 provided NER tagging, these were inconsistent, and we did not have a large enough sample to train SpaCy on it.

## Performance

| Test set | POS | TAG | Lemma |
|----------|-----|-----|-------|
| Hand-corrected gold (99 sentences) | 98.02% | 97.57% | 98.03% |
| Second gold set (100 sentences) | 97.63% | 96.66% | 97.76% |
| Held-out test set (4,999 sentences) | 98.50% | 97.85% | 97.53% |
| 1500-1600 | 97.52% | 96.93% | 95.05% |
| 1600-1700 | 98.59% | 97.67% | 97.41% |
| 1700-1800 | 98.88% | 98.20% | 98.27% |
| 1800-1900 | 98.81% | 98.27% | 98.59% |
| 1900-2000 | 98.75% | 98.27% | 98.49% |
| Sequoia (modern French, UD) | 95.39% | – | 93.95% |

Both gold sets contain 20 sentences per century, follow the conventions above, and were not used for training. The held-out test set is annotated by GPT4 with the same corrections as the training data.

## Installation

1. Download the model from the [v6.0 release](https://github.com/ARTFL-Project/spacy-historic-models/releases/tag/v6.0)

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

Run these commands from the `fr/` directory.

1. `python prepare_annotations.py` rebuilds `training_data/` and `test_data/` from the original GPT4 annotations (commit `6df8d70`), splitting and merging tokens to match the tokenizer and applying the label fixes.
2. `python convert_annot_to_spacy_bin.py` writes the spaCy training files to `training_space/data/`, holding out 5% of each century for dev.
3. `cd training_space && python -m spacy train config.cfg --code ../historic_tokenizer.py --output <output_dir> --gpu-id 0`
4. `python ../shared/package_model.py <output_dir>/model-best <model_dir>` makes the trained model load without `historic_tokenizer.py`.
5. `python spacy_model_evaluator.py` scores the model and `fr_dep_news_trf` on the Sequoia treebank, both gold sets and the test sets.

The tokenizer is spaCy's French tokenizer adapted to historic printing (`historic_tokenizer.py`): dashes glued to words, capitalized and historic spellings of spaCy's exceptions (*Peut-être*, *peut-estre*, *aujourd'huy*), and clitics missing from spaCy's list (*as-tu*, *a-t'il*).

Gold sentences are edited in `validated_test_data_readable/txt/` (`*_test_set.txt` and `*_test_set_2.txt`) and converted with `txt_to_jsonl.py` into `validated_test_data/validated_jsonl/`.
