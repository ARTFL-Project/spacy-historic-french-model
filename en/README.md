# spacy-historic-english-model
A spaCy model for British English of the 16th to 20th centuries

The model does part-of-speech tagging and lemmatization: coarse POS (Universal Dependencies, `pos_`), Penn Treebank tags (`tag_`) and lemmas in modern British spelling (`lemma_`). It reads texts as printed, with long s, u/v and i/j by position, abbreviation marks and historic spellings (*vnto*, *haue*, *ſaid*, *loueth*, *publick*), and lemmatizes them to the modern word (*unto*, *have*, *say*, *love*, *public*).

It was trained on 65,000 sentences annotated by Gemini 3.8 Flash and then corrected: 20,000 from 1500–1600 and 15,000 from 1600–1700, whose spelling varies most, and 10,000 from each later century.
- **1500–1800:** EEBO-TCP and ECCO-TCP, without the books printed in America. The early centuries take at most 5 sentences per document, from more than 3,000 documents each, to cover as many spellings as possible.
- **1800–2000:** British books from Project Gutenberg (English literature and British history in the Library of Congress classification).
- **Tokenization:** sentences are tokenized with spaCy's English tokenizer, adapted to early modern forms (*'Tis* → *'T* + *is*, *th'*, *I'le* → *I* + *'le*, *do't* → *do* + *'t*). The model labels these tokens.
- **Conventions:** they are documented in [ANNOTATION_GUIDELINES.md](ANNOTATION_GUIDELINES.md). They follow the UD English treebanks (EWT and GUM) wherever these decide a case, with rules of their own for historic forms.
- **Corrections:** a set of rules fixes the annotator's systematic slips; each was checked against the gold sets before use.
- **Long s:** a third of the 1500–1800 training sentences printed with ſ were rewritten with s, so that the model handles editions that normalize it.

## Performance

| Test set | POS | TAG | Lemma |
|----------|-----|-----|-------|
| Gold set 1 (100 sentences) | 98.48% | 98.31% | 97.07% |
| Gold set 2 (100 sentences) | 98.26% | 98.12% | 97.25% |
| Challenge set (39 sentences of rare historic forms) | 96.99% | 96.79% | 94.24% |
| Held-out test set (4,996 sentences) | 98.53% | 98.27% | 97.88% |
| 1500-1600 | 97.91% | 97.77% | 95.54% |
| 1600-1700 | 98.37% | 98.40% | 97.82% |
| 1700-1800 | 98.69% | 98.48% | 98.46% |
| 1800-1900 | 98.92% | 98.37% | 98.90% |
| 1900-2000 | 98.81% | 98.35% | 98.84% |
| UD English EWT test (modern web English) | 95.62% | 93.92% | 94.38% |

For comparison, spaCy's `en_core_web_trf` scores 88.53% / 89.58% / 86.05% on gold set 1 and 76.59% / 76.41% / 68.75% on the 16th-century test set.

How to read these figures:
- **Held-out test set:** annotated like the training data, so its scores measure agreement with the annotating model, not accuracy. Its sentences come from documents not used for training.
- **Gold sets:** 20 sentences per century each. They were made by adjudicating the annotations of four models (Gemini 3.8 Flash, Gemini 4, Claude Sonnet 5.5, Claude Opus 5.5) against the guidelines, and have not yet been corrected by hand, so errors shared by all the models may remain. Their documents were not used for training.
- **UD English EWT:** the only fully independent test. It is modern American web text, and its lemmas keep the spelling of the text rather than modernizing it.

## Installation

1. Download the model from the [latest release](https://github.com/ARTFL-Project/spacy-historic-english-model/releases/latest)

2. Extract the model:
```bash
tar -xzf historic-english-model.tar.gz
```

Then load it (spaCy 3.8 and `spacy-transformers` are required):
```python
import spacy

nlp = spacy.load("historic-english-model")
doc = nlp("'Tis true, thou know'st it not; vnto him hath the Lord giuen ſuch a Publick Charge.")
for token in doc:
    print(token.text, token.pos_, token.tag_, token.lemma_)
```

## Rebuilding the model

1. `python extract_sentences.py` extracts candidate sentences per century from the TCP texts and from the Gutenberg books chosen by `select_gutenberg.py`.
2. `python make_gold_sets.py` draws the gold and challenge sentences, `python split_data.py` the training and test sentences, keeping documents apart, and `python extend_data.py` the additional sentences for 1500–1700 from documents not used elsewhere.
3. `python annotate.py gemini-3.8-flash data/sentences_train_<period>.jsonl --output annotations/train_<period> --batch` annotates through OpenRouter's Batch API (likewise `sentences_train2_<period>` into `annotations/train2_<period>`, and the test sentences). The prompt is `ANNOTATION_GUIDELINES.md`.
4. `python fix_annotations.py` corrects systematic slips and writes `training_data/` and `test_data/`. Run it with `--score` to see each rule's effect on the gold sets. `python ud_conventions.py check <file>` lists departures from the UD English conventions.
5. `python convert_to_spacy.py` writes the spaCy files to `training_space/data/`, holding out 5% of each century, by document, for dev, and rewrites a third of the early training sentences without the long s.
6. `cd training_space && python -m spacy train config.cfg --code ../historic_tokenizer.py --output <output_dir> --gpu-id 0`
7. `python package_model.py <output_dir>/model-best historic-english-model` removes the training-time tokenizer hook, so that the model loads without this repository's code.
8. `python evaluate_models.py` scores the model and `en_core_web_trf` on the gold sets, the test sets and UD English EWT.

Gold sentences are in `gold/readable/*.txt` for hand correction; `python readable_gold.py to-jsonl gold/readable/gold_1.txt gold/gold_1.jsonl` converts them back.
