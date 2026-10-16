# Historic English model
A spaCy model for British English of the 16th to 20th centuries

The model does part-of-speech tagging and lemmatization: coarse POS (Universal Dependencies, `pos_`), Penn Treebank tags (`tag_`) and lemmas in modern British spelling (`lemma_`). It reads texts as printed, with long s, u/v and i/j by position, abbreviation marks and historic spellings (*vnto*, *haue*, *ſaid*, *loueth*, *publick*), and lemmatizes them to the modern word (*unto*, *have*, *say*, *love*, *public*).

It was trained on 65,000 sentences annotated by Gemini 3.8 Flash and then corrected: 20,000 from 1500–1600 and 15,000 from 1600–1700, whose spelling varies most, and 10,000 from each later century.
- **1500–1800:** EEBO-TCP and ECCO-TCP, without the books printed in America. The early centuries take at most 5 sentences per document, from more than 3,000 documents each, to cover as many spellings as possible.
- **1800–2000:** British books from Project Gutenberg (English literature and British history in the Library of Congress classification).
- **Tokenization:** sentences are tokenized with spaCy's English tokenizer, adapted to early modern forms (*'Tis* → *'T* + *is*, *th'*, *I'le* → *I* + *'le*, *do't* → *do* + *'t*). The model labels these tokens.
- **Conventions:** they are documented in [ANNOTATION_GUIDELINES.md](ANNOTATION_GUIDELINES.md). They follow the UD English treebanks (EWT and GUM) wherever these decide a case, with rules of their own for historic forms.
- **Corrections:** a set of rules fixes the annotator's systematic slips; each was checked against the gold sets before use.
- **Long s:** a third of the 1500–1800 training sentences printed with ſ were rewritten with s, so that the model handles editions that normalize it.
- **Encoder:** ModernBERT-base. Version 1.0 used RoBERTa-large; version 2.0 is as accurate, about 1.6 times as fast (1.9 times with flash-attention, see [Installation](#installation)) and less than half the size.

## Performance

| Test set | POS | TAG | Lemma |
|----------|-----|-----|-------|
| Gold set 1 (100 sentences) | 98.42% | 98.42% | 97.10% |
| Gold set 2 (100 sentences) | 98.01% | 97.76% | 97.28% |
| Challenge set (39 sentences of rare historic forms) | 96.85% | 96.72% | 94.11% |
| Held-out test set (4,996 sentences) | 98.50% | 98.25% | 97.90% |
| 1500-1600 | 97.80% | 97.68% | 95.64% |
| 1600-1700 | 98.41% | 98.30% | 97.86% |
| 1700-1800 | 98.59% | 98.44% | 98.48% |
| 1800-1900 | 98.92% | 98.52% | 98.86% |
| 1900-2000 | 98.80% | 98.31% | 98.78% |
| Penn historical corpora, 1500–1914 (5.1 million words annotated by hand) | 98.43% | – | 98.30% |
| Penn historical corpora, 1500–1600 | 97.26% | – | 94.74% |
| UD English EWT test (modern web English) | 95.40% | 93.72% | 94.08% |

For comparison, spaCy's `en_core_web_trf` scores 88.53% / 89.58% / 86.05% on gold set 1, 76.59% / 76.41% / 68.75% on the 16th-century test set, and 92.59% (POS) and 91.67% (lemmas) on the Penn corpora (82.62% and 73.85% for 1500–1600).

How to read these figures:
- **Held-out test set:** annotated like the training data, so its scores measure agreement with the annotating model, not accuracy. Its sentences come from documents not used for training.
- **Gold sets:** 20 sentences per century each. They were made by adjudicating the annotations of four models (Gemini 3.8 Flash, Gemini 4, Claude Sonnet 5.5, Claude Opus 5.5) against the guidelines, and have not yet been corrected by hand, so errors shared by all the models may remain. Their documents were not used for training.
- **Penn historical corpora:** PPCEME and PPCMBE2 ([LDC2025T09](https://catalog.ldc.upenn.edu/LDC2025T09)), annotated by hand: the only historic test independent of the annotating model. Their tags are mapped to the POS of these conventions using their parse trees, and their lemmas (from the corpora's lemma patch) likewise; where the mapping cannot decide between two categories, either counts as correct. TAG is not scored, as their tagset does not map onto the Penn Treebank tags. The model tokenizes the text itself; the 0.5% of tokens it splits differently from the corpora are not scored, nor are the lemmas of proper nouns, numbers and punctuation.
- **UD English EWT:** modern American web text, also independent. Its lemmas keep the spelling of the text rather than modernizing it.

## Installation

1. Download the model from the [en-v2.0 release](https://github.com/ARTFL-Project/spacy-historic-models/releases/tag/en-v2.0). Version 1.0, on RoBERTa-large, remains available from the [en-v1.0 release](https://github.com/ARTFL-Project/spacy-historic-models/releases/tag/en-v1.0).

2. Extract the model:
```bash
tar -xzf historic-english-model.tar.gz
```

Then load it (spaCy 3.8, `spacy-transformers` 1.3 and `transformers` 4.48 or 4.49 are required):
```python
import spacy

nlp = spacy.load("historic-english-model")
doc = nlp("'Tis true, thou know'st it not; vnto him hath the Lord giuen ſuch a Publick Charge.")
for token in doc:
    print(token.text, token.pos_, token.tag_, token.lemma_)
```

On a GPU, installing `flash-attn` 2.7.4 makes the model about 15% faster, with the same results. Take the wheel matching your Python, PyTorch and CUDA from the [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1); version 2.8 does not work with transformers 4.49. Install it only where the model runs on the GPU: once installed, it is used on the CPU too, where it fails.

## Rebuilding the model

Run these commands from the `en/` directory.

1. `python extract_sentences.py` extracts candidate sentences per century from the TCP texts and from the Gutenberg books chosen by `select_gutenberg.py`.
2. `python make_gold_sets.py` draws the gold and challenge sentences, `python split_data.py` the training and test sentences, keeping documents apart, and `python extend_data.py` the additional sentences for 1500–1700 from documents not used elsewhere.
3. `python annotate.py gemini-3.8-flash data/sentences_train_<period>.jsonl --output annotations/train_<period> --batch` annotates through OpenRouter's Batch API (likewise `sentences_train2_<period>` into `annotations/train2_<period>`, and the test sentences). The prompt is `ANNOTATION_GUIDELINES.md`.
4. `python fix_annotations.py` corrects systematic slips and writes `training_data/` and `test_data/`. Run it with `--score` to see each rule's effect on the gold sets. `python ud_conventions.py check <file>` lists departures from the UD English conventions.
5. `python convert_to_spacy.py` writes the spaCy files to `training_space/data/`, holding out 5% of each century, by document, for dev, and rewrites a third of the early training sentences without the long s.
6. `cd training_space && python -m spacy train config.cfg --code ../historic_tokenizer.py --output <output_dir> --gpu-id 0`
7. `python ../shared/package_model.py <output_dir>/model-best historic-english-model` removes the training-time tokenizer hook, so that the model loads without this repository's code. It also adds a rule for *cannot*, which the tokenizer splits into *can* + *not*: some encoders see a single word there and give both tokens the same label. And it lets transformers choose the fastest attention available when the model is loaded.
8. `python evaluate_models.py` scores the model and `en_core_web_trf` on the gold sets, the test sets and UD English EWT.

Gold sentences are in `gold/readable/*.txt` for hand correction; `python readable_gold.py to-jsonl gold/readable/gold_1.txt gold/gold_1.jsonl` converts them back.
