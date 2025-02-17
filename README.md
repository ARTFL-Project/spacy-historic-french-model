# spacy-historic-french-model
A SpaCy French model for 16th-20th century French

This model was trained on a corpus of 50,000 sentences derived from ARTFL-Frantext, and automatically annotated by GPT4. 

The model only supports lemmatization, and corse and fine-grained POS tagging (the pos_ and tag_ attributes in SpaCy). Although GPT4 provided NER tagging, these were inconsistent, and we did not have a large enough sample to train SpaCy on it.

## Installation

1. Download the model from the [latest release](https://github.com/ARTFL-Project/spacy-historic-french-model/releases/latest)

2. Extract the model:
```bash
tar -xzf historic-french-model.tar.gz
```

The load the model (ensure you have the correct path to the historic-french-model folder):
```python
import spacy

# Load the model from the extracted directory
nlp = spacy.load("historic-french-model")

# Process text
text = "Votre texte en français"
doc = nlp(text)
```
