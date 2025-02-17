# spacy-historic-french-model
A SpaCy French model for 16th-21st century French

This model was trained on a corpus of 50,000 sentences derived from ARTFL-Frantext, and automatically annotated by GPT4. 

The model only supports lemmatization, and corse and fine-grained POS tagging (the pos_ and tag_ attributes in SpaCy). Although GPT4 provided NER tagging, these were inconsistent, and we did not have a large enough sample to train SpaCy on it.
