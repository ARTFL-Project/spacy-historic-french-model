#!/usr/bin/env python3
"""
Add training sentences for the earliest centuries, whose spelling varies most, from TCP
documents used nowhere yet (neither in the training, test or gold sets), at most 5 per
document so that the new sentences bring as many new spellings as possible.

Ids are numbered from 3,000,000. Writes data/sentences_train2_<period>.jsonl.
"""

import json
import random
from multiprocessing import Pool
from pathlib import Path

from extract_sentences import tcp_doc, tcp_docs

ADDED = {"1500-1600": 10000, "1600-1700": 5000}
PER_DOC = 5
USED = ["data/sentences_train_{p}.jsonl", "data/sentences_test_{p}.jsonl", "gold/gold_1_sentences.jsonl",
        "gold/gold_2_sentences.jsonl", "gold/challenge_sentences.jsonl"]


def used_docs(period):
    docs = set()
    for pattern in USED:
        with open(pattern.format(p=period), encoding="utf8") as f:
            docs.update(json.loads(line)["doc"] for line in f)
    return docs


def main():
    all_docs = tcp_docs()
    next_id = 3_000_000
    for period, wanted in ADDED.items():
        used = used_docs(period)
        docs = [d for d in all_docs[period] if d["doc"] not in used]
        random.Random(f"extend {period}").shuffle(docs)
        sentences = []
        with Pool(16) as pool:
            for doc, doc_sents in pool.imap(tcp_doc, [(d, PER_DOC) for d in docs], chunksize=4):
                for text in doc_sents:
                    sentences.append({"id": next_id, "text": text, "doc": doc["doc"], "year": doc["year"],
                                      "author": doc["author"], "title": doc["title"][:200], "source": doc["source"],
                                      "period": period})
                    next_id += 1
                if len(sentences) >= wanted:
                    pool.terminate()
                    break
        sentences = sentences[:wanted]
        with open(Path("data") / f"sentences_train2_{period}.jsonl", "w", encoding="utf8") as f:
            for sentence in sentences:
                f.write(json.dumps(sentence, ensure_ascii=False) + "\n")
        print(f"{period}: {len(sentences)} new sentences from {len({s['doc'] for s in sentences})} new documents "
              f"({len(docs)} unused documents available)")


if __name__ == "__main__":
    main()
