#!/usr/bin/env python3
"""
Pick the Project Gutenberg books used for the 1800-1900 and 1900-2000 sentences.

The model is for British English, and Gutenberg records no nationality: only books the
Library of Congress classes as English literature (PR) or British history (DA) are kept.
Gutenberg's catalog has no first-publication dates either, so a book's century is
inferred from its author's lifespan: an author born 1770-1885 who died by 1910 wrote in
the 19th century; one born from 1870 on wrote in the 20th. Translations are left out
(the translator's dates, not the author's, would matter) and so are reference works
(dictionaries, grammars, bibliographies, encyclopedias).

Books are taken round-robin over authors, so that the sample has as many authors as
possible: an author gets a second book only when there are fewer authors than books
needed (the 20th century has fewer British authors in Gutenberg).

Writes raw/gutenberg/selected.tsv and raw/gutenberg/rsync_files.txt, then fetch the
texts from the official mirror with:
    rsync -a --files-from=raw/gutenberg/rsync_files.txt aleph.gutenberg.org::gutenberg-epub/ raw/gutenberg/texts/
"""

import csv
import random
import re
from collections import defaultdict
from pathlib import Path

CATALOG = Path("raw/gutenberg/pg_catalog.csv")
BOOKS_PER_PERIOD = 650
BOOKS_PER_AUTHOR = 5
BRITISH_LOCC = ("PR", "DA")  # English literature, history of Great Britain
# Library of Congress classes for reference works: English language, bibliography, encyclopedias, dictionaries
EXCLUDED_LOCC = ("PE", "Z", "AE", "AG", "AY")
EXCLUDED_ROLES = ("Translator",)


def first_author(authors):
    """The first author without a role (editor, illustrator...), with birth and death years."""
    for part in authors.split("; "):
        if "[" in part:
            continue
        match = re.search(r"(\d{3,4})\??-(\d{3,4})\??\s*$", part)
        if match:
            name = part[: match.start()].strip().rstrip(",")
            return name, int(match[1]), int(match[2])
        return None
    return None


def period_of(birth, death):
    if 1770 <= birth <= 1885 and death <= 1910:
        return "1800-1900"
    if birth >= 1870:
        return "1900-2000"
    return None


def main():
    random.seed(0)
    by_period = defaultdict(lambda: defaultdict(list))
    with open(CATALOG, encoding="utf8") as f:
        for row in csv.DictReader(f):
            if row["Type"] != "Text" or row["Language"] != "en":
                continue
            if any(role in row["Authors"] for role in EXCLUDED_ROLES):
                continue
            loccs = [locc.strip() for locc in row["LoCC"].split(";") if locc.strip()]
            if any(locc.startswith(EXCLUDED_LOCC) for locc in loccs):
                continue
            if not any(locc.startswith(BRITISH_LOCC) for locc in loccs):
                continue
            author = first_author(row["Authors"])
            if author is None:
                continue
            name, birth, death = author
            period = period_of(birth, death)
            if period is None:
                continue
            by_period[period][name].append({"id": row["Text#"], "title": row["Title"].replace("\n", " "),
                                            "author": name, "birth": birth, "death": death, "period": period})

    selected = []
    for period, authors in sorted(by_period.items()):
        # Round-robin over authors so prolific ones don't crowd out the rest
        queues = []
        for name in sorted(authors):
            books = authors[name][:]
            random.shuffle(books)
            queues.append(books[:BOOKS_PER_AUTHOR])
        random.shuffle(queues)
        chosen = []
        while len(chosen) < BOOKS_PER_PERIOD and any(queues):
            for queue in queues:
                if queue and len(chosen) < BOOKS_PER_PERIOD:
                    chosen.append(queue.pop())
        print(f"{period}: {len(authors)} authors, {len(chosen)} books selected")
        selected.extend(chosen)

    out = Path("raw/gutenberg")
    with open(out / "selected.tsv", "w", encoding="utf8") as f:
        f.write("id\tperiod\tauthor\tbirth\tdeath\ttitle\n")
        for book in selected:
            f.write(f"{book['id']}\t{book['period']}\t{book['author']}\t{book['birth']}\t{book['death']}\t{book['title']}\n")
    with open(out / "rsync_files.txt", "w", encoding="utf8") as f:
        for book in selected:
            f.write(f"{book['id']}/pg{book['id']}.txt\n")


if __name__ == "__main__":
    main()
