#!/usr/bin/env python3
"""
Extract candidate sentences, per century, from EEBO-TCP / ECCO-TCP (1500-1800) and
Project Gutenberg (1800-2000), for a model of British English: books printed in America
are left out of TCP, and the Gutenberg books are British (see select_gutenberg.py).

Sentences are kept as printed: original spelling, long s, abbreviation marks. A sentence
is kept when it has 20 to 60 tokens, ends with . ? or !, is in English, and contains no
illegible passages (<gap> in TCP) or editorial markup. Only a few sentences are taken from
each document, so that the sample covers as many documents as possible: 5 per TCP document
(more where documents are scarce), 20 per Gutenberg book.

Writes sentences/sentences_<period>.jsonl, shuffled, one {"text", "doc", "year", "author",
"title", "source"} object per line.
"""

import argparse
import csv
import json
import math
import random
import re
import sqlite3
import unicodedata
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import regex
from lxml import etree

from historic_tokenizer import make_nlp

TCP_DB = Path("/var/www/html/philologic5/eebo_ecco_combo/data")
GUTENBERG_DIR = Path("raw/gutenberg")
OUTPUT_DIR = Path("sentences")
TCP_PERIODS = ["1500-1600", "1600-1700", "1700-1800"]
GUTENBERG_PERIODS = ["1800-1900", "1900-2000"]
POOL_SIZE = 20000  # per period: training, test and gold sentences are all drawn from this pool
MIN_TOKENS, MAX_TOKENS = 20, 60
MAX_PER_DOC = 5
GUTENBERG_PER_BOOK = 20  # books are long; ~650 books per century, one per author
MAX_PER_AUTHOR = 25

# TEI elements dropped with their content: metadata, notes, headings, speaker names, tables...
TEI_DROPPED = {"teiHeader", "note", "figure", "figDesc", "speaker", "head", "fw", "label", "table", "bibl",
               "closer", "signed", "salute", "dateline", "trailer", "epigraph", "stage", "opener", "desc",
               "byline", "docImprint", "docAuthor", "titlePart", "titlePage", "argument"}
# divisions that are not running text (div types are normalized: lowercase, no "_")
TEI_DROPPED_DIVS = {"titlepage", "halftitle", "imprimatur", "license", "licence", "colophon", "errata",
                    "tableofcontents", "contents", "index", "dramatispersonae", "publishersadvertisement",
                    "advertisement", "subscribers", "listofsubscribers", "catalogue", "booksellerscatalogue"}
TEI_BLOCKS = {"p", "lg", "item"}
UNREADABLE = "◊"

# Not printed text: illegible letters, editorial brackets, music and astronomical signs, markup
BAD_CHARS = set("•〈〉◊▪♫□*[]{}|_<>=#@^~\\☞☜⊕❧†‡§¶℥") | {chr(c) for c in range(0x1D100, 0x1D200)}
NON_LATIN = regex.compile(r"[\p{Greek}\p{Hebrew}\p{Arabic}\p{Cyrillic}]")
ENGLISH_WORDS = set("""the and of to that in is it a be his which with for not as he they this by but are was all
have we or shall from them will so their our him ye you i thou hath my me unto vnto an haue there then what
when who if no at on upon were may thy thee doth also these those being such her she they your one more
than had would should could do did been has into any other most must great can out up how its them our""".split())
FOREIGN_WORDS = set("""et est ad non cum quod qui quae sed ut per enim sunt esse nec ab ex autem hoc vel quam sicut
le la les des du que qui une dans pour au aux ce il ne pas sont avec sur y el los las""".split())
# do not end a sentence after these
ABBREVIATIONS = set("""mr mrs dr st sr jr esq viz i.e e.g etc &c cap chap ch vol p pp ver v vs no fig ibid
matth mat mar luk ioh joh rom cor gal eph phil col thess tim tit heb iam jam pet rev gen exod exo lev num
deut iosh josh iudg judg sam kin chron neh esth psal ps prov eccles eccl cant isa esa ier jer lam ezek
ezech dan hos mic hab zech mal macc lib sect art gov capt col gen lieut maj rev hon messrs""".split())
SENTENCE_END = regex.compile(r"""([.?!]['"’”)]*)(\s+)(?=['"‘“(]?[\p{Lu}¶])""")

NLP = make_nlp()


def split_sentences(paragraph):
    """Split at . ? ! followed by a capital, unless the period ends an abbreviation or an initial."""
    sentences = []
    start = 0
    for match in SENTENCE_END.finditer(paragraph):
        end = match.end(1)
        last_word = regex.split(r"[\s—–\-('\"‘“]+", paragraph[start:match.start(1)])[-1].lower()
        if match.group(1).startswith(".") and (last_word in ABBREVIATIONS or regex.fullmatch(r"\p{L}", last_word)):
            continue
        sentences.append(paragraph[start:end].strip())
        start = match.end()
    sentences.append(paragraph[start:].strip())
    return [s for s in sentences if s]


def normalize_space(text):
    # NFC: a letter and a combining macron (TCP's abbreviation stroke) become one character where Unicode has one
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
    # stray spaces before punctuation, as left by line breaks
    return re.sub(r" ([,;:.?!])", r"\1", text)


def keep_sentence(sentence):
    """Return the sentence's tokens if it qualifies, else None."""
    if not sentence or not (sentence[0].isalpha() or sentence[0] in "'\"‘“("):
        return None
    if not re.search(r"""[.?!]['"’”)]*$""", sentence):
        return None
    if any(c in BAD_CHARS for c in sentence) or NON_LATIN.search(sentence):
        return None
    if re.search(r"https?:|www\.|gutenberg|e-?book|transcriber", sentence, re.I):
        return None
    tokens = [t.text for t in NLP.tokenizer(sentence)]
    if not MIN_TOKENS < len(tokens) < MAX_TOKENS:
        return None
    words = [t.lower().replace("ſ", "s") for t in tokens if any(c.isalpha() for c in t)]
    if len(words) < MIN_TOKENS * 0.7:
        return None
    if sum(any(c.isdigit() for c in t) for t in tokens) > 3:
        return None
    if sum(w.isupper() and len(w) > 1 for w in tokens) > len(words) * 0.3:
        return None
    english = sum(w in ENGLISH_WORDS for w in words) / len(words)
    foreign = sum(w in FOREIGN_WORDS for w in words) / len(words)
    if english < 0.25 or foreign > 0.06:
        return None
    return tokens


# ---------------------------------------------------------------- EEBO / ECCO (TEI)

def tcp_year(pub_date, create_date, year):
    """Publication year from the TCP imprint date ("1,5,5,7 [i.e. 1557]", "Anno Dom. 1678.")."""
    for text in (pub_date, create_date):
        if not text:
            continue
        if "i.e." in text:
            text = text.split("i.e.", 1)[1]
        text = re.sub(r"(?<=\d)[.,](?=\d)", "", text)
        for match in re.finditer(r"(?<!\d)(1[4-8]\d\d)(?!\d)", text):
            return int(match[1])
    try:
        y = abs(int(year))
        return y if 1450 <= y <= 1830 else None
    except (TypeError, ValueError):
        return None


# printed in America: the model is for British English
AMERICAN_IMPRINT = re.compile(r"Boston|Philadelphia|New[- ]?York|Mass|New[-= ]?England|N\. ?E\.|Germantown|Baltimore|"
                              r"Annapolis|Hartford|New[- ]Haven|Newport, R|Providence|Williamsburg|Charles-?town|"
                              r"Albany|Pa\.|Conn\.|America")


def tcp_period(year):
    if year is None or not 1500 <= year < 1800:
        return None
    start = year // 100 * 100
    return f"{start}-{start + 100}"


def local(element):
    return etree.QName(element).localname if isinstance(element.tag, str) else None


def remove_keep_tail(element, tail=None):
    """Remove an element but keep the text that follows it."""
    tail = element.tail if tail is None else tail
    parent = element.getparent()
    previous = element.getprevious()
    if tail:
        if previous is not None:
            previous.tail = (previous.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
    parent.remove(element)


def tei_blocks(path):
    """Paragraphs, verse groups and list items of a TCP file, as plain text."""
    parser = etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False)
    try:
        root = etree.parse(str(path), parser).getroot()
    except (etree.XMLSyntaxError, OSError):
        return []
    if root is None:
        return []
    for element in list(root.iter()):
        name = local(element)
        if name is None:
            if element.getparent() is not None:
                remove_keep_tail(element)  # comments, processing instructions
        elif name in TEI_DROPPED or (
                name == "div" and element.get("type", "").lower().replace("_", "") in TEI_DROPPED_DIVS):
            remove_keep_tail(element)
        elif name == "gap":
            remove_keep_tail(element, UNREADABLE + (element.tail or ""))
        elif name == "g" and element.get("ref", "") in ("char:EOLhyphen", "char:EOLunhyphen"):
            # a word broken across lines: join the halves
            remove_keep_tail(element, (element.tail or "").lstrip())
        elif name == "lb" and element.get("type") == "hyphenInWord":
            remove_keep_tail(element, (element.tail or "").lstrip())  # ECCO's line-end hyphen
        elif name in ("lb", "pb", "milestone", "cb"):
            remove_keep_tail(element, " " + (element.tail or ""))
    blocks = []
    for element in root.iter():
        name = local(element)
        if name in TEI_BLOCKS and not any(local(a) in TEI_BLOCKS for a in element.iterancestors()):
            blocks.append(normalize_space("".join(element.itertext())))
        elif name == "l" and not any(local(a) in TEI_BLOCKS for a in element.iterancestors()):
            # verse lines outside a line group: a sentence can run over several lines
            previous = element.getprevious()
            if previous is not None and local(previous) == "l" and blocks:
                blocks[-1] = normalize_space(blocks[-1] + " " + "".join(element.itertext()))
            else:
                blocks.append(normalize_space("".join(element.itertext())))
    return blocks


def doc_sentences(blocks, limit, seed):
    candidates = []
    for block in blocks:
        for sentence in split_sentences(block):
            if keep_sentence(sentence):
                candidates.append(sentence)
    random.Random(seed).shuffle(candidates)
    return candidates[:limit]


def tcp_doc(args):
    doc, limit = args
    blocks = tei_blocks(TCP_DB / "TEXT" / doc["filename"])
    return doc, doc_sentences(blocks, limit, doc["filename"])


def tcp_docs():
    con = sqlite3.connect(TCP_DB / "toms.db")
    con.row_factory = sqlite3.Row
    docs = defaultdict(list)
    for row in con.execute("SELECT filename, year, pub_date, create_date, author, title, collection, idno, pub_place "
                           "FROM toms WHERE philo_type = 'doc'"):
        if AMERICAN_IMPRINT.search(row["pub_place"] or ""):
            continue
        year = tcp_year(row["pub_date"], row["create_date"], row["year"])
        period = tcp_period(year)
        if period:
            source = "EEBO-TCP" if "Early English" in (row["collection"] or "") else "ECCO-TCP"
            docs[period].append({"filename": row["filename"], "year": year, "author": row["author"] or "",
                                 "title": row["title"] or "", "doc": row["idno"] or row["filename"],
                                 "source": source})
    return docs


# ---------------------------------------------------------------- Gutenberg (plain text)

def gutenberg_blocks(path):
    text = path.read_text(encoding="utf8", errors="replace")
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", text)
    text = text[start.end() if start else 0 : end.start() if end else len(text)]
    text = re.sub(r"\[(Illustration|Footnote|Sidenote|Transcriber)[^\]]*\]", "", text, flags=re.S | re.I)
    text = re.sub(r"(?<!\w)_([^_\n]+(?:\n[^_\n]+)*)_(?!\w)", r"\1", text)  # _italics_
    blocks = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        paragraph = normalize_space(" ".join(lines))
        if len(paragraph) < 80 or paragraph.isupper():
            continue  # headings, page furniture
        blocks.append(paragraph)
    return blocks


def gutenberg_doc(args):
    book, limit = args
    path = GUTENBERG_DIR / "texts" / book["id"] / f"pg{book['id']}.txt"
    if not path.exists():
        return book, []
    return book, doc_sentences(gutenberg_blocks(path), limit, book["id"])


def gutenberg_docs():
    docs = defaultdict(list)
    with open(GUTENBERG_DIR / "selected.tsv", encoding="utf8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            docs[row["period"]].append({"id": row["id"], "author": row["author"], "title": row["title"],
                                        "year": f"{row['birth']}-{row['death']}", "doc": f"PG{row['id']}",
                                        "source": "Project Gutenberg"})
    return docs


# ---------------------------------------------------------------- sampling

def sample_period(period, docs, worker, pool_size, processes, scarce=True):
    """Take up to `limit` sentences per document, from documents in random order, until the pool is full."""
    random.Random(period).shuffle(docs)
    # TCP documents are scarce in the 16th and 18th centuries: allow more sentences per document there
    limit = max(MAX_PER_DOC, math.ceil(2 * pool_size / max(len(docs), 1))) if scarce else GUTENBERG_PER_BOOK
    per_author = defaultdict(int)
    author_cap = max(MAX_PER_AUTHOR, limit * 5)
    sentences = []
    with Pool(processes) as pool:
        for doc, doc_sents in pool.imap(worker, [(doc, limit) for doc in docs], chunksize=4):
            author = doc["author"].strip() or doc["doc"]
            room = min(len(doc_sents), author_cap - per_author[author])
            for sentence in doc_sents[:max(room, 0)]:
                sentences.append({"text": sentence, "doc": doc["doc"], "year": doc["year"], "author": doc["author"],
                                  "title": doc["title"][:200], "source": doc["source"]})
            per_author[author] += max(room, 0)
            if len(sentences) >= pool_size:
                pool.terminate()
                break
    random.Random(period).shuffle(sentences)
    print(f"{period}: {len(docs)} documents available, up to {limit} sentences each, "
          f"{len(sentences)} sentences from {len({s['doc'] for s in sentences})} documents")
    return sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", nargs="*", default=TCP_PERIODS + GUTENBERG_PERIODS)
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    parser.add_argument("--processes", type=int, default=16)
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    tcp = tcp_docs() if set(args.periods) & set(TCP_PERIODS) else {}
    gutenberg = gutenberg_docs() if set(args.periods) & set(GUTENBERG_PERIODS) else {}
    for period in args.periods:
        if period in TCP_PERIODS:
            sentences = sample_period(period, tcp[period], tcp_doc, args.pool_size, args.processes)
        else:
            sentences = sample_period(period, gutenberg[period], gutenberg_doc, args.pool_size, args.processes,
                                      scarce=False)
        with open(OUTPUT_DIR / f"sentences_{period}.jsonl", "w", encoding="utf8") as f:
            for sentence in sentences:
                f.write(json.dumps(sentence, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
