#!/usr/bin/env python3
"""
Annotate sentences with POS, TAG and LEMMA using an LLM, following ANNOTATION_GUIDELINES.md.

Each sentence is tokenized with the historic tokenizer first, and the model labels the
numbered tokens: tokenization never has to be reconciled afterwards. Several sentences
go in each request. The model answers with one tab-separated line per token
(token, POS, TAG, lemma): JSON costs about three times as many output tokens. Answers
whose tokens or tags don't match are retried.

Models are reached through an OpenAI-compatible API: OpenRouter, or a local llama-server
(whose output is constrained by a grammar). With --batch, the requests go through
OpenRouter's Batch API instead: half price, results within 24 hours. The batch id is
saved, so an interrupted run resumes polling the same batch.

    python annotate.py sonnet-5.5 gold/gold_1_sentences.jsonl --reasoning low --output annotations/gold
    python annotate.py gemini-4 gold/gold_1_sentences.jsonl --reasoning high --batch --output annotations/gold

Writes <output>/<run>.jsonl ({"id", "text", "annotations": {"tokens": [...]}} per sentence,
in input order) and <output>/<run>.usage.json (tokens, cost, time). Re-running the same
command resumes: sentences already in the output file are not sent again.
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from openai import APIError, OpenAI

from historic_tokenizer import make_nlp

OPENROUTER_URL = "https://openrouter.ai/api/v1"
MODELS = {
    "gemma4-31b": {"base_url": "http://127.0.0.1:8796/v1", "model": "gemma-4-31B-it", "local": True},
    "gemini-3.8-flash": {"base_url": OPENROUTER_URL, "model": "google/gemini-3.8-flash"},
    "sonnet-5.5": {"base_url": OPENROUTER_URL, "model": "anthropic/claude-sonnet-5.5"},
    "gemini-4": {"base_url": OPENROUTER_URL, "model": "google/gemini-4-argon"},
    "opus-5.5": {"base_url": OPENROUTER_URL, "model": "anthropic/claude-opus-5.5"},
}
BATCH_POLL_SECONDS = 60

POS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
            "SCONJ", "SYM", "VERB", "X"]
PTB_TAGS = ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD", "JJ", "JJR", "JJS", "RB",
            "RBR", "RBS", "RP", "IN", "TO", "CC", "DT", "PDT", "PRP", "PRP$", "WDT", "WP", "WP$", "WRB", "EX", "CD",
            "POS", "UH", "FW", "SYM", ",", ".", ":", "``", "''", "-LRB-", "-RRB-", "HYPH", "NFP"]

EXAMPLE = [("'T", "PRON", "PRP", "it"), ("is", "AUX", "VBZ", "be"), ("true", "ADJ", "JJ", "true"),
           (",", "PUNCT", ",", ","), ("thou", "PRON", "PRP", "thou"), ("know'st", "VERB", "VBP", "know"),
           ("it", "PRON", "PRP", "it"), ("not", "PART", "RB", "not"), (";", "PUNCT", ":", ";"),
           ("vnto", "ADP", "IN", "unto"), ("him", "PRON", "PRP", "he"), ("hath", "AUX", "VBZ", "have"),
           ("the", "DET", "DT", "the"), ("Lord", "PROPN", "NNP", "Lord"), ("giuen", "VERB", "VBN", "give"),
           ("ſuch", "DET", "PDT", "such"), ("a", "DET", "DT", "a"), ("Publick", "ADJ", "JJ", "public"),
           ("Charge", "NOUN", "NN", "charge"), (".", "PUNCT", ".", ".")]


def gbnf_literal(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# llama.cpp grammar for the answer: a header per sentence, then token lines with valid tags
GRAMMAR = "\n".join([
    # no newline is required after the last line, so that the model can stop there
    'root ::= sentence ("\\n"+ sentence)* "\\n"*',
    'sentence ::= "# sentence " [0-9]+ ("\\n" line)+',
    'line ::= [^\\t\\n]+ "\\t" pos "\\t" tag "\\t" [^\\t\\n]+',
    "pos ::= " + " | ".join(gbnf_literal(p) for p in POS_TAGS),
    "tag ::= " + " | ".join(gbnf_literal(t) for t in PTB_TAGS),
])


def system_prompt():
    guidelines = Path(__file__).with_name("ANNOTATION_GUIDELINES.md").read_text(encoding="utf8")
    example = "\n".join(["# sentence 0"] + ["\t".join(token) for token in EXAMPLE])
    return f"""You are an expert annotator of English texts printed between 1500 and 2000. You label every token of the sentences you are given with a part of speech (POS), a Penn Treebank tag (TAG) and a lemma, following the guidelines below exactly.

{guidelines}

## Output

For each sentence, write the line `# sentence <id>`, then one line per numbered token, in order, with four fields separated by tabs: the token exactly as given, its POS, its TAG, its lemma. Never skip, split or merge tokens. Write nothing else.

Example. Input tokens: 0='T 1=is 2=true 3=, 4=thou 5=know'st 6=it 7=not 8=; 9=vnto 10=him 11=hath 12=the 13=Lord 14=giuen 15=ſuch 16=a 17=Publick 18=Charge 19=.
Output:
{example}"""


def user_prompt(batch):
    parts = ["Annotate every token of these sentences."]
    for sentence in batch:
        tokens = " ".join(f"{i}={t}" for i, t in enumerate(sentence["tokens"]))
        parts.append(f"Sentence id {sentence['id']} (printed {sentence.get('period', 'unknown date')}):\n"
                     f"Text: {sentence['text']}\nTokens: {tokens}")
    return "\n\n".join(parts)


# the usual POS of each TAG, to repair answers that put a TAG in the POS column ("NNS NNS word")
TAG_TO_POS = {"NN": "NOUN", "NNS": "NOUN", "NNP": "PROPN", "NNPS": "PROPN", "VB": "VERB", "VBD": "VERB", "VBG": "VERB",
              "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB", "MD": "AUX", "JJ": "ADJ", "JJR": "ADJ", "JJS": "ADJ",
              "RB": "ADV", "RBR": "ADV", "RBS": "ADV", "RP": "ADP", "IN": "ADP", "TO": "PART", "CC": "CCONJ", "DT": "DET",
              "PDT": "DET", "PRP": "PRON", "PRP$": "PRON", "WDT": "PRON", "WP": "PRON", "WP$": "PRON", "WRB": "ADV",
              "EX": "PRON", "CD": "NUM", "POS": "PART", "UH": "INTJ", "FW": "X", "SYM": "SYM", ",": "PUNCT", ".": "PUNCT",
              ":": "PUNCT", "``": "PUNCT", "''": "PUNCT", "-LRB-": "PUNCT", "-RRB-": "PUNCT", "HYPH": "PUNCT", "NFP": "PUNCT"}
REPAIRS = Counter()


def nfc(text):
    return unicodedata.normalize("NFC", text)


def parse(batch, answer):
    """Annotations for each sentence of the batch, or raise ValueError if the answer doesn't match the tokens."""
    by_id, current = {}, None
    for line in answer.strip().strip("`").splitlines():
        header = re.fullmatch(r"#\s*sentence\s+(\d+)\s*", line.strip())
        if header:
            current = by_id.setdefault(int(header[1]), [])
        elif line.strip() and current is not None:
            fields = line.split("\t")
            if len(fields) != 4:
                raise ValueError(f"malformed line {line!r}")
            current.append(fields)
    results = {}
    for sentence in batch:
        lines = by_id.get(sentence["id"])
        if lines is None:
            raise ValueError(f"sentence {sentence['id']} missing")
        # models sometimes echo a letter and combining mark as one precomposed character, or the reverse
        if [nfc(w) for w, _, _, _ in lines] != [nfc(t) for t in sentence["tokens"]]:
            raise ValueError(f"sentence {sentence['id']}: tokens differ")
        for line in lines:
            _, pos, tag, lemma = line
            if pos not in POS_TAGS and tag in PTB_TAGS:
                line[1] = TAG_TO_POS[tag]
                REPAIRS[f"POS {pos} -> {line[1]} (from TAG {tag})"] += 1
            if line[1] not in POS_TAGS or tag not in PTB_TAGS or not lemma.strip():
                raise ValueError(f"sentence {sentence['id']}: bad labels {pos} {tag} {lemma!r}")
        results[sentence["id"]] = [{"text": token, "pos": p, "tag": t, "lemma": nfc(l.strip())}
                                   for token, (_, p, t, l) in zip(sentence["tokens"], lines)]
    return results


class Annotator:
    def __init__(self, name, reasoning, max_tokens):
        config = MODELS[name]
        self.local = config.get("local", False)
        key = "none" if self.local else Path(__file__).with_name("OPENROUTER.key").read_text().strip()
        self.client = OpenAI(base_url=config["base_url"], api_key=key, timeout=900, max_retries=3)
        self.headers = {"Authorization": f"Bearer {key}"}
        self.model = config["model"]
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.system = system_prompt()

    def body(self, batch):
        """The chat completion request for a batch of sentences, without the model."""
        body = {"messages": [{"role": "system", "content": self.system}, {"role": "user", "content": user_prompt(batch)}],
                "max_tokens": self.max_tokens}
        if self.local:
            body.update(grammar=GRAMMAR, temperature=0)
        else:
            # None of the OpenRouter models used lets reasoning be turned off
            body.update(usage={"include": True}, reasoning={"effort": self.reasoning})
            if "gemini" in self.model:
                body["temperature"] = 0  # the Claude models reject non-default sampling parameters
        return body

    def request(self, batch):
        body = self.body(batch)
        standard = {key: body.pop(key) for key in ("messages", "max_tokens", "temperature") if key in body}
        response = self.client.chat.completions.create(model=self.model, extra_body=body, **standard)
        usage = response.usage.model_dump() if response.usage else {}
        return response.choices[0].message.content or "", usage

    def submit(self, groups):
        """Send groups of sentences to OpenRouter's Batch API, one request per group; return the batch id."""
        requests = [{"custom_id": str(k), "body": self.body(group)} for k, group in enumerate(groups)]
        # the Batch API wants endpoint and model before requests
        payload = {"endpoint": "/v1/chat/completions", "model": self.model, "requests": requests}
        response = httpx.post(f"{OPENROUTER_URL}/batches", headers=self.headers, json=payload, timeout=600)
        if response.status_code >= 400:
            raise RuntimeError(f"batch submission failed: {response.status_code} {response.text[:500]}")
        return response.json()["id"]

    def wait(self, batch_id):
        while True:
            try:
                response = httpx.get(f"{OPENROUTER_URL}/batches/{batch_id}", headers=self.headers, timeout=600)
                response.raise_for_status()
                batch = response.json()
            except httpx.HTTPError as error:
                print(f"\npolling {batch_id} failed ({error}), retrying", file=sys.stderr)
                time.sleep(BATCH_POLL_SECONDS)
                continue
            counts = batch.get("request_counts") or {}
            print(f"\rbatch {batch_id}: {batch['status']}, {counts.get('completed', 0)} of {counts.get('total', '?')} "
                  f"requests done", end="", file=sys.stderr)
            if batch["status"] in ("completed", "failed", "expired", "cancelled"):
                print(file=sys.stderr)
                return batch
            time.sleep(BATCH_POLL_SECONDS)

    def annotate(self, batch):
        """Annotate a batch; on a malformed answer, retry once, then sentence by sentence."""
        usages, last_error = [], None
        for _ in range(2):
            try:
                answer, usage = self.request(batch)
                usages.append(usage)
                return parse(batch, answer), usages, []
            except (ValueError, APIError) as error:
                last_error = error
        if len(batch) == 1:
            return {}, usages, [(batch[0]["id"], str(last_error)[:300])]
        results, failures = {}, []
        for sentence in batch:
            r, u, f = self.annotate([sentence])
            results.update(r)
            usages += u
            failures += f
        return results, usages, failures


def batch_annotate(annotator, todo, batch_size, state_path, save, all_sentences, requests_per_batch=500):
    """Annotate through the Batch API, in batches of at most `requests_per_batch` requests.
    Groups that fail are sent again one sentence at a time, in a second and then a third round.
    The batch ids are saved in `state_path`, so that an interrupted run resumes polling them, and
    the results are saved with `save` after each round. If a submission is refused (e.g. for lack
    of credit), the sentences it would have sent are left for a later run."""
    # a saved state can include sentences annotated since: look them up among all sentences
    by_id = {s["id"]: s for s in all_sentences}
    results, usages, errors = {}, [], {}
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    attempt = state["attempt"] if state else 0
    refused = False
    while attempt < 3 and not refused:
        if state is None:
            remaining = [s for s in todo if s["id"] not in results]
            if not remaining:
                break
            size = batch_size if attempt == 0 else 1
            groups = [[s["id"] for s in remaining[i : i + size]] for i in range(0, len(remaining), size)]
            state = {"attempt": attempt, "batches": []}
            for i in range(0, len(groups), requests_per_batch):
                chunk = groups[i : i + requests_per_batch]
                try:
                    batch_id = annotator.submit([[by_id[k] for k in g] for g in chunk])
                except RuntimeError as error:
                    print(f"{error}\nThe sentences not yet submitted are left for a later run.", file=sys.stderr)
                    for group in groups[i:]:
                        for k in group:
                            errors[k] = str(error)[:300]
                    refused = True
                    break
                state["batches"].append({"id": batch_id, "groups": chunk})
                state_path.write_text(json.dumps(state))
                print(f"submitted batch {batch_id}: {len(chunk)} requests", file=sys.stderr)
            if not state["batches"]:
                break
        failed = total = 0
        for entry in state["batches"]:
            batch = annotator.wait(entry["id"])
            if batch["status"] != "completed":
                raise RuntimeError(f"batch {entry['id']} ended as {batch['status']}: {batch.get('error')}")
            usages.append(batch.get("usage") or {})
            for item in batch.get("results") or []:
                total += 1
                group = [by_id[k] for k in entry["groups"][int(item["custom_id"])]]
                response = item.get("response") or {}
                if response.get("status_code") == 200:
                    try:
                        results.update(parse(group, response["body"]["choices"][0]["message"]["content"] or ""))
                        continue
                    except (ValueError, KeyError, IndexError, TypeError) as error:
                        message = str(error)
                else:
                    message = json.dumps(item.get("error") or response)
                failed += 1
                for sentence in group:
                    errors[sentence["id"]] = message[:300]
        save(results)
        if attempt == 0 and total and failed > total / 2:
            # most likely a problem with the requests or with reading the results: don't resend everything
            raise RuntimeError(f"{failed} of {total} requests failed, e.g. {next(iter(errors.values()))}; "
                               f"state kept in {state_path}")
        state, attempt = None, attempt + 1
        state_path.unlink()
    failures = [(s["id"], errors.get(s["id"], "not annotated")) for s in todo if s["id"] not in results]
    return results, usages, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(MODELS))
    parser.add_argument("input")
    parser.add_argument("--output", default="annotations")
    parser.add_argument("--reasoning", default="low", choices=["minimal", "low", "medium", "high"],
                        help="reasoning effort for OpenRouter models (the local model runs without reasoning)")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--batch", action="store_true", help="use OpenRouter's Batch API (half price, up to 24 hours)")
    args = parser.parse_args()

    nlp = make_nlp()
    with open(args.input, encoding="utf8") as f:
        sentences = [json.loads(line) for line in f]
    for sentence in sentences:
        sentence["text"] = nfc(sentence["text"])
        sentence["tokens"] = [t.text for t in nlp.tokenizer(sentence["text"])]

    annotator = Annotator(args.model, args.reasoning, args.max_tokens)
    run = args.model if annotator.local else f"{args.model}_{args.reasoning}"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # resume: keep sentences already annotated, add their cost to this run's
    results, previous = {}, {}
    if (output / f"{run}.jsonl").exists():
        with open(output / f"{run}.jsonl", encoding="utf8") as f:
            results = {s["id"]: s["annotations"]["tokens"] for s in map(json.loads, f)}
        if (output / f"{run}.usage.json").exists():
            previous = json.loads((output / f"{run}.usage.json").read_text())
    todo = [s for s in sentences if s["id"] not in results]
    batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    print(f"{run}: {len(results)} sentences already annotated, {len(todo)} to go", file=sys.stderr)

    def save(new_results):
        with open(output / f"{run}.jsonl", "w", encoding="utf8") as f:
            for sentence in sentences:
                tokens = new_results.get(sentence["id"]) or results.get(sentence["id"])
                if tokens:
                    f.write(json.dumps({"id": sentence["id"], "text": sentence["text"], "period": sentence.get("period"),
                                        "doc": sentence.get("doc"), "annotations": {"tokens": tokens}},
                                       ensure_ascii=False) + "\n")

    usages, failures = [], []
    start = time.time()
    if args.batch:
        r, usages, failures = batch_annotate(annotator, todo, args.batch_size, output / f"{run}.batch.json", save,
                                             sentences)
        results.update(r)
    else:
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(annotator.annotate, batch) for batch in batches]
            for done, future in enumerate(as_completed(futures), 1):
                r, u, f = future.result()
                results.update(r)
                usages += u
                failures += f
                print(f"\r{run}: {done}/{len(batches)} batches", end="", file=sys.stderr)
        print(file=sys.stderr)
    elapsed = time.time() - start

    save(results)
    totals = {key: sum(u.get(key) or 0 for u in usages) for key in ("prompt_tokens", "completion_tokens", "cost")}
    totals["reasoning_tokens"] = sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0 for u in usages)
    totals["requests"] = len(usages)
    totals["seconds"] = round(elapsed)
    totals = {key: value + previous.get(key, 0) for key, value in totals.items()}
    summary = {"run": run, "sentences": len(sentences), "annotated": len(results), "failures": failures,
               "repaired_labels": dict(REPAIRS), **totals}
    (output / f"{run}.usage.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
