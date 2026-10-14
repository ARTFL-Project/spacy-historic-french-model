"""
spaCy's English tokenizer, adapted to early modern spellings.

The default tokenizer already keeps elided forms such as "know'st", "lov'd", "o'er" and
"e'en" whole, and splits "don't" -> "do" + "n't". It mishandles a few historic forms:

    'Tis   -> "'" + "Tis"        now "'T" + "is", like "it is"
    Th'    -> "Th" + "'"         now "Th'" (elided "the"), also when glued: "th'Almighty"
    do't   -> "do't"             now "do" + "'t" (elided "it")
    &c.    -> "&" + "c."         now "&c." (etc.)
    I'le   -> "I'le"             now "I" + "'le", like "I" + "'ll": early modern contractions of
                                 will, wilt and would after a pronoun (we'l, hee'l, thou'lt, I'de)
    thus!—nor -> one token       now "thus" + "!" + "—" + "nor" (also with "--" for the dash)
    wont   -> "wo" + "nt"        now "wont" (accustomed): spaCy's contractions written without
                                 an apostrophe (wont, cant, wed, id, thats...) are dropped, since
                                 in historic texts these are mostly real words
"""

import itertools
import re

import spacy
from spacy.attrs import ORTH
from spacy.lang.char_classes import ALPHA
from spacy.util import compile_infix_regex

APOSTROPHES = ("'", "’")

# 'tis, 'twas...: elided "it" + verb
ELIDED_IT = ["is", "was", "were", "will", "would", "had", "has", "hath", "shall", "should"]
# do't, on't...: host word + elided "it"
IT_HOSTS = ["do", "on", "is", "in", "to", "for", "was", "see", "take", "make", "have", "has", "had", "let",
            "put", "upon", "of", "at", "by", "with", "know", "hear", "bear", "get", "done", "did", "does",
            "ha", "be", "think", "show", "shew", "give", "tell", "keep", "about", "doe", "doo"]
# I'le, we'l, hee'l, thou'lt, I'de: pronoun + contracted will, wilt or would. Modern forms
# (I'll, he'd) are already split by spaCy; the historic hosts (hee, wee, thou...) are added too.
CLITIC_HOSTS = ["i", "we", "wee", "you", "ye", "he", "hee", "she", "shee", "they", "thou", "it", "who", "there"]
CLITICS = ["le", "l", "lt", "de", "ll", "d"]
# whole tokens: elided "the", "in", "of", "you"
WHOLE = ["th'", "i'", "o'", "y'", "t'"]
# apostrophe-less exceptions of spaCy's that are kept
KEPT_EXCEPTIONS = {"cannot", "gonna", "gotta", "&c.,"}


def case_variants(word):
    return {word, word.capitalize(), word.upper()}


def add_special_cases(tokenizer):
    for apostrophe in APOSTROPHES:
        for verb in ELIDED_IT:
            for it_form, verb_form in (("t", verb), ("T", verb), ("T", verb.upper())):
                tokenizer.add_special_case(f"{apostrophe}{it_form}{verb_form}",
                                           [{ORTH: f"{apostrophe}{it_form}"}, {ORTH: verb_form}])
        for host in IT_HOSTS:
            for host_form in case_variants(host):
                t = "T" if host_form.isupper() else "t"
                tokenizer.add_special_case(f"{host_form}{apostrophe}{t}",
                                           [{ORTH: host_form}, {ORTH: f"{apostrophe}{t}"}])
        for host in CLITIC_HOSTS:
            for host_form in case_variants(host) | {host.upper() if host == "i" else host}:
                for clitic in CLITICS:
                    clitic_form = clitic.upper() if host_form.isupper() and len(host_form) > 1 else clitic
                    for form in {clitic_form, clitic}:
                        tokenizer.add_special_case(f"{host_form}{apostrophe}{form}",
                                                   [{ORTH: host_form}, {ORTH: f"{apostrophe}{form}"}])
        for word in WHOLE:
            for form in case_variants(word.replace("'", apostrophe)):
                tokenizer.add_special_case(form, [{ORTH: form}])
        # i'th', o'th' (also without the final apostrophe): "in the", "of the"
        for first, second in itertools.product(("i", "o", "I", "O"), ("th", "Th")):
            tokenizer.add_special_case(f"{first}{apostrophe}{second}{apostrophe}",
                                       [{ORTH: f"{first}{apostrophe}"}, {ORTH: f"{second}{apostrophe}"}])
            tokenizer.add_special_case(f"{first}{apostrophe}{second}",
                                       [{ORTH: f"{first}{apostrophe}"}, {ORTH: second}])
    for form in ("&c.", "&c", "&C.", "&c.,"):
        if form.endswith(","):
            tokenizer.add_special_case(form, [{ORTH: form[:-1]}, {ORTH: ","}])
        else:
            tokenizer.add_special_case(form, [{ORTH: form}])


def remove_contractions_without_apostrophe(tokenizer):
    tokenizer.rules = {text: tokens for text, tokens in tokenizer.rules.items()
                       if "'" in text or "’" in text or len(tokens) == 1 or re.search(r"[\d°]", text)
                       or text.lower() in KEPT_EXCEPTIONS}


def add_infixes(nlp):
    infixes = list(nlp.Defaults.infixes) + [
        # elided "the"/"to" glued to the next word: "th'Almighty", "t'obey"
        r"(?<=\b[Tt]h['’])(?=[A-Za-zſ])", r"(?<=\b[Tt]['’])(?=[a-zſ]{2})",
        # punctuation and a dash between two words: "thus!—nor", "her.—But", "Sun,--disdaining"
        rf"(?<=[{ALPHA}])[!?;:,.](?=[—–]|--)", rf"(?<=[!?;:,.])(?:[—–]+|-{{2,}})(?=[{ALPHA}])",
    ]
    nlp.tokenizer.infix_finditer = compile_infix_regex(infixes).finditer


@spacy.registry.callbacks("historic_english_tokenizer.v1")
def create_tokenizer_callback():
    """For training configs: [nlp] after_creation = {"@callbacks":"historic_english_tokenizer.v1"}.
    The rules are saved with the model's tokenizer, so a trained model loads without this code."""
    def customize(nlp):
        add_infixes(nlp)
        remove_contractions_without_apostrophe(nlp.tokenizer)
        add_special_cases(nlp.tokenizer)
        return nlp
    return customize


def make_nlp():
    nlp = spacy.blank("en")
    add_infixes(nlp)
    remove_contractions_without_apostrophe(nlp.tokenizer)
    add_special_cases(nlp.tokenizer)
    return nlp


if __name__ == "__main__":
    nlp = make_nlp()
    for text in [
        "'Tis true, thou know'st it not; I lov'd her o'er much, &c. and 'twas so.",
        "Th' Almighty hath giv'n unto vs th'Almighty's grace, i'th' name of God: do't, on't, Is't so?",
        "Don't, can't, I'll, ne'er, ta'en, t'obey, to-morrow, cannot, Mr. Smith's dog, 'TIS, 'Twas.",
        "'Tis, 'tis! Speak'st thou to me? Thou wouldst not do't, wouldst thou?",
        "He was wont to say thus!—nor her.—But i'th midst, o'th world, the Cant of men wed to it; cannot.",
        "I'le tell thee, we'l see, WE'l fight, they'l come, hee'l goe, I'de rather, thou'lt not; I'll go, lov'd.",
        "The Sun,--disdaining all; said he.--But well--known, to--morrow.",
    ]:
        print([t.text for t in nlp(text)])
