# Annotation guidelines: historic English (1500–2000)

These conventions apply to the training data, the test sets and the gold sets. Each token gets three labels: **POS** (Universal Dependencies UPOS), **TAG** (Penn Treebank) and **LEMMA**.

**Reference.** For everything that is not specific to historic English, follow the conventions of the UD English treebanks EWT and GUM, which use the same three labels: when they label a word or a construction one way in the large majority of cases, do the same (`python ud_conventions.py lookup <word or phrase>` shows how they label it). The rules below cover what they don't: historic spellings and forms, British spelling of lemmas, and the cases where the treebanks are divided.

Tokenization is fixed: never split, merge or edit a token. Spellings are historic and must be read as printed: long s (*ſay* = say), u/v and i/j used by position (*vnto* = unto, *haue* = have, *Iohn* = John), a macron or tilde for a missing n or m (*cōtrary* = contrary, *thē* = them), *-eth* and *-est* endings, elided forms (*lov'd*, *know'st*, *th'*). Capitalized common nouns are frequent before 1800 (*the Multitude of Patients*): they are still common nouns.

## Tagset: TAG and the POS it goes with

| TAG | POS | Use |
|---|---|---|
| NN | NOUN | singular or mass noun |
| NNS | NOUN | plural noun |
| NNP | PROPN | singular proper noun; also a title that is part of a name (*Sir Thomas*, *King William*, *Mr.*, *St.*) |
| NNPS | PROPN | plural proper noun (*the Indies*, *Americans*) |
| VB | VERB/AUX | base form: infinitive, imperative, subjunctive (*if he come*, *God save the King*) |
| VBD | VERB/AUX | past tense, including 2nd singular *didst, wast, wert, hadst, lovedst* |
| VBG | VERB/AUX | gerund or present participle |
| VBN | VERB/AUX | past participle |
| VBP | VERB/AUX | present tense, not 3rd singular: *I love, they be, thou art, thou hast, thou lovest, thou dost* |
| VBZ | VERB/AUX | present 3rd singular, including *-eth* forms: *hath, doth, saith, loveth* |
| MD | AUX | modal: *will, would, shall, should, can, could, may, might, must, ought*, also *wilt, shalt, canst, couldst, wouldst, shouldst, mayst, mightst* |
| JJ, JJR, JJS | ADJ | adjective, comparative, superlative |
| RB, RBR, RBS | ADV | adverb, comparative, superlative; *not* and *n't* are PART with TAG RB |
| RP | ADP | verb particle (*give up*, *set forth*); *away* and *back* are RB ADV |
| IN | ADP | preposition (*of, in, unto, upon*, preposition *to*) |
| IN | SCONJ | subordinating conjunction and complementizer (*if, because, though, lest, whilst, ere, that* before a clause, *an* meaning *if*); a preposition before a gerund (*by going*, *from peeping*) |
| TO | PART | infinitive marker *to* only; preposition *to* is IN/ADP |
| CC | CCONJ | coordinating conjunction (*and, &, or, but, nor, yet*; *both*, *either*, *neither* in *both … and*) |
| DT | DET | article and determiner (*the, a, this, that, these, no, every, each, some, any, all, th'*) |
| DT | PRON | demonstrative used without a noun (*this is true*, *those who came*) |
| PDT | DET | predeterminer (*all the men*, *such a man*, *half the*) |
| PRP | PRON | personal and reflexive pronoun (*I, me, thou, thee, ye, you, him, it, 'T, 't, himself*) |
| PRP$ | PRON | possessive determiner (*my, thy, mine eyes, thine own, his, its, our, their*) |
| WDT | PRON | relative *which, that* |
| WDT | DET | interrogative or relative determiner (*which way*, *what man*, *whatsoever thing*) |
| WP | PRON | *who, whom, what, whoso, whosoever* |
| WP$ | PRON | *whose* |
| WRB | ADV | *when, where, why, how, whereof, wherein, whereby, wherefore, whither, whence* |
| EX | PRON | existential *there* |
| CD | NUM | number, including Roman numerals |
| POS | PART | possessive *'s* and *'* |
| UH | INTJ | interjection (*O, oh, alas, yea, nay, ay*, interjection *no*, *yes*) |
| FW | X | foreign word (Latin, French) and *&c.* / *etc.* |
| SYM | SYM | symbol (*£, $, %*) |
| `,` | PUNCT | comma |
| `.` | PUNCT | sentence-final *. ? !* |
| `:` | PUNCT | *; : — – …* and other mid-sentence punctuation |
| `` ` `` `` ` `` | PUNCT | opening quotation mark |
| `''` | PUNCT | closing quotation mark |
| -LRB- / -RRB- | PUNCT | opening / closing parenthesis |
| HYPH | PUNCT | hyphen inside a hyphenated word (*to - morrow*) |
| NFP | PUNCT | other punctuation |

TAG must be consistent with POS as in the table.

## Verbs

- **be** is AUX as auxiliary (*he is come*, *it was done*, *I am going*) and as copula (*he is wise*, *it was in the house*, *'tis true*). It is VERB only when it means "exist" or "happen": *there is a God*, *to be or not to be*, *so it be*.
- **have** is AUX only when followed by a past participle (*he hath said*, *I have been*). Otherwise it is VERB: possession and *have to* (*he hath three sons*, *I had rather*).
- **do** is AUX in questions, negation and the periphrastic or emphatic form (*doth appear*, *did see*, *I do confess*, *do not*). Otherwise it is VERB (*he did his duty*).
- **Modals** (TAG MD) are always AUX. Their lemma is the modern spelling of the modal itself, present or past: *will, would, shall, should, can, could, may, might, must* (*myght* → might, *coulde* → could, *wilt* → will, *shalt* → shall, *canst* → can, *wouldst* → would, *mayst* → may).
- **Lemma**: the modern base form. Examples: *hath, hast, had, 'd* (= had) → have; *doth, dost, didst* → do; *art, wert, wast, be, 's* (= is) → be; *saith, sayth* → say; *lov'd, loveth, lovest* → love; *shew* → show; *spake* → speak; *holpen* → help.
- **Participles** are VERB (VBN or VBG, with the verb as lemma) wherever they occur: compound tenses, the passive, and also before a noun or after *be* (*the wounded idol*, *the following pages*, *running lines*, *healing wings*, *well paid rents*, *the said viewers*, *a net spread*). They are JJ ADJ, with the participle as lemma, only when they have become ordinary adjectives: when they take *very*, *too* or *more* (*very trifling*, *too frightened*, *much pleased*), have a sense the verb lacks (*learned* "erudite", *blessed* "holy", *a moving argument*, *broken phrases*, *the left wing*), or have no verb in use (*diseased*, *drunken*). Typical adjectives: *interested, pleased, tired, satisfied, acquainted, charming, scattered, scared*.
- **Gerunds** are VBG VERB (*her sitting on the throne*, *the destroying it*, *my enquiring after*). They are NN/NNS NOUN, with the gerund as lemma, when used as a plain noun: with *the … of* (*the working of God*, *the advancing of our men*), in the plural (*comings and goings*), or as a noun of their own (*learning* "erudition", *a beginning*, *a happy ending*, *a blessing*).
- *need* and *dare* followed by a bare infinitive (*I need not mention*) are MD AUX.

## Pronouns, determiners and function words

- Pronoun lemmas are the nominative: *me* → I; *him* → he; *her* (object) → she; *us, vs* → we; *them, 'em, thē* → they; *thee* → thou; *ye* → you; *'T, 't* → it; *whom* → who. Reflexives keep their own lemma: *himself*, *thyself*. Written apart (*him selfe*, *it self*, *my ſelf*), the parts are analyzed separately: *self* is NN NOUN, lemma *self*.
- Possessive determiners keep their own form: *my, thy, his, her, its, our, your, their*. *mine*/*thine* before a noun are PRP$ with lemma *my*/*thy*. Used alone (*it is mine*, *thine is the kingdom*), they are PRP PRON with lemma *my*/*thy*.
- *a*, *an* → DT DET, lemma *a*. *these* → lemma *this*, *those* → lemma *that*. A demonstrative used without a noun (*this is true*, *those who came*) → DT PRON; quantifiers stay DT DET without a noun (*some of your deputies*, *all were lost*, *each of them*, *another*).
- *that*: relative pronoun → WDT PRON; complementizer → IN SCONJ; demonstrative before a noun → DT DET; demonstrative alone → DT PRON.
- *which*: relative → WDT PRON; before a noun → WDT DET.
- *as*: before a noun phrase → IN ADP; before a clause, also an elliptical one (*as in some tent*, *as heretofore*) → IN SCONJ; first *as* of *as … as* → RB ADV. *than* likewise.
- *for* introducing a clause ("because") or a subject and infinitive (*impossible for me to grant*) → IN SCONJ; otherwise IN ADP. A preposition before a gerund (*from peeping*, *upon going*, *by doing*) is IN SCONJ.
- *but*: conjunction → CC CCONJ; "only" → RB ADV, lemma *but*; "except" → IN ADP.
- *since, till, until, before, after*: before a noun phrase → IN ADP; before a clause → IN SCONJ. *when*, *where* → WRB ADV, also when they introduce a clause; *while*, *whilst* → IN SCONJ.
- *then* used for *than* (*more clearly then on Mount Sinai*) → IN ADP or SCONJ like *than*, lemma *than*. *about*, *above*, *near* meaning "approximately" (*about 80 l.*) → RB ADV. *out* in *out of* → IN ADP.
- *such* before a noun or an adjective (*such sort*, *such men*) → JJ ADJ; it is PDT DET only before an article (*such a man*). *same* (also in *the same* used alone), *own* and *other* → JJ ADJ; *others* → NNS NOUN, lemma *other*.
- *nothing, something, anything, everything, nobody, somebody, anybody, everybody, none* → PRON with TAG NN.
- *one*: a number (*one of them*, *one had his wife*) → CD NUM; after a determiner or adjective (*the one*, *every one*, *a good one*) → NN NOUN; generic (*one must*, *like one in a dream*) → PRP PRON; in *one another* → CD PRON.
- *many* and *such* used without a noun (*many believed*, *such is the case*) → JJ ADJ. Exclamative *what a* → WDT DET. *all* before a demonstrative (*all this*) → PDT DET.
- *so* → RB ADV, including in *so that*. *ere, lest, whilst, albeit, till, until* before a clause → IN SCONJ.
- *ye* or *yͤ* used as the article (*ye same*) → DT DET, lemma *the*. *yt* = *that*.
- *th'* → DT DET, lemma *the*; *i'* → IN ADP, lemma *in*; *o'* → IN ADP, lemma *of* or *on* by sense; *t'* → TO PART or IN ADP, lemma *to*; *y'* → PRP PRON, lemma *you*.
- Split contractions: *n't* → RB PART, lemma *not*; *ca* (*can't*) → MD AUX, lemma *can*; *wo* (*won't*) → *will*; *sha* → *shall*; *'ll* and the early modern *'le*, *'l* (*I'le*, *we'l*, *hee'l*) → MD, lemma *will*; *'lt* (*thou'lt*) → MD, lemma *will*; *'d*, *'de* (*I'de*) → MD *would* or VBD *have*, by sense; *'ve* → *have*; *'re*, *'m* → *be*; *'s* → *be*, *have*, POS *'s* or *us* (in *let's*), by sense.

## Proper nouns, titles, foreign words

- Names of people, places, nations, months and days are NNP/NNPS PROPN, even when lowercased. Members of peoples and religions (*a Christian*, *the French*, *Presbyterians*) are PROPN, lemma singular and capitalized; adjectives derived from them (*English poetry*, *British*, *Protestant*, *Anglo - Saxon*) are JJ ADJ with the capital kept in the lemma. Titles directly before a name are NNP PROPN (*King James*, *Sir John*, *Lord Bacon*, *Doctor Faustus*). Used without a name (*the King*, *his Majesty*, *the Lord said*) they are NN NOUN. *God* is NNP PROPN; *Lord* referring to God, *Christ* and the persons of the Trinity (*God the Son*) are NNP PROPN; *heaven* and *the devil* are NOUN. A title with *of* and a place but no personal name (*the Bishop of Rome*, *the Lord of Mayenne*) is NN NOUN. Names of events, works and places keep their own number in the lemma (*the Passage of the Bidasoa*, *Specimens*, *Paper - Buildings*).
- Proper noun lemmas take the modern standard spelling, capitalized, singular: *Iohn* → John; *Henrie* → Henry; *Englands* → England; *Iesus* → Jesus. Keep the printed form when there is no standard modern form.
- Latin or French words inside an English sentence → FW X, with the word itself (lowercased) as lemma. *&c.*, *etc.* → FW X, lemma *etc.*

## Lemmas

- Every lemma is a single word in modern standard **British** spelling, with the common *-ise* endings: *favour, honour, colour, labour, centre, theatre, defence, realise, organise*. Lemmas are lowercase except for proper nouns and *I*. Never put a space in a lemma.
- Normalize historic spellings: *publick* → public; *musick* → music; *compleat* → complete; *shew* → show; *onely* → only; *goe* → go; *doe* → do; *selfe* → self; *kynges* → king; *loue* → love; *euery* → every; *iust* → just; *vnto* → unto; *ſaid* → say; *cōtrary* → contrary.
- Archaic words with no modern equivalent keep their own lemma, modernized in spelling only: *unto, thou, thy, ere, whilst, betwixt, hither, thither, whither, hence, whence, wherefore, whereof, thereof, peradventure, methinks*.
- Abbreviations keep their printed form as lemma, lowercased unless they stand for a proper noun: *viz.*, *Matth.*, *Zep.*, *l.* (pounds).
- Nouns → singular (*children* → child, *brethren* → brother, *kine* → cow, *eyen* → eye), except nouns used only in the plural in that sense (*means*, *riches*, *odds*, *thanks*). Adjectives and adverbs → positive degree (*better* → good or well, *worst* → bad, *largest* → large, *elder* → old), but *more, most, less, least* keep their own lemma. Verbs → base form.
- Possessives printed without an apostrophe (*mens estates*, *the Abbots goods*) are nouns: NN or NNS by sense, with the noun's lemma.
- Dialect spellings of standard words take the standard lemma (*wame* → womb, *hame* → home, *tak* → take, *goin* → go); dialect words with no standard equivalent keep their own (*mun*, *gang*, *syne*). Obsolete words take their OED headword. In northern dialect, *t'* can be *the*. The reduced preposition in *a-whoring*, *a-zettin'* is IN ADP, lemma *a*.
- A word printed in two pieces that are not words on their own (*a ſonder*, *amō ge*) gives each piece the labels of the whole word (*asunder*, *among*).
- Numbers keep their form (*twenty*, *1666*, *xii*). Punctuation is its own lemma, except quotation marks, whose lemma is the straight mark (*“ ”* → *"*, *‘ ’* → *'*); *&* → and; the possessive *’s* → *'s*. An apostrophe split off an elided word (*goin '*, *tho '*) is NFP PUNCT. A period inside a sentence (*viz .*, *Acts 13 . 16*) is `.`; the virgule */* is `:`. Vocative *O* keeps its capital.
- In a hyphenated word split into pieces (*to - morrow*, *well - known*), each piece is lemmatized on its own; the hyphen is HYPH PUNCT with lemma *-*.
- A token that fuses two words takes the labels and lemma of its main word: the verb in a pronoun and verb fusion (*Ile* = I will → MD AUX *will*; *tis* = it is → VBZ AUX *be*), the noun in an article and noun fusion (*thabuse* = the abuse → NN NOUN *abuse*). Thorn abbreviations are read as the word they stand for: *yu* = thou, *yt* = that, *ye*/*yͤ* = the (when used as an article), *wt* = with.
