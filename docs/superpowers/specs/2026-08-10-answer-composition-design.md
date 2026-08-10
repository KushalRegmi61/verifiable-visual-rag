# Answer composition: prose that survives the gate

Date: 2026-08-10
Status: design approved, not implemented
Depends on: S4 (grounding), S5 (reader, verifier, abstention), S6 (product UI)

## 1. What this changes

The reader and verifier prompts, the claim schema, and the three states the
answer surface can be in. No change to retrieval, grounding, the rubric, the
abstention score, or the region strip at the wire boundary.

The goal is an answer that reads like one an assistant wrote, while every
displayed word remains text a second model verified against a region of the
page. The two are in tension and most of this document is about where exactly
the tension resolves.

## 2. What is wrong today

The reader prompt says "Answer ONLY from what is visible on this page" and never
says answer the question. It gets what it asked for. Measured on `proposal.pdf`
page 14, question "What is the evaluation methodology?", six claims came back:

```
1 Both systems are evaluated using answer accuracy, grounding overlap, and abstention quality.
2 The contribution of the added layer can be measured directly through an ablation.
3 Evaluation uses automatically derived ground truth to avoid a manual annotation phase.
4 Figure 5.1 shows the overall approach.
5 Figure 5.1 includes an evaluation step on SlideVQA with three metrics.
6 Figure 5.1 includes an ablation step with layer versus without layer.
```

Every one is true and grounded. Joined, they are a fact dump. Three of six
describe the figure rather than the methodology, nothing states what the
methodology IS, and the order is page order rather than answer order. The
sentence a person actually wants, "the evaluation is a three-way ablation on
SlideVQA scored on accuracy, grounding overlap and abstention quality", is
never written.

The abstain path is worse. The reader returns an empty list and the UI renders
"No claims to show", which reads as a failure of the software rather than as a
decision the software made on purpose.

## 3. The structural problem

Verification runs after drafting. The reader writes a connected answer, the
verifier removes some sentences, and the UI joins what is left. Any sentence
can disappear, and the ones that remain have to still read.

That rules out the normal way prose coheres. Consider:

```
drafted   1 The evaluation compares three system variants on SlideVQA.
          2 Each of them is scored on three metrics.
          3 It also isolates the added layer through an ablation.

1 withheld  "Each of them is scored on three metrics. It also isolates the
             added layer through an ablation."
```

Grammatical, and meaningless. "Each of them" and "It" point at nothing. The
paragraph does not announce that it is broken; it just quietly says something
other than what was verified.

## 4. The resolution: chain by noun phrase, never by pronoun

Human prose flows because each sentence opens on information the previous one
ended with. That is worth keeping. The failure above is not caused by flow, it
is caused by achieving flow through anaphora.

Repeating the noun phrase gives the same flow and survives deletion:

```
1 The evaluation compares three system variants on SlideVQA.
2 Each of the three variants is scored on answer accuracy, grounding overlap,
  and abstention quality.
3 Grounding overlap is measured as intersection over union against the
  annotated region.
```

Sentence 2 opens on "three variants", where 1 ended. Sentence 3 opens on
"grounding overlap", where 2 ended. Delete 2 and the remainder is:

```
The evaluation compares three system variants on SlideVQA. Grounding overlap is
measured as intersection over union against the annotated region.
```

A topic jump, and nothing worse. The cost is mild repetitiveness, which is how
careful technical writing reads anyway.

## 5. The four reader rules

1. **The first claim answers the question directly.** Not context, not what the
   page covers. Someone reading only that sentence has the answer.
2. **Each claim asserts exactly one thing.** Unchanged from today, and it is a
   grounding requirement rather than a style one: a claim is matched to a single
   region, and a claim asserting two things cannot be evidenced by one region.
3. **Each claim stands alone.** No sentence opens with a pronoun or a
   demonstrative, and no sentence refers to "the above" or "the previous point".
4. **Each claim connects to the one before it by repeating a noun phrase from
   its end.**

Plus a register instruction. The reader currently writes like an extractor
describing a page ("Figure 5.1 includes an evaluation step on SlideVQA with
three metrics"). It should write like a person answering a colleague ("The
evaluation runs on SlideVQA and reports three metrics").

### 5.1 The reader prompt

```
You are answering a question from one page of a document, for someone who
cannot see the page.

Answer only from what is visible on this page. If the page does not answer the
question, return an empty list of claims.

Compose the answer as connected prose, then return it as one sentence per
claim, in the order they should be read. Every sentence is checked separately
against the page, and any sentence may be removed before the answer is shown,
so the sentences must obey four rules.

1. The FIRST sentence answers the question directly. Not background, not what
   the page is about. Someone who read only that sentence should have the
   answer.

2. Each sentence asserts exactly ONE thing. Each sentence is matched to a
   single region of the page as its evidence, and a sentence asserting two
   things cannot be evidenced by one region.

3. Each sentence stands on its own. Never begin a sentence with it, they, them,
   this, these, those, such, or Additionally, and never refer to the previous
   sentence or to anything above. Repeat the noun instead.

4. Connect each sentence to the one before it by repeating a noun phrase from
   the end of that sentence, never by a pronoun. "The evaluation compares three
   variants. Each of the three variants is scored on ..." reads as one answer.
   "The evaluation compares three variants. Each of them is scored on ..."
   becomes nonsense the moment the first sentence is removed.

Write the way you would answer a colleague who asked you out loud. Do not
describe the page. State what it says.

Set starts_paragraph on a sentence that opens a new topic, and leave it false
otherwise. Most answers are a single paragraph.

Question: {question}
```

## 6. Schema change

`ClaimList` becomes a list of objects rather than a list of strings:

```python
class DraftedClaim(BaseModel):
    text: str = Field(min_length=1)
    starts_paragraph: bool = False
```

`starts_paragraph` is metadata, not text. It adds nothing unverified to the
screen and it leaves every sentence mapped to exactly one region, which is what
keeps hover-to-region and click-to-evidence working. It exists because an answer
long enough to cover two topics reads badly as one block.

This is a breaking change to the reader's return type. `read()` currently
returns `list[str]` and `answer_stream` iterates it. `Claim` in `contracts.py`
gains the same boolean so it survives to the wire.

## 7. Mechanical checks

These make "semantic flow" measurable rather than a matter of taste. Both live
in `reader.py` beside `is_compound`, and both follow its philosophy: flag, do
not reject. Dropping a claim would lose verified content, and the useful
response is to surface the rate in the S7 eval and tune the prompt until it is
near zero.

**`opens_with_anaphora(claim)`** matches a sentence-initial pronoun,
demonstrative or bare additive/resultative connective:

```python
_PAGE_DEICTIC = r"page|document|slide|figure|table|chart|section|paragraph"

_ANAPHORA = re.compile(
    r"^\s*(?:it|its|they|them|their|such|"
    r"additionally|also|furthermore|moreover|however|"
    r"therefore|hence|consequently|instead|"
    rf"(?:this|these|those|that)(?!\s+(?:{_PAGE_DEICTIC}))"
    r")\b",
    re.IGNORECASE,
)
```

The demonstrative group carries a negative lookahead that exempts it before a
page-deictic noun. "This page", "this table", "this figure" point at the
image the reader is looking at, not at the previous sentence, and survive
their predecessor being withheld completely intact; only a demonstrative that
is not pointing at the page dangles. `there` is deliberately excluded as an
expletive subject, not a referring pronoun, so "There are three variants" is
self-contained. Bare quantifier openers ("Both are scored", "Each of them")
are a known miss, not caught: the shape splits between "Both are scored on
three metrics" (dangles) and "Both variants are scored" (does not), and a
flat word list cannot tell those apart.

**`shares_content_word(previous, claim)`** is the chaining proxy. Lowercase,
strip punctuation, drop a closed stopword list, strip a trailing "s", and test
for a non-empty intersection. Crude, and it fails loudly in the case that
matters: a reader that reverts to listing disconnected facts produces adjacent
claims with no shared content word.

**Dangling references are a display-time property, not a drafting one.** A claim
that opens with anaphora is only a problem when the claim before it was
withheld. That combination is what S7 counts. It is not suppressed at display
time: the claim is verified content with a real region, and removing verified
evidence to protect the look of a paragraph inverts this project's priorities.
If the measured rate is not near zero, the fix is the prompt.

## 8. The lead rule and the three states

The lead claim is the first one the reader drafted, by index. If the lead is
withheld, the answer abstains as a whole, even when supporting claims passed.
Supporting detail without the answer is the fact dump this document exists to
remove, and presenting it as an answer would claim to have answered a question
it did not.

| State | Condition | Surface |
|---|---|---|
| Answered | lead shown, nothing withheld | the joined paragraph |
| Partial | lead shown, at least one claim withheld | the joined paragraph, plus a withheld count |
| Abstained | lead withheld, or no claims at all | the decline, plus surviving claims reframed as context |

`Answer.abstained_overall` gains the lead condition. Half of this was already
fixed on 2026-08-10, ahead of this spec, and the rest builds on that fix.

**Already done.** `abstained_overall` was a stored field that `core.py` filled
in with `all(c.abstained for c in claims)`, a different predicate from the
`Claim.withheld` that `Answer.shown` filters on. `withheld` is broader by
design: it is also true when no verdict exists, because absence of a verdict is
not a passing verdict. The model's own default said `False` for an empty answer
while `core.py` computed `True` for the same object. It is now a `computed_field`
returning `not self.shown`, so the rule is stated once and no caller can record
an abstention the claims contradict.

That defect was **latent, not live**. Every claim on the `answer_stream` path
gets a verdict, so `abstained` and `withheld` agreed by accident and no bad
screen was reachable. It was worth fixing because a safety guarantee resting on
a caller's habits is one refactor from being wrong, not because it was
misbehaving.

**Still to do.** The lead condition joins it in the same property:

```python
@computed_field
@property
def abstained_overall(self) -> bool:
    return not self.shown or (bool(self.claims) and self.claims[0].withheld)
```

The lead is `claims[0]`, and `claims` holds every claim in drafted order
including withheld ones, so the lead is still identifiable after the gate has
removed it.

## 9. What fixed copy is allowed to say

**Fixed copy may assert facts about the system's own process. It may never
assert facts about the document.**

"I was not able to confirm an answer from this page" is introspection. It is
true by construction whenever it is shown, because the system is the thing that
failed to confirm it. "This page does not mention the metrics" is a claim about
the world. It has no region, no model judged it, and the system never says it.

That line is what lets the abstain copy be warm and humble without being
dishonest. Every sentence below is on the safe side of it.

### 9.1 Partial

The paragraph, then, quietly:

> I left out 2 statements I could not confirm against this page.

### 9.2 Abstained

> **I could not answer this from this page**
>
> I read proposal.pdf page 14, but I was not able to confirm an answer to your
> question from what is on it. Rather than give you something I cannot stand
> behind, I have left it out.
>
> *Here is what I was able to confirm from the page, in case it helps:*
> - The page defines three system variants. ¹
> - Ground truth is derived automatically. ²
>
> Retrieval also ranked page 19 and page 3.  `[read page 19]`

The bulleted claims are real verified claims with real regions, still hoverable
and still zoomable. Only the framing sentences are fixed copy. When no claim
survived at all, the context block is omitted and the decline stands alone.

## 10. The verifier

None of the above matters if the verifier does not reject anything. Across the
live runs observed on 2026-08-10, roughly 30 of 32 claims came back `supported`
at 90 to 100 percent confidence. That is consistent with a genuinely easy
corpus, where the reader only claims what it can see, and it is equally
consistent with a lenient judge. Nothing currently distinguishes the two, and
if it is the second then the lead rule never fires and the abstain copy is never
seen.

**The strictness probe runs before any prompt tuning.** It feeds the verifier
real regions from a real page with claims that are false in four specific ways,
and asserts none is labelled `supported`:

1. **A changed number.** "The evaluation reports seven metrics" against a region
   establishing three.
2. **A swapped entity.** "The evaluation runs on DocVQA" against a region naming
   SlideVQA.
3. **Absent content.** "The system uses conformal calibration" against any region
   on a page that does not discuss calibration.
4. **True of the page, absent from the regions.** A claim the page supports
   elsewhere, paired with regions that do not establish it. This is the case
   that separates a judge reading the evidence from one reading the image, and
   it is the one most likely to fail today.

### 10.1 The verifier prompt

```
You are checking whether a claim is supported by specific evidence from a
document page. You did not write the claim. Your job is to catch claims that
should not be shown to a user.

Claim: {claim}

Evidence regions selected from the page:
{evidence}

Judge the claim against the EVIDENCE REGIONS, not against the page as a whole
and not against what you already know. If the regions do not establish the
claim, the label is not supported, even when the claim looks correct and even
when you can see it elsewhere on the page.

Choose exactly one label:
- supported: the evidence regions establish the whole claim
- partially_supported: the regions establish part of the claim, or establish it
  only with a qualification the claim leaves out
- unsupported: the regions contradict the claim, or are about something else
- insufficient_evidence: there is not enough here to judge

supported is the strongest label and the only one shown by default. Do not use
it for a claim you merely believe to be true. Use it when you can point at the
evidence and say the claim follows from it.

Give a confidence between 0 and 1, and one sentence of reasoning that names the
part of the evidence you relied on.
```

## 11. Testing

Unit, no network:

- `opens_with_anaphora`: matches each listed opener, and does NOT match a
  sentence merely containing one ("The variant that supports it is Grounded
  RAG").
- `shares_content_word`: true for a real chained pair, false for two unrelated
  claims, and false when the only overlap is a stopword. The stopword case is
  the one that makes the check meaningful rather than always-true.
- Deletion robustness: given a drafted answer fixture, remove each claim in turn
  and assert no remaining claim opens with anaphora whose predecessor is gone.
- `Answer.abstained_overall` is true when the lead is withheld and supporting
  claims passed. This is the lead rule, and it is the one behaviour a reviewer
  is most likely to consider a bug.
- `Answer.abstained_overall` is true when every claim is unverified. Already
  written as `test_an_answer_with_no_verdicts_abstains`, along with
  `test_an_answer_with_no_claims_abstains` and
  `test_the_abstention_flag_cannot_be_set_against_the_claims`. Restoring the old
  predicate fails the first of the three, checked by mutation.
- `wire.py` emits `starts_paragraph`, and a withheld claim still carries no
  regions.

Live, marked slow, one reader call:

- On a known page, the first claim answers the question, no claim opens with
  anaphora, and adjacent claims share a content word.

Live, marked slow, verifier:

- The four strictness probes in section 10. Assert `label != "supported"`, not a
  specific label: which of the three non-supported labels is correct for a given
  probe is a judgement call, and pinning it would make the test brittle for no
  gain.

## 12. Files

| File | Change |
|---|---|
| `agent/reader.py` | new prompt, `opens_with_anaphora`, `shares_content_word`, `read()` returns drafted claims |
| `agent/schemas.py` | `DraftedClaim`, `ClaimList` holds objects |
| `agent/verifier.py` | new prompt |
| `agent/core.py` | carry `starts_paragraph` onto `Claim`, lead-aware `abstained_overall` |
| `contracts.py` | `Claim.starts_paragraph`, lead rule on `Answer` |
| `api/wire.py` | emit `starts_paragraph` |
| `frontend/lib/api.ts` | `starts_paragraph` on `ClaimEvent` |
| `frontend/components/AnswerPanel.tsx` | paragraph breaks, partial-state line |
| `frontend/app/page.tsx` | the three states and their copy |

## 13. Non-goals

- **A composer model.** Rejected during design. A third call that rewrites
  verified claims into fluent prose reads best and breaks the sentence to region
  mapping, which is the interaction the product is built on.
- **Multi-page synthesis.** The reader still sees one page. Comparing what two
  pages support is the user's job, through the retrieved-pages rail.
- **Calibrated confidence.** Self-reported confidence is not calibrated and the
  report must keep saying so. Conformal calibration is named future work in
  `proposal.tex` line 381.
- **Suppressing a dangling claim at display time.** Section 7 states why.

## 14. Assumptions carried, not verified

- `starts_paragraph` is worth its schema change. If real answers turn out to be
  three or four sentences without exception, it is dead weight and should be
  removed rather than kept for a case that does not occur.
- The lead claim is the first by index. This holds only while the reader returns
  claims in reading order, which the prompt requires and nothing enforces.
- Adjacent content-word overlap is a usable proxy for chaining. It is a proxy.
  A reader could satisfy it while writing badly, and the live test is a floor
  rather than a measure of quality.
