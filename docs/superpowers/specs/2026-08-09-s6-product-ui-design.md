# S6: Product UI

Date: 2026-08-09
Status: design approved, not implemented
Depends on: S2 (page images, boxes), S3 (retrieval), S4 (grounding), S5 (reader, verifier, abstention)

## 1. What this slice delivers

A question typed in a browser comes back as an answer with the supporting region
drawn on the page image, or as an abstain badge.

Nothing new is reasoned here. S6 exposes what S3 through S5 already compute, and
it is the first place a person who has not read the code can see the three
pillars at once: a region on a page, a verdict on a claim, and a system that
declines to answer.

It is also the first place the online pipeline exists end to end. `vvrag ask`
requires the caller to name a document and a page; `vvrag search` ranks pages and
stops. The seam between them has never been joined, because the CLI had no reason
to. A user types a question, not a page number, so the service has to join it.

## 2. Why the split is a FastAPI service plus a Next.js frontend

`proposal.tex` states that the grounding and verification logic is a
dependency-light core module reusable independently of the user interface, with
the product layer and the evaluation layer consuming it through the same
interface.

A JavaScript frontend physically cannot import the Python module. That makes the
boundary the proposal describes real rather than asserted: a single-process
Python UI could always have reached past it, and nothing would have caught that.

`proposal.tex` does not name a UI framework, so this choice contradicts nothing
in the graded deliverable. Two earlier internal design notes mention Streamlit
and are superseded.

## 3. Scope

The service is **read-only over a corpus built beforehand**. `vvrag ingest` and
`vvrag embed` stay CLI commands run ahead of time.

Upload was considered and rejected on hardware grounds. Embedding measures about
21 s per page on this card, and the card is single-tenant with 3.63 GB. An upload
job would contend with the query-time embedder, so asking a question would block
while any document indexed. Ingest-only upload (render and extract boxes on the
CPU, embed later) avoids the contention but leaves a document that is visible and
unsearchable, which is a state with no honest explanation in a defense.

## 4. Architecture

```
frontend/                       Next.js, TypeScript. Talks HTTP only.
      |  fetch + SSE
src/visual_verify/api/          FastAPI. Holds the models. Owns HTTP.
      |  plain Python calls
src/visual_verify/pipeline.py   prepare_page(): doc, page -> boxes, vectors, grid, image
src/visual_verify/agent/core.py answer_stream(): the claim loop, yielding events
```

The API layer contains no grounding, no rubric arithmetic, and no threshold
comparison. It converts events into SSE frames and serves PNGs. That is the whole
job, and it is what keeps the guarantee in one place.

`api` becomes a fourth optional extra (`fastapi`, `uvicorn`). `fastapi`,
`starlette`, and `uvicorn` join `FORBIDDEN` in `tests/test_core_is_light.py`, so
the core stays at its four dependencies and the guard keeps running in a
subprocess.

**No `sse-starlette`.** An SSE frame is `event: x\ndata: {...}\n\n`. Hand-rolling
that is about ten lines, is directly unit-testable, and is one less dependency in
a repository that gates everything behind extras.

### 4.1 Why `answer_stream` rather than a callback or a second loop

`agent/core.py` gains `answer_stream(...)`: the same read, ground, verify loop,
yielding an event per claim. `answer()` becomes a short drain over it and keeps
its exact signature and return type, so S5's tests, `vvrag ask`, and S7's eval
harness are untouched.

Two alternatives were rejected.

**A second loop in the service** would put the `GroundingError` recovery, the
same-model guard, and the `score < threshold` comparison in two places, and the
second copy is the one users actually hit. Every S5 test would keep passing while
the product diverged. This repository's recorded failures are almost all of that
shape, where the tested path and the real path came apart silently.

**An `on_claim` callback** is a smaller diff but solves half the problem. It
inverts control against an async SSE generator, and it cannot emit the
`retrieved` or `reading` events at all, because those happen outside `answer()`.
The orchestration extraction would still be needed.

`pipeline.prepare_page(...)` lifts the doc, page, boxes, vectors, and grid
assembly out of `cmd_ask` (`cli.py:520`), and `cmd_ask` is rewritten to call it.
The CLI and the service then differ only in what they do with the result.

## 5. Model residency and concurrency

`lifespan` loads once at startup and holds on `app.state`: the ColQwen2 embedder,
the `QdrantIndex`, the SQLAlchemy engine, and both `CachedChat` clients. Startup
therefore costs about 20 s and the first request is fast, which is the inversion
the roadmap asks for: each `vvrag search` invocation currently reloads ColQwen2 at
about 20 s and 2.6 GB, and a request-scoped embedder would make the demo unusable.

Three constraints follow, all enforced rather than documented.

**Single worker.** Two workers means two ColQwen2 copies and an immediate OOM on
a 3.63 GB card. The run command pins `--workers 1` and startup logs the worker
count.

**One `/ask` at a time.** The GPU is single-tenant. An `asyncio.Semaphore(1)`
serialises the ask handler, so a second concurrent request waits rather than
racing into the embedder. It is released in a `finally`, or a disconnected client
deadlocks every later request.

**The generator runs in a worker thread.** `answer_stream` is synchronous and
does GPU work plus blocking HTTP to two providers. Iterating it inside an async
endpoint would freeze the event loop for the full duration, so `/health` would
hang and the browser would receive nothing until the end, which defeats the
reason for streaming. It runs via `anyio.to_thread`, pushing events into an
`asyncio.Queue` that the SSE response drains.

## 6. Endpoints

| method | path | returns |
| --- | --- | --- |
| `GET` | `/health` | model ids, whether the embedder is resident, indexed page count |
| `GET` | `/documents` | ingested documents: sha, filename, page count |
| `GET` | `/documents/{sha}/pages/{n}/image` | the rendered PNG |
| `POST` | `/ask` | `text/event-stream` |

`/ask` is a POST consumed with `fetch` and a `ReadableStream` reader, not
`EventSource`. `EventSource` is GET-only, which would force the question into a
query string and give up request bodies for `doc`, `page`, `k`, and `threshold`.
Streaming a POST response is well supported and needs no extra machinery.

Image serving takes a sha and an integer page and resolves the filename through
the database, so no user-supplied string ever reaches a filesystem path.

The `/ask` body is `{question, doc?, page?, k?, threshold?}`. `k` defaults to 5
and `threshold` defaults to `Settings.abstain_threshold`, so the service honours
`VVRAG_ABSTAIN_THRESHOLD` exactly as the CLI does and no module hardcodes either
value. A non-finite `threshold` is a `422`, matching the check `cmd_ask` already
performs. `doc` and `page` must be supplied together or not at all.

## 7. Retrieval in the loop

The service runs MaxSim retrieval at `k=5`, reads the top-1 page, and returns the
remaining candidates in the `retrieved` event so the UI can show what was
considered.

The candidate list is the escape hatch. Retrieval can miss, and without it a miss
becomes an unexplainable wrong answer with no recourse in front of an examiner.
Clicking a candidate re-asks with an explicit `doc` and `page`; retrieval is
skipped and `retrieved` still fires, pinned, with an empty candidate list. One
code path, one event shape.

## 8. The event sequence

```
event: retrieved  {doc_sha, doc_name, page, score, candidates:[...]}
event: reading    {}
event: claims     {n: 3}
event: claim      {index, text, label, confidence, reason, compound, withheld, regions:[...]}
event: claim      {...}
event: claim      {...}
event: done       {shown: 2, withheld: 1, abstained_overall: false}
```

S5 section 9 ruled out streaming the reader's tokens, because that would put a
claim on screen before the verifier judged it. It explicitly left open streaming
**verified** claims one at a time, which is what this is. Nothing reaches the
browser before its verdict exists.

The alternative was a single blocking JSON response. One query embed, one reader
call, then a grounding and a verifier call per claim is four hosted round trips
for a three-claim answer, so a blocking response is tens of seconds of dead air
with no indication of whether the service hung. The stage events turn that into
the pipeline explaining itself, which is the demonstration.

Job-plus-polling was rejected as strictly more machinery for strictly less, in a
single-user single-worker service.

### 8.1 Two consequences

**`Claim` gains `reason`.** `verify()` returns a one-sentence reason and
`answer()` currently drops it (`agent/core.py:95`), yet the withheld panel is
built entirely around showing it. `contracts.py` gains
`reason: str | None = None` and `answer()` carries it through. That is an
additive optional field, which the contracts docstring explicitly permits.

**Regions are stripped from a withheld claim at the API boundary, not in the
core.** The event serialiser drops `regions` when `abstained` is true, so the
browser never receives geometry it must be trusted not to draw. Enforcing it
inside `answer()` instead would break S7, which needs the regions of rejected
claims to compute IoU against coverage. The guarantee belongs at the point where
the data leaves the process, not at the point where it is computed.

## 9. Frontend

One page. Question box on top, answer column left, page image right.

Regions are absolutely-positioned `div`s over an `<img>`, sized in percentages
directly from the normalized 0-1 bbox. No canvas: percentages rescale for free at
any display size, which is the reason S1 normalized the coordinate convention in
the first place. Hovering a claim highlights its regions, and each claim has its
own colour.

Three fields drive rendering that no other surface exposes:

- `resolution: "block"` draws a dashed outline instead of a solid one, with a
  tooltip saying the lines inside the block could not be separated. S4 added the
  field so a coarse fallback is distinguishable from a confident line hit, and
  this is the only place a human ever sees the difference.
- `modality` distinguishes an exact text-layer span from a snapped visual region.
- `compound` marks a claim asserting more than one thing, whose single region can
  only evidence part of it.

### 9.1 Abstention

If `abstained_overall`, the abstain badge is the primary result in the answer
column, not an empty list with a footnote. It is visually distinct from a
low-confidence supported claim: different shape, different colour, its own row.
A badge that reads as "weak answer" rather than "no answer" would misrepresent
the one behaviour this project exists to demonstrate.

Rejected claims appear in a collapsed panel below the answer: the claim text, its
rubric label, its confidence, and the verifier's reason. **No region is drawn for
a rejected claim.**

This differs deliberately from `vvrag ask`, which shows every claim. The CLI is a
diagnostic surface and its default is right for that. The UI is the product
surface, so it uses `Answer.shown`, which requires a verdict rather than merely
`not abstained`. Same data, opposite default, and the difference is intentional.

Showing the label and reason but withholding the region is the line that matters.
The reason string is what S5 built to make a wrong verdict debuggable, and hiding
it would make the count meaningless in the only surface an examiner will look at.
Drawing the region would point at evidence the verifier refused, which is the
precise failure the system exists to prevent, and styling is not a guarantee.

## 10. Error handling

| situation | behaviour |
| --- | --- |
| API key missing, or provider unknown | fails at **startup**, not on first ask |
| reader and verifier are the same model | same, at startup, from `answer()`'s existing guard |
| nothing indexed | `409` with a sentence naming `vvrag embed` |
| retrieval returns no page | `done` with `abstained_overall: true`; not an error |
| provider fails mid-stream | `error` event, then close; claims already delivered stay on screen |
| client disconnects | queue drain stops; the semaphore is released in `finally` |
| unknown document, page, or image | `404` |

Startup-time failure for misconfiguration is the deliberate one. A service that
comes up and fails on the first question looks healthy to `/health` and to
anybody watching it start, which is the worst moment to discover a missing key is
during a demo.

## 11. Testing

API tests run against `FakeChat`, `FakeEmbedder`, and an in-memory Qdrant, with
the lifespan resources dependency-overridden. No GPU, no key, no network, so they
run in CI alongside the existing suite and do not add a fourth model-loading test
module, which the recorded CUDA fragmentation finding says would need process
separation.

Four tests carry the weight:

1. **A withheld claim's frame carries no regions.** Mutating the serialiser to
   include them must fail this test. It is the one guarantee the frontend cannot
   re-derive from anything else it receives.
2. **`answer()` and `answer_stream()` agree.** The same `FakeChat` script produces
   an identical `Answer` either way, which pins the drain so the generator
   refactor cannot silently diverge from merged S5 behaviour.
3. **Every `claim` event carries a non-null `label`.** A claim reaching the wire
   unverified is the failure the streaming decision exists to avoid.
4. **Overlay coordinates**, in the frontend: a normalized bbox maps to the
   expected percentages, and an x/y transposition must fail the assertion. S3
   shipped a patch-grid transposition that looked entirely correct and was found
   by constructing the failing case, not by a test failing on its own.

SSE framing gets its own unit tests for multi-line and unicode payloads, since a
`data:` field containing a newline silently truncates the event.

The startup guards (single worker, missing key, identical models) are tested by
constructing the app with bad settings and asserting it refuses, not by reading
the log.

## 12. What S7 gets

One field. S7 consumes `answer()` in Python and is unaffected by the streaming
variant, by design: that is what test 2 above exists to hold. The `reason` added
in section 8.1 is the change it sees, and it makes a wrong verdict explicable in
the eval output rather than only in a browser.

S7 also keeps the regions of rejected claims, which section 8.1 strips only at
the API boundary. Confident-wrong against coverage cannot be computed without
them.

## 13. Out of scope

No upload or in-browser ingestion, per section 3. No multi-page answers; one page
per question, matching S5. No authentication, no multi-user session state, and no
deployment beyond a local single-worker run. No streaming of reader tokens, per
S5 section 9. No conformal calibration; the proposal names it future work.
