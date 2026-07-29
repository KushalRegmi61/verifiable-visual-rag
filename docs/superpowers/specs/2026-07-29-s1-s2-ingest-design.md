# S1 + S2 — Project Skeleton and Ingest Pipeline

**Date:** 2026-07-29
**Status:** Approved, ready for implementation planning
**Project:** Verifiable Visual RAG (BE Minor Project, IOE Pulchowk BCT)

---

## 1. Context: where this slice sits

The full system is four lanes (retrieval, core + agent, product, eval), which is far too
large for one spec. It is cut into seven slices, each with its own spec, plan, and
implementation cycle:

| # | Slice | Delivers | GPU? | Depends on |
|---|---|---|---|---|
| **S1** | Skeleton + contracts | uv project, `visual_verify` package, frozen Pydantic models, test harness | No | — |
| **S2** | Ingest pipeline | PDF to page images, text-layer candidate boxes, persistence, CLI | No | S1 |
| S3 | Retrieval index | ColQwen2 embeddings, Qdrant multivector MaxSim | Yes (batch) | S2 |
| S4 | Grounding core | `ground()`: text-span path, then visual snap-to-box | Query-embed | S2, S3 |
| S5 | Reader + verifier | Atomic claims, independent judge, abstention gate | No | S4 |
| S6 | Product UI | Streamlit: answer, highlighted regions, abstain badges | No | S5 |
| S7 | Eval harness | SlideVQA, auto-gold boxes, EM/F1 + IoU + confident-wrong, ablation | No | S4, S5 |

**This spec covers S1 and S2 only.** They are merged because together they are roughly 400
lines of code, and splitting them would mean two ceremony cycles for one coherent deliverable.

S1 + S2 are chosen as the starting point for two reasons. They are entirely GPU-free and
model-free, so they are testable deterministically and never blocked on a shared campus GPU
slot. And S2 produces the text-layer candidate box set, which both grounding paths in S4
consume and which the S7 eval derives its gold boxes from. It is the genuine foundation, not
merely the easy start.

---

## 2. Scope

**In scope**

- uv-managed Python project, single package with optional-dependency extras
- Frozen public contracts (`GroundedRegion`, `Claim`, `Answer`, `RetrievedPage`)
- Born-digital gate, content-hash fingerprinting, deduplication
- Page rendering at fixed DPI
- Word-level and table-cell box extraction, normalized coordinates
- SQLAlchemy 2.0 persistence with Alembic migrations
- `vvrag` CLI: `ingest`, `status`, `inspect`
- Test suite and lint configuration

**Explicitly out of scope**

- Embeddings, Qdrant, any vector search (S3)
- Any model call, any API key (S3 onward)
- Grounding or verification logic (S4, S5)
- FastAPI, web endpoints, background job queues (S5, S6)
- OCR of any kind (permanently out of scope: born-digital only)

The slice must remain runnable on a laptop with no GPU, no network, and no credentials.

---

## 3. Decisions and rationale

### 3.1 Repo layout: single package with extras

One `pyproject.toml` at the repo root. The core dependency list stays at **pydantic,
pymupdf, pillow, numpy**. Everything vendor-specific lives in optional extras.

```
verifiable-visual-rag/
  pyproject.toml
  alembic.ini
  migrations/                    # Alembic
  src/visual_verify/
    contracts.py                 # frozen public Pydantic models
    config.py                    # env-driven settings, no hardcoded URLs
    ingest/
      gate.py                    # born-digital + fingerprint
      render.py                  # page to PNG at fixed DPI
      boxes.py                   # word + table-cell extraction
      pipeline.py                # orchestration, resumability
    store/                       # SQLAlchemy, behind the `store` extra
      models.py
      repository.py
    derive.py                    # line/block/span union helpers
    cli.py                       # `vvrag`
  tests/
  data/                          # gitignored: index.db, page images
  proposal_report/ presentation/ research/ ...   # untouched
```

Rejected: a uv workspace monorepo (stricter boundary, but too much ceremony for a
three-person one-month build) and a flat single dependency list (the boundary would exist
only as a rule people remember, which the HLD explicitly warns against).

### 3.2 The core stays dependency-light, enforced mechanically

The HLD's day-one rule is that `visual_verify` is standalone and publishable, and that the
core never imports FastAPI, Qdrant, or a vendor SDK.

`ground()` and `verify()` take **data** (a page image and a list of boxes), never a database
handle or a client object. The store is therefore not part of the shippable core's API
surface: `visual_verify/store/` sits behind the `store` extra and is imported by the ingest
CLI and later the app, but never by the grounding path.

`tests/test_core_is_light.py` enforces this. It runs a **subprocess** that imports
`visual_verify` and asserts `sqlalchemy` is absent from `sys.modules` afterwards. The
subprocess matters: the test environment necessarily has SQLAlchemy installed in order to
test the store at all, so an in-process check could only prove the package is uninstalled,
not that the core avoids importing it. The subprocess check proves the real property — that
importing the core does not transitively pull in the store — and holds in a single
environment. The boundary is a test, not a convention.

### 3.3 ORM: SQLAlchemy 2.0 + Alembic

SQLAlchemy 2.0 typed declarative models (`Mapped[...]`, `mapped_column()`) with Alembic for
migrations.

SQLModel was considered and rejected. It is a thin wrapper over SQLAlchemy, so SQLAlchemy is
underneath either way; it has no migration story of its own and defers to Alembic; and its
main selling point is a single class serving as both wire schema and table. This design
deliberately separates those (`contracts.py` versus `store/models.py`), so that benefit does
not apply, leaving only the cost of a dependency with a smaller maintenance team that lags
SQLAlchemy releases.

Raw stdlib `sqlite3` was also considered. It is leaner, but gives no typed models, no
migration tooling, and locks queries to the SQLite dialect, making the later Postgres move a
rewrite.

### 3.4 Storage backend: local now, cloud by config

Development runs on local SQLite. Neon and Qdrant Cloud are deferred, not rejected.

Ingest is write-heavy (roughly 500 word rows per page, so a 300-page corpus is about 150k
inserts), and doing that over network latency to a managed Postgres would slow the dev loop
while buying nothing at this stage. Qdrant has nothing to hold until S3.

All connection details come from config, so the switch is an environment variable:

```
VVRAG_DB_URL=sqlite:///data/index.db            # dev default
VVRAG_DB_URL=postgresql+psycopg://...neon.tech  # deploy, S6
VVRAG_QDRANT_URL=http://localhost:6333          # S3 dev
VVRAG_QDRANT_URL=https://...qdrant.io           # deploy
```

No module hardcodes a connection string. Inserts are bulk, not per-row ORM adds, so the
Postgres path stays viable.

Known limits for later: Neon's free tier is 0.5 GB, adequate for box rows. Qdrant Cloud's
free 1 GB holds roughly 300 pages of ColQwen2 multi-vectors (about 350 KB per page) before
quantization becomes necessary. Local Docker has neither limit.

### 3.5 Box granularity: words plus parent hierarchy

Boxes are extracted and stored at **word** level, each carrying its `block_no`, `line_no`,
and `word_no`, plus table-cell boxes from `find_tables()`.

`page.get_text("words")` returns exactly
`(x0, y0, x1, y1, word, block_no, line_no, word_no)`, so the hierarchy needs no extra
bookkeeping.

Coarser granularities are **derived by grouping at query time, not stored**:

- line box = union of words sharing `(block_no, line_no)`
- block box = union of words sharing `block_no`
- span rects = the words covering a given answer substring, **split at line boundaries into one rect per line**

**Revised during implementation:** the span path originally returned a single union rect. Measurement showed that a match wrapping across a line break then produced a rect 5.7x the true ink area, enclosing every word on a two-line fixture page when the answer was two of them (IoU ceiling 0.176). Because this same function generates the evaluation harness's gold boxes, that is fabricated ground truth rather than mere imprecision. It now returns `list[BoxRecord]`, one rect per line the match spans. Matching still runs over the flat reading-order token sequence, so wrapped answers are still found; only the returned geometry is split.

This matters downstream in three ways. Snap-to-box in S4 can be retuned to rank a different
candidate set (words, lines, or merged spans) without re-ingesting the corpus. The text-span
path can return a sub-line region, which a line-only store could never recover. And the S7
eval's auto-derived gold box requires locating an arbitrary answer substring and unioning its
word boxes, which is only possible at word granularity.

Rejected: lines only (a one-number answer inside a long line over-covers, dragging IoU down,
and sub-line spans are unrecoverable) and lines-plus-words-on-demand (reintroduces per-query
PDF re-parsing, defeating the offline/online split).

### 3.6 Interface: library plus CLI

```bash
uv run vvrag ingest proposal_report/proposal.pdf
uv run vvrag ingest --dir references/
uv run vvrag status
uv run vvrag inspect <doc> --page 3 --overlay out.png
```

`inspect --overlay` renders the page with its candidate boxes drawn on top. This is the
human check that box extraction is actually correct, and it is expected to be the highest-
value debugging tool during S4.

No FastAPI in this slice: the async upload endpoint has no consumer until the UI exists at
S6, and pulling in web dependencies would compromise this slice's defining property of being
GPU-free and deterministic.

---

## 4. Architecture

```
vvrag ingest <pdf>
      |
      +-- [1] gate      sha256 fingerprint; reject if no text layer;
      |                 skip if hash already ingested
      |
      +-- [2] render    PyMuPDF get_pixmap at fixed 150 DPI -> PNG per page
      |
      +-- [3] boxes     get_text("words") -> (x0,y0,x1,y1,word,block,line,word_no)
      |                 + find_tables() -> table-cell boxes
      |                 normalize all bboxes to 0-1 against the page rect
      |
      +-- [4] persist   bulk insert; mark page done (resumable)
```

Three invariants this pipeline exists to guarantee:

**Fixed 150 DPI and normalized coordinates.** S4's heatmap-to-patch-to-pixel math requires
ingest and display to agree on geometry, and the frontend must scale boxes to an arbitrary
viewport. Storing pixel coordinates would couple every consumer to the render DPI.

**Resumability keyed on content hash.** A 300-page batch that fails at page 200 must not
redo 200 pages. Re-ingesting an unchanged PDF is free.

**Boxes and images are the only outputs.** No embeddings, no derived text analysis. Anything
a later slice might want to recompute differently stays uncomputed here.

---

## 5. Data model

Two families, kept separate on purpose.

### 5.1 Frozen public contracts (`contracts.py`)

Pure Pydantic v2, taken verbatim from the HLD. Nothing in this slice produces them yet;
freezing them now is the point, because S3 through S7 build against them.

```python
GroundedRegion(page: int, bbox: tuple[float,float,float,float], score: float,
               modality: Literal["visual","text"],
               crop_ref: str | None = None,
               text: str | None = None)
Claim(text: str, regions: list[GroundedRegion], confidence: float, abstained: bool)
Answer(question: str, claims: list[Claim], abstained_overall: bool)
RetrievedPage(doc_id: str, page: int, image_ref: str,
              text_layer: str | None, score: float)
```

**One deliberate deviation from the HLD:** `GroundedRegion.bbox` is typed
`tuple[float, float, float, float]` and is **normalized to 0-1**, where the HLD wrote
`tuple[int, int, int, int]`. The HLD's integer type implies pixel coordinates, which would
put two coordinate systems in the codebase: normalized in the store, pixels on the wire.
That is exactly the coupling to render DPI that normalizing exists to prevent, and the
frontend wants normalized values anyway in order to scale to any viewport. One coordinate
system, normalized, everywhere. Pixel conversion happens only at the point of drawing.

### 5.2 Store models (`store/models.py`)

SQLAlchemy 2.0 declarative. This slice's actual output.

| Table | Fields |
|---|---|
| `documents` | `sha256` (PK), `path`, `n_pages`, `status`, `created_at` |
| `pages` | `id` (PK), `doc_sha` (FK), `page_no`, `image_path`, `width_px`, `height_px`, `dpi` |
| `boxes` | `id` (PK), `page_id` (FK), `kind` (`word` \| `table_cell`), `x0`, `y0`, `x1`, `y1` (normalized), `text`, `block_no`, `line_no`, `word_no` |
| `jobs` | `doc_sha`, `stage`, `state` (`pending` \| `done` \| `failed`), `page_no`, `error` |

Indexes on `boxes.page_id` and `pages.(doc_sha, page_no)`, which are the access patterns
grounding will use.

Derived granularities (line, block, span) are query-time helpers in `derive.py`, not rows.

---

## 6. Failure modes

Ingest consumes untrusted PDFs, so rejection and recovery are most of the real work. Every
rejection records a reason on the `jobs` row rather than discarding the attempt.

| Case | Behavior |
|---|---|
| No text layer (scanned) | Reject the document. A page "has text" if `get_text("words")` is non-empty; the document passes if at least 60% of pages do. Born-digital-only is a hard scope boundary, so this fails loudly rather than ingesting pages with zero candidate boxes. |
| Encrypted or password-protected | Reject with a distinct reason code. |
| Corrupt or unopenable | Reject; `jobs.state='failed'` with the exception text. |
| Already ingested (same sha256) | No-op. Idempotency is by content hash, not filename. |
| Failure mid-document | Per-page job rows; re-running processes only pages not marked done. |
| Degenerate box (`x1 <= x0` or `y1 <= y0`) | Dropped at extraction. |
| Coordinates outside the page rect | Clamped to [0, 1]. |
| Empty or whitespace-only word | Dropped. |

### 6.1 Rotated pages

If a page carries `/Rotate 90`, the coordinate space PyMuPDF reports text in and the space
the pixmap renders in can disagree, which would silently offset every box by a rotation and
poison all downstream grounding.

Both the pixmap and the normalization divisor are derived from the **same** `page.rect`, and
a test asserts their agreement on a rotated fixture.

### 6.2 Ligatures and irregular spacing (deferred to S4)

The text layer contains ligatures (`fi` as a single glyph) and hyphenated line breaks. This
will make exact answer-string matching fail on real pages.

This slice stores text **raw and unmodified**. Normalization is a grounding-time concern, and
mangling text at ingest would break its correspondence to the stored boxes. Recorded here so
it is a known S4 task rather than a surprise.

---

## 7. Testing

Fixtures are **synthesized at test time with PyMuPDF**: a small PDF with known strings placed
at known coordinates. This keeps binary blobs out of git and allows exact expected values
instead of tolerances. One real page from `proposal_report/proposal.pdf` serves as a smoke
test against genuine output.

| Test | Asserts |
|---|---|
| `test_core_is_light` | A subprocess that imports `visual_verify` has no `sqlalchemy` in `sys.modules` afterwards, even though the test environment has SQLAlchemy installed. |
| `test_dpi_invariance` | The same page rendered at 72 and 150 DPI yields identical normalized boxes. This is the invariant every downstream consumer depends on, and it catches the rotation bug. |
| `test_rotated_page` | A `/Rotate 90` fixture produces boxes that align with the rendered pixmap. |
| `test_boxes_known_coords` | A synthetic PDF with text at a known point yields the expected normalized bbox and `block_no`/`line_no`/`word_no`. |
| `test_gate` | An image-only PDF is rejected; a born-digital PDF is accepted; an encrypted PDF is rejected with the right reason. |
| `test_idempotent` | Ingesting the same file twice leaves row counts unchanged. |
| `test_resume` | After simulated failure at page N, re-running processes only pages N+1 onward. |
| `test_derive` | Line, block, and span union helpers over word rows return correct unions. |
| `test_degenerate_boxes` | Zero-area and out-of-bounds boxes are dropped or clamped. |

Tooling is **ruff** (lint and format) and **pytest**. mypy is skipped: the SQLAlchemy 2.0
models and Pydantic contracts are already typed where it matters, and running mypy over
PyMuPDF is mostly stub-wrangling for no defect-catching return.

---

## 8. Definition of done

1. `uv sync` (core dependencies only, no extras) resolves and imports cleanly.
   `uv sync --all-extras --group dev` gives the full test environment, and
   `test_core_is_light` passes there too.
2. `alembic upgrade head` builds the schema from an empty database.
3. `uv run vvrag ingest proposal_report/proposal.pdf` writes page PNGs and document, page,
   and box rows.
4. `uv run vvrag ingest --dir references/` processes multiple documents, rejecting any
   without a text layer with a clear reason.
5. `uv run vvrag status` reports per-document ingest state.
6. `uv run vvrag inspect <doc> --page 3 --overlay out.png` produces an image in which the
   drawn boxes visibly sit on the text. This human check is required; no numeric test
   replaces it.
7. `pytest` passes; `ruff check` and `ruff format --check` are clean.

---

## 9. What this unblocks

S3 gets a stable page-image corpus with known geometry to embed. S4 gets the candidate box
set that snap-to-box ranks and the word rows the text-span path matches against. S7 gets the
word-level granularity its auto-derived gold boxes require.

The contracts frozen in S1 are the seam all three build against.
