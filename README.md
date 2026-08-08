# Verifiable Visual RAG

Region-level, verifiable evidence for question answering over research documents. Given a question and a PDF, the system retrieves the right page, snaps the answer onto a real region (box) of that page, has an independent verifier confirm the region supports the answer, and abstains when confidence is low.

Three pillars: **region-level evidence**, **independent verification**, **abstention**.

BE Minor Project, IOE Pulchowk (BCT). Team: Bhim Prasad Upadhaya (PUL080BCT019), Biprash Pandey (PUL080BCT022), Kushal Regmi (PUL080BCT042).

## Project structure

| Folder | Contents |
|---|---|
| `proposal_report/` | The proposal deliverable: `proposal.tex` (LaTeX source), `proposal.pdf` (compiled), `emblem.png`. Base for the final report. |
| `research/` | Design and research docs: `related_works.md`, `system_design.md`, `recommended_stack.md`, `visual_verify_brief.md`, `visual_verify_hld.md`, `visual_verify_research_report.md`, `cite_or_abstain_spec.md`. |
| `presentation/` | Defense deck: `build_deck.py` (python-pptx generator), `Minor_Verifiable_Visual_RAG.pptx`, preview PDF, `section4_5_speaker_guide.md`/`.pdf`, `assets/` (compiled TikZ figures and equations). |
| `notes/` | `defense_qa.md` (concept + Q&A prep) and the raw `session_transcript.jsonl`. |
| `references/` | Supporting and earlier PDFs: `reference_proposal.pdf`, `Minor.pdf`, and the submitted `ENCT354MINORPROJECT_*.pdf`. |
| `emblem.png` | TU emblem at repo root (the deck build reads it from here). |

## Running the ingest pipeline

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras --group dev     # full dev environment
uv run alembic upgrade head          # create the schema

uv run vvrag ingest proposal_report/proposal.pdf
uv run vvrag ingest --dir references/
uv run vvrag status
uv run vvrag inspect proposal_report/proposal.pdf --page 3 --overlay overlay.png
```

`inspect` takes a substring of the document path, a sha256, or a sha256 prefix.
If what you give it matches more than one ingested document it lists the
candidates rather than guessing. (Note that the bare substring `proposal.pdf`
matches both `proposal_report/proposal.pdf` and `references/reference_proposal.pdf`,
which is why the example above passes the directory too.)

Two further flags control what gets drawn:

```bash
# --kind: overlay a coarser granularity, derived from the stored word boxes
uv run vvrag inspect proposal_report/proposal.pdf --page 3 --kind line --overlay lines.png

# --find: draw only the rects covering a phrase, the project's claim, as a picture
uv run vvrag inspect proposal_report/proposal.pdf --page 3 \
    --find "region-level evidence" --overlay found.png
```

Only word boxes are stored; `line`, `block`, and `--find` spans are computed at
query time, so the granularity can be retuned without re-ingesting. A phrase
that wraps across a line break comes back as one rect per line rather than a
single union, so the highlight never sweeps in the words between the two halves.
`--find` prints how many rects matched, or says the phrase is not on that page,
and exits 0 either way.

Render DPI is fixed per document: re-ingesting with a different `--dpi` is
refused rather than silently mixing page sizes inside one document.

Configuration is environment-driven; nothing is hardcoded:

| Variable | Default | Purpose |
|---|---|---|
| `VVRAG_DB_URL` | `sqlite:///data/index.db` | Metadata store; swap for a Postgres URL to deploy |
| `VVRAG_DATA_DIR` | `data` | Page images and the SQLite file |
| `VVRAG_RENDER_DPI` | `150` | Page render DPI, fixed per corpus |
| `VVRAG_MIN_TEXT_PAGE_RATIO` | `0.6` | Born-digital gate threshold |
| `VVRAG_QDRANT_URL` | unset | Qdrant instance for the S3 retrieval index |
| `VVRAG_QDRANT_API_KEY` | unset | Qdrant API key |

Only born-digital PDFs are accepted. Scanned documents are rejected by design:
this project runs no OCR.

### Testing

```bash
uv run pytest                        # 135 tests
uv run ruff check . && uv run ruff format --check .
```

One test is marked `slow`: it builds the wheel, installs it into a throwaway
virtualenv, and runs the `vvrag` console script from there, so that a path that
only resolves inside a source checkout cannot pass unnoticed. It takes about 25
seconds and runs by default; `uv run pytest -m "not slow"` skips it.

## Running the retrieval index

Requires the `retrieval` extra and a CUDA GPU with at least 3 GB free.

```bash
uv sync --all-extras --group dev
export VVRAG_QDRANT_URL=...        # or put both in a gitignored .env
export VVRAG_QDRANT_API_KEY=...

uv run vvrag embed --all           # ~21 s/page, resumable
uv run vvrag search "your question" -k 5
uv run vvrag ground "<claim>" --doc <sha> --page <n> --overlay out.png
uv run vvrag ground "<claim>" --doc <sha> --page <n> --force-visual   # what the eval measures
uv run vvrag ask "your question" --doc <sha> --page <n>               # read, ground, judge, abstain
```

`ask` is the full pipeline: the reader model answers from the page, each atomic
claim is grounded, and a different model judges each claim against its region.
Weak judgements abstain. The reader defaults to a hosted endpoint
(`VVRAG_READER_URL`, `VVRAG_READER_KEY`); the verifier defaults to a local VLM
(`VVRAG_VERIFIER_BACKEND=local`), and both can be flipped to `hosted` or `local`
independently, as long as the pairing stays independent (never the same model
for both roles).

`embed` is a separate command from `ingest` on purpose. Ingest needs only the
four core dependencies and no GPU; embedding needs a 2.5 GB torch stack and
about 21 seconds per page. Keeping them apart means a machine with no GPU can
still ingest a corpus, and only the machine that actually has one has to pull
in the heavier stack.

Interrupting `vvrag embed` is safe: each page is committed to Qdrant as it
completes, and re-running resumes from the first unembedded page.

## Rebuilding the deck

```bash
cd presentation
python3 -m venv .pptvenv && .pptvenv/bin/pip install python-pptx Pillow
.pptvenv/bin/python build_deck.py        # writes Minor_Verifiable_Visual_RAG.pptx
```

`build_deck.py` is path-relative: it reads `assets/` beside it and `emblem.png` from the repo root.
