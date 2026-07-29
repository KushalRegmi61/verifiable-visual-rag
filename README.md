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

Configuration is environment-driven; nothing is hardcoded:

| Variable | Default | Purpose |
|---|---|---|
| `VVRAG_DB_URL` | `sqlite:///data/index.db` | Metadata store; swap for a Postgres URL to deploy |
| `VVRAG_DATA_DIR` | `data` | Page images and the SQLite file |
| `VVRAG_RENDER_DPI` | `150` | Page render DPI, fixed per corpus |
| `VVRAG_MIN_TEXT_PAGE_RATIO` | `0.6` | Born-digital gate threshold |
| `VVRAG_QDRANT_URL` | unset | Vector index, used from S3 onward |

Only born-digital PDFs are accepted. Scanned documents are rejected by design:
this project runs no OCR.

### Testing

```bash
uv run pytest                        # 127 tests
uv run ruff check . && uv run ruff format --check .
```

## Rebuilding the deck

```bash
cd presentation
python3 -m venv .pptvenv && .pptvenv/bin/pip install python-pptx Pillow
.pptvenv/bin/python build_deck.py        # writes Minor_Verifiable_Visual_RAG.pptx
```

`build_deck.py` is path-relative: it reads `assets/` beside it and `emblem.png` from the repo root.
