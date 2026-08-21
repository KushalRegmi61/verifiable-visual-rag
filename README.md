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
uv run vvrag ask "<question>" --doc <sha> --page <n>
uv run vvrag ask "<question>" --doc <sha> --page <n> --threshold 4.0   # admit partial support
```

The reader and the verifier must be **different models**. A model grading its own
output is biased toward it, so the separation is what makes the verification mean
anything. A different vendor makes that true by construction; the current default
uses two OpenAI models of different sizes instead, because no Google API key is
provisioned yet. That is a weaker independence guarantee (shared training
pipeline and RLHF), so switch the verifier to a different vendor
(`VVRAG_VERIFIER_PROVIDER=google`, `VVRAG_VERIFIER_MODEL=gemini-2.0-flash`) once
`GOOGLE_API_KEY` is available:

```bash
VVRAG_READER_PROVIDER=openai      VVRAG_READER_MODEL=gpt-5-mini
VVRAG_VERIFIER_PROVIDER=openai    VVRAG_VERIFIER_MODEL=gpt-5-nano
```

`OPENAI_API_KEY` goes in `.env`, which is gitignored. Both models are sent a
page image, so both must be vision-capable.

Any OpenAI-shaped gateway works as a third provider, so the vendor is an
environment variable rather than a code change. OpenRouter, Groq, Together,
DeepSeek, or a local vLLM or Ollama:

```bash
VVRAG_VERIFIER_PROVIDER=openai_compatible
VVRAG_VERIFIER_MODEL=<a vision model that supports tool calling>
VVRAG_VERIFIER_BASE_URL=https://openrouter.ai/api/v1
VVRAG_VERIFIER_API_KEY=<key for that gateway>
```

The key is per role, not shared, because the point of the arrangement is for
the reader and the verifier to sit behind different vendors. Two requirements
are not negotiable and neither is checked for you: the model must accept images,
and it must support tool calling or a native JSON-schema mode. Without the
second, `with_structured_output` falls back to prompt-and-parse, which is the
correctly-shaped wrong output this layer exists to prevent. Confirm on a real
call rather than assuming.

The model id, which the response cache keys on and which the reader-verifier
independence check compares, is the endpoint HOST rather than the literal
`openai_compatible`. Two gateways serving a model of the same name are different
weights behind an identical string, and a shared cache key would attribute one
vendor's answer to another. The flip side is a limit worth stating: the same
model served by two gateways reads as two different models to that check, so
pointing both roles at, say, the same Llama through OpenRouter and Groq passes
while giving you no independence at all. Responses are
cached under `data/agent_cache`, which lets a demo run offline and is what makes
a reported number reproducible against a model that may drift.

`embed` is a separate command from `ingest` on purpose. Ingest needs only the
four core dependencies and no GPU; embedding needs a 2.5 GB torch stack and
about 21 seconds per page. Keeping them apart means a machine with no GPU can
still ingest a corpus, and only the machine that actually has one has to pull
in the heavier stack.

Interrupting `vvrag embed` is safe: each page is committed to Qdrant as it
completes, and re-running resumes from the first unembedded page.

## Running the product UI

The service is read-only over a corpus built beforehand, so ingest and embed
first:

```bash
uv sync --all-extras --group dev
uv run vvrag ingest <pdf>
uv run vvrag embed --all
```

Then the service and the frontend, in two terminals:

```bash
uv run uvicorn visual_verify.api.app:create_app --factory --workers 1 --port 8000
cd frontend && npm install && npm run dev
```

`--workers 1` is not a default worth changing. Each worker loads its own
ColQwen2 at about 2.6 GB and the development card has 3.63 GB, so a second
worker OOMs at startup. For the same reason the service answers one question at
a time: a semaphore serialises `/ask` across retrieval and the answer loop,
because both use the one resident embedder.

Startup takes about 20 seconds, and that is the point. The model loads once for
the process lifetime rather than once per request, which is what makes the first
question fast instead of every question slow.

The service refuses to start if `VVRAG_QDRANT_URL` is unset, if an API key is
missing, or if the reader and the verifier resolve to the same model. That last
one is the reason the whole design is shaped as it is, and a misconfiguration
would otherwise be invisible in the output: the service would come up, report
itself healthy, and only produce biased verdicts once somebody asked something.

Verified claims stream in one at a time as each verdict lands. The reader's
output is never streamed, so nothing reaches the screen before it has been
judged. A claim the verifier rejects is listed with its label and the verifier's
reason and **no region**: the geometry is stripped before it leaves the process,
so the browser cannot draw it even by mistake.

## Rebuilding the deck

```bash
cd presentation
python3 -m venv .pptvenv && .pptvenv/bin/pip install python-pptx Pillow
.pptvenv/bin/python build_deck.py        # writes Minor_Verifiable_Visual_RAG.pptx
```

`build_deck.py` is path-relative: it reads `assets/` beside it and `emblem.png` from the repo root.
