"""The `vvrag ask` command: the full read-ground-judge-gate pipeline.

The models are fakes via monkeypatched builders; what is asserted is the
adapter: fetching, text-layer reconstruction, wiring, verdict output,
abstention, and the overlay. No GPU, no network.
"""

import pytest
from PIL import Image

from visual_verify.cli import main
from visual_verify.verify.claims import ReaderOutput
from visual_verify.verify.rubric import Judgement


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.delenv("VVRAG_QDRANT_API_KEY", raising=False)
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    return tmp_path


class CliFakeReader:
    def __init__(self, output: ReaderOutput):
        self.output = output

    def read(self, question, image, text_layer):
        return self.output


class CliFakeVerifier:
    def __init__(self, label: str = "supported"):
        self.label = label

    def judge(self, claim, evidence):
        return Judgement(label=self.label)


def wire_fakes(monkeypatch, output=None, label="supported"):
    reader = CliFakeReader(output or ReaderOutput(answer="It grew 42.", claims=["grew 42"]))
    verifier = CliFakeVerifier(label)
    monkeypatch.setattr("visual_verify.cli._build_reader", lambda: reader)
    monkeypatch.setattr("visual_verify.cli._build_verifier", lambda: verifier)
    return reader, verifier


def test_ask_command_is_registered():
    from visual_verify.cli import build_parser

    args = build_parser().parse_args(
        ["ask", "how much did it grow?", "--doc", "abc", "--page", "3"]
    )
    assert args.question == "how much did it grow?"
    assert args.doc == "abc"
    assert args.page == 3
    assert args.threshold == 0.5
    assert args.force_visual is False


def test_ask_runs_text_path_end_to_end(env, born_digital_pdf, capsys, monkeypatch):
    wire_fakes(monkeypatch)
    assert main(["ingest", str(born_digital_pdf)]) == 0
    capsys.readouterr()

    assert main(["ask", "how much did it grow?", "--doc", "born_digital", "--page", "0"]) == 0
    out = capsys.readouterr().out

    assert "grew 42" in out
    assert "judged 1.000" in out
    assert "text" in out.lower()
    assert "abstained_overall: False" in out


def test_ask_abstains_on_a_weak_verdict(env, born_digital_pdf, capsys, monkeypatch):
    wire_fakes(monkeypatch, label="unsupported")
    assert main(["ingest", str(born_digital_pdf)]) == 0
    capsys.readouterr()

    assert main(["ask", "q?", "--doc", "born_digital", "--page", "0"]) == 0
    out = capsys.readouterr().out

    assert "ABSTAINED 0.000" in out
    assert "abstained_overall: True" in out


def test_ask_with_no_checkable_claims_abstains_whole_answer(
    env, born_digital_pdf, capsys, monkeypatch
):
    wire_fakes(monkeypatch, output=ReaderOutput(answer="No idea.", claims=[]))
    assert main(["ingest", str(born_digital_pdf)]) == 0
    capsys.readouterr()

    assert main(["ask", "q?", "--doc", "born_digital", "--page", "0"]) == 0
    out = capsys.readouterr().out

    assert "no checkable claims" in out


def test_ask_without_reader_config_fails_cleanly(env, born_digital_pdf, capsys, monkeypatch):
    monkeypatch.delenv("VVRAG_READER_URL", raising=False)
    monkeypatch.delenv("VVRAG_READER_BACKEND", raising=False)
    assert main(["ingest", str(born_digital_pdf)]) == 0
    capsys.readouterr()

    assert main(["ask", "q?", "--doc", "born_digital", "--page", "0"]) == 1
    out = capsys.readouterr().out.lower()
    assert "vvrag_reader_url" in out


def test_ask_writes_an_overlay_that_differs_from_the_page(
    env, born_digital_pdf, tmp_path, capsys, monkeypatch
):
    wire_fakes(monkeypatch)
    assert main(["ingest", str(born_digital_pdf)]) == 0
    capsys.readouterr()

    overlay_path = tmp_path / "ask-overlay.png"
    assert (
        main(
            [
                "ask",
                "how much did it grow?",
                "--doc",
                "born_digital",
                "--page",
                "0",
                "--overlay",
                str(overlay_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert overlay_path.exists()

    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from visual_verify.config import Settings
    from visual_verify.store.engine import make_engine
    from visual_verify.store.models import Document, Page

    settings = Settings.from_env()
    with Session(make_engine(settings.db_url)) as session:
        doc = session.scalars(select(Document)).one()
        page = session.scalar(select(Page).where(Page.doc_sha == doc.sha256, Page.page_no == 0))
        source_image_path = settings.pages_dir / page.image_path

    source_bytes = Image.open(source_image_path).convert("RGB").tobytes()
    overlay_bytes = Image.open(overlay_path).convert("RGB").tobytes()
    assert overlay_bytes != source_bytes


def test_ask_backend_error_returns_nonzero(env, born_digital_pdf, capsys, monkeypatch):
    def broken_reader():
        raise ValueError("no such backend")

    monkeypatch.setattr("visual_verify.cli._build_reader", broken_reader)
    assert main(["ingest", str(born_digital_pdf)]) == 0
    capsys.readouterr()

    assert main(["ask", "q?", "--doc", "born_digital", "--page", "0"]) == 1
    assert "no such backend" in capsys.readouterr().out
