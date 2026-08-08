import pytest
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from visual_verify.cli import main
from visual_verify.config import Settings
from visual_verify.store.engine import make_engine
from visual_verify.store.models import Document, Page


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("VVRAG_DB_URL", f"sqlite:///{tmp_path / 'i.db'}")
    monkeypatch.setenv("VVRAG_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VVRAG_QDRANT_URL", ":memory:")
    monkeypatch.delenv("VVRAG_QDRANT_API_KEY", raising=False)
    # Keep the CLI off the GPU: the fake embedder makes these fast tests.
    monkeypatch.setenv("VVRAG_FAKE_EMBEDDER", "1")
    return tmp_path


def test_embed_then_search(env, born_digital_pdf, capsys):
    assert main(["ingest", str(born_digital_pdf)]) == 0
    capsys.readouterr()

    assert main(["embed", "--all"]) == 0
    assert "embedded" in capsys.readouterr().out.lower()

    assert main(["search", "anything"]) == 0
    assert "page" in capsys.readouterr().out.lower()


def test_embed_is_idempotent(env, born_digital_pdf, capsys):
    main(["ingest", str(born_digital_pdf)])
    main(["embed", "--all"])
    capsys.readouterr()
    assert main(["embed", "--all"]) == 0
    assert "skipped" in capsys.readouterr().out.lower()


def test_search_before_embed_reports_empty(env, born_digital_pdf, capsys):
    main(["ingest", str(born_digital_pdf)])
    capsys.readouterr()
    assert main(["search", "anything"]) == 1
    assert "no pages indexed" in capsys.readouterr().out.lower()


def test_embed_requires_a_target(env, capsys):
    assert main(["embed"]) == 1
    assert "give a document" in capsys.readouterr().out.lower()


def test_ground_command_is_registered():
    from visual_verify.cli import build_parser

    args = build_parser().parse_args(["ground", "some claim", "--doc", "abc", "--page", "3"])

    assert args.claim == "some claim"
    assert args.doc == "abc"
    assert args.page == 3
    assert args.force_visual is False


def test_ground_command_runs_the_text_path_end_to_end(env, born_digital_pdf, capsys):
    """Nothing previously called main(["ground", ...]), so cmd_ground's body
    (overlay drawing, modality/resolution markers) never actually ran. This
    exercises the text path, which needs no vectors and stays off the GPU.
    """
    assert main(["ingest", str(born_digital_pdf)]) == 0
    capsys.readouterr()

    assert main(["ground", "grew 42", "--doc", "born_digital", "--page", "0"]) == 0
    out = capsys.readouterr().out

    assert "grew 42" in out
    assert "text" in out.lower()


def test_ground_command_writes_an_overlay_that_differs_from_the_page(
    env, born_digital_pdf, tmp_path, capsys
):
    assert main(["ingest", str(born_digital_pdf)]) == 0
    capsys.readouterr()

    overlay_path = tmp_path / "overlay.png"
    assert (
        main(
            [
                "ground",
                "grew 42",
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
    out = capsys.readouterr().out
    assert f"wrote {overlay_path}" in out
    assert overlay_path.exists()

    settings = Settings.from_env()
    with Session(make_engine(settings.db_url)) as session:
        doc = session.scalars(select(Document)).one()
        page = session.scalar(select(Page).where(Page.doc_sha == doc.sha256, Page.page_no == 0))
        source_image_path = settings.pages_dir / page.image_path

    source_bytes = Image.open(source_image_path).convert("RGB").tobytes()
    overlay_bytes = Image.open(overlay_path).convert("RGB").tobytes()

    assert overlay_bytes != source_bytes


def test_ground_before_embed_reports_a_helpful_message(env, born_digital_pdf, capsys):
    """index.get_payload indexes recs[0] and raises a raw IndexError when the
    point does not exist. "ingested but forgot to embed" is the most likely
    first-run mistake, so this must fail the same clean way cmd_search does
    (a count() == 0 check), not with a traceback.
    """
    assert main(["ingest", str(born_digital_pdf)]) == 0
    capsys.readouterr()

    # No claim in the text layer, so cmd_ground must reach for vectors.
    assert main(["ground", "not on this page at all", "--doc", "born_digital", "--page", "0"]) == 1
    out = capsys.readouterr().out.lower()
    assert "vvrag embed" in out


def test_ground_command_accepts_force_visual_and_overlay():
    from visual_verify.cli import build_parser

    args = build_parser().parse_args(
        ["ground", "c", "--doc", "abc", "--page", "1", "--force-visual", "--overlay", "o.png"]
    )

    assert args.force_visual is True
    assert args.overlay == "o.png"


def test_ask_command_is_registered():
    from visual_verify.cli import build_parser

    args = build_parser().parse_args(["ask", "what is X?", "--doc", "abc", "--page", "2"])

    assert args.question == "what is X?"
    assert args.doc == "abc"
    assert args.page == 2
    assert args.threshold == 6.0


def test_ask_command_accepts_a_threshold():
    from visual_verify.cli import build_parser

    args = build_parser().parse_args(
        ["ask", "q", "--doc", "abc", "--page", "1", "--threshold", "4.0"]
    )

    assert args.threshold == 4.0


def test_ask_result_puts_withheld_claims_in_their_own_section_after_shown(capsys):
    """Pins the structural split the coordinator asked for: a withheld claim's
    text must never appear before the "Withheld" heading, because that heading
    is what tells a reader the claim was not part of the answer rather than
    simply not produced.
    """
    from visual_verify.cli import _print_ask_result
    from visual_verify.contracts import Answer, Claim, GroundedRegion

    result = Answer(
        question="q",
        claims=[
            Claim(
                text="Revenue grew 42 percent in Q3.",
                confidence=0.91,
                abstained=False,
                label="supported",
                regions=[
                    GroundedRegion(
                        page=0, bbox=(0.181, 0.141, 0.408, 0.155), score=1.0, modality="text"
                    )
                ],
            ),
            Claim(
                text="The company acquired a competitor.",
                confidence=0.79,
                abstained=True,
                label="unsupported",
                regions=[
                    GroundedRegion(
                        page=0, bbox=(0.258, 0.169, 0.848, 0.183), score=1.0, modality="visual"
                    )
                ],
            ),
        ],
        abstained_overall=False,
    )

    _print_ask_result(result)
    out = capsys.readouterr().out

    withheld_idx = out.index("Withheld")
    shown_text_idx = out.index("Revenue grew 42 percent in Q3.")
    withheld_text_idx = out.index("The company acquired a competitor.")

    assert shown_text_idx < withheld_idx
    assert withheld_idx < withheld_text_idx
    assert "Answer (1 claim(s) shown):" in out
    assert "Withheld (1 claim(s), not part of the answer):" in out


def test_ask_result_omits_withheld_section_when_nothing_was_withheld(capsys):
    from visual_verify.cli import _print_ask_result
    from visual_verify.contracts import Answer, Claim

    result = Answer(
        question="q",
        claims=[Claim(text="Fine.", confidence=0.9, abstained=False, label="supported")],
        abstained_overall=False,
    )

    _print_ask_result(result)
    out = capsys.readouterr().out

    assert "Withheld" not in out


def test_ask_result_shows_withheld_section_and_abstention_line_when_all_withheld(capsys):
    from visual_verify.cli import _print_ask_result
    from visual_verify.contracts import Answer, Claim

    result = Answer(
        question="q",
        claims=[Claim(text="Not supported.", confidence=0.2, abstained=True, label="unsupported")],
        abstained_overall=True,
    )

    _print_ask_result(result)
    out = capsys.readouterr().out

    assert "Answer (0 claim(s) shown):" in out
    assert "Withheld (1 claim(s), not part of the answer):" in out
    assert "abstained: no claim on this page met the support threshold" in out
    assert out.index("Withheld") < out.index("abstained: no claim")
