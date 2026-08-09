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

    _print_ask_result(result, 6.0)
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

    _print_ask_result(result, 6.0)
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

    _print_ask_result(result, 6.0)
    out = capsys.readouterr().out

    assert "Answer (0 claim(s) shown):" in out
    assert "Withheld (1 claim(s), not part of the answer):" in out
    assert "abstained: no claim on this page met the support threshold" in out
    assert out.index("Withheld") < out.index("abstained: no claim")


def test_ask_result_prints_the_threshold_before_the_answer_heading(capsys):
    """A transcript at --threshold 0 would otherwise be structurally identical
    to a fully verified run: unsupported claims land under the same "Answer"
    heading with nothing recording how permissive the gate was.
    """
    from visual_verify.cli import _print_ask_result
    from visual_verify.contracts import Answer, Claim

    result = Answer(
        question="q",
        claims=[Claim(text="Fine.", confidence=0.9, abstained=False, label="supported")],
        abstained_overall=False,
    )

    _print_ask_result(result, 4.5)
    out = capsys.readouterr().out

    assert "threshold: 4.5" in out
    assert out.index("threshold: 4.5") < out.index("Answer (")


def test_ask_rejects_a_non_finite_threshold(env, capsys):
    """--threshold nan makes `score < nan` always False, so every claim would
    be shown and abstained_overall would be False: the exact transcript this
    project exists to prevent from looking verified.
    """
    assert main(["ask", "q", "--doc", "abc", "--page", "0", "--threshold", "nan"]) == 1
    assert "finite" in capsys.readouterr().out.lower()

    assert main(["ask", "q", "--doc", "abc", "--page", "0", "--threshold", "inf"]) == 1
    assert "finite" in capsys.readouterr().out.lower()


def test_ask_command_runs_end_to_end_with_cached_fake_models(
    env, born_digital_pdf, capsys, monkeypatch
):
    """Nothing previously called main(["ask", ...]); the only prior coverage
    was build_parser and _print_ask_result in isolation, which is exactly
    where the CachedChat(make_chat(...)) -> answer() wiring bug shipped
    unnoticed. This drives the real cmd_ask body: it fetches the page's
    vectors the way cmd_ground does, wraps two FakeChat models in CachedChat
    over one shared cache directory, and answers for real.

    It also pins that the reader's and verifier's caches, which share that one
    directory, land in two distinct files rather than one overwriting the
    other. That currently depends entirely on model_id staying part of the
    cache key and nothing was exercising it end to end.
    """
    import visual_verify.agent.models as models_module
    from visual_verify.agent.schemas import ClaimList, Verdict
    from visual_verify.agent.types import FakeChat

    assert main(["ingest", str(born_digital_pdf)]) == 0
    capsys.readouterr()
    assert main(["embed", "--all"]) == 0
    capsys.readouterr()

    def fake_make_chat(role, settings):
        if role == "reader":
            return FakeChat("openai:fake-reader", [ClaimList(claims=["Revenue grew 42 percent"])])
        return FakeChat(
            "google:fake-verifier",
            [Verdict(label="supported", confidence=0.9, reason="matches the page")],
        )

    monkeypatch.setattr(models_module, "make_chat", fake_make_chat)

    assert main(["ask", "What happened to revenue?", "--doc", "born_digital", "--page", "0"]) == 0
    out = capsys.readouterr().out

    assert "Revenue grew 42 percent" in out
    assert "supported" in out.lower()
    assert "Answer (1 claim(s) shown):" in out

    settings = Settings.from_env()
    cached = list(settings.agent_cache_dir.glob("*.json"))
    assert len(cached) == 2, "reader and verifier share one cache dir but must not collide"


def test_ask_warns_before_the_reader_runs_when_the_page_is_not_embedded(
    env, born_digital_pdf, tmp_path, capsys, monkeypatch
):
    """prepare_page returns page_vectors=None for an ingested-but-unembedded
    page. ground() then has no visual fallback, so every claim the reader
    paraphrases comes back insufficient_evidence after a full reader call and
    one verifier call, and nothing says the real cause was a missing `vvrag
    embed`. cmd_ask's index.count() == 0 check does not fire, because some
    OTHER document is embedded, which is exactly what this test sets up.
    """
    import fitz

    import visual_verify.agent.models as models_module
    from visual_verify.agent.schemas import ClaimList, Verdict
    from visual_verify.agent.types import FakeChat

    assert main(["ingest", str(born_digital_pdf)]) == 0
    assert main(["embed", "--all"]) == 0

    unembedded = tmp_path / "unembedded.pdf"
    doc = fitz.open()
    doc.new_page(width=612.0, height=792.0).insert_text(
        (72.0, 100.0), "Revenue grew 42 percent", fontsize=12
    )
    doc.save(unembedded)
    doc.close()
    assert main(["ingest", str(unembedded)]) == 0
    capsys.readouterr()

    def fake_make_chat(role, settings):
        if role == "reader":
            return FakeChat("openai:fake-reader", [ClaimList(claims=["Revenue grew 42 percent"])])
        return FakeChat(
            "google:fake-verifier",
            [Verdict(label="supported", confidence=0.9, reason="matches the page")],
        )

    monkeypatch.setattr(models_module, "make_chat", fake_make_chat)

    assert main(["ask", "What happened to revenue?", "--doc", "unembedded", "--page", "0"]) == 0
    out = capsys.readouterr().out

    assert "not embedded" in out
    assert "unembedded.pdf" in out
    assert "insufficient_evidence" in out
    assert "vvrag embed" in out
    # Before the reader runs, so a user can Ctrl-C before paying for any call.
    assert out.index("not embedded") < out.index("Answer (")
