"""Content-addressed response cache: offline demo, and the reproducibility record."""

from visual_verify.agent.cache import CachedChat
from visual_verify.agent.schemas import ClaimList
from visual_verify.agent.types import FakeChat


def test_a_repeated_call_hits_the_cache_instead_of_the_model(tmp_path):
    inner = FakeChat("m1", [ClaimList(claims=["a"])])
    chat = CachedChat(inner, tmp_path)

    first = chat.structured("p", None, ClaimList)
    second = chat.structured("p", None, ClaimList)

    assert first.claims == second.claims == ["a"]
    assert len(inner.calls) == 1, "the second call should not have reached the model"


def test_a_different_prompt_misses(tmp_path):
    inner = FakeChat("m1", [ClaimList(claims=["a"]), ClaimList(claims=["b"])])
    chat = CachedChat(inner, tmp_path)

    assert chat.structured("p1", None, ClaimList).claims == ["a"]
    assert chat.structured("p2", None, ClaimList).claims == ["b"]


def test_a_different_model_id_misses(tmp_path):
    """Otherwise switching provider silently returns the other model's answer,
    which would make an A/B comparison compare a model against itself."""
    a = FakeChat("openai:gpt-4o", [ClaimList(claims=["from-a"])])
    b = FakeChat("google:gemini", [ClaimList(claims=["from-b"])])

    assert CachedChat(a, tmp_path).structured("p", None, ClaimList).claims == ["from-a"]
    assert CachedChat(b, tmp_path).structured("p", None, ClaimList).claims == ["from-b"]


def test_the_cache_survives_a_new_process(tmp_path):
    """The point of writing to disk: the defense demo runs from a cache built
    on a different day, with no network."""
    inner = FakeChat("m1", [ClaimList(claims=["a"])])
    CachedChat(inner, tmp_path).structured("p", None, ClaimList)

    cold = FakeChat("m1", [])  # empty script: any model call now fails loudly
    assert CachedChat(cold, tmp_path).structured("p", None, ClaimList).claims == ["a"]


def test_a_different_image_misses(tmp_path):
    """Same question, different page, must not reuse the answer."""
    one = tmp_path / "one.png"
    two = tmp_path / "two.png"
    one.write_bytes(b"page-one-bytes")
    two.write_bytes(b"page-two-bytes")

    inner = FakeChat("m1", [ClaimList(claims=["a"]), ClaimList(claims=["b"])])
    chat = CachedChat(inner, tmp_path / "cache")

    assert chat.structured("p", one, ClaimList).claims == ["a"]
    assert chat.structured("p", two, ClaimList).claims == ["b"]
