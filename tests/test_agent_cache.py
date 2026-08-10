"""Content-addressed response cache: offline demo, and the reproducibility record."""

from helpers import claim_list
from visual_verify.agent.cache import CachedChat
from visual_verify.agent.schemas import ClaimList
from visual_verify.agent.types import FakeChat


def test_a_repeated_call_hits_the_cache_instead_of_the_model(tmp_path):
    inner = FakeChat("m1", [ClaimList(claims=["a"])])
    chat = CachedChat(inner, tmp_path)

    first = chat.structured("p", [], ClaimList)
    second = chat.structured("p", [], ClaimList)

    assert first.claims == second.claims
    assert first.claims[0].text == "a"
    assert len(inner.calls) == 1, "the second call should not have reached the model"


def test_a_different_prompt_misses(tmp_path):
    inner = FakeChat("m1", [ClaimList(claims=["a"]), ClaimList(claims=["b"])])
    chat = CachedChat(inner, tmp_path)

    assert chat.structured("p1", [], ClaimList).claims[0].text == "a"
    assert chat.structured("p2", [], ClaimList).claims[0].text == "b"


def test_a_different_model_id_misses(tmp_path):
    """Otherwise switching provider silently returns the other model's answer,
    which would make an A/B comparison compare a model against itself."""
    a = FakeChat("openai:gpt-4o", [ClaimList(claims=["from-a"])])
    b = FakeChat("google:gemini", [ClaimList(claims=["from-b"])])

    assert CachedChat(a, tmp_path).structured("p", [], ClaimList).claims[0].text == "from-a"
    assert CachedChat(b, tmp_path).structured("p", [], ClaimList).claims[0].text == "from-b"


def test_the_cache_survives_a_new_process(tmp_path):
    """The point of writing to disk: the defense demo runs from a cache built
    on a different day, with no network."""
    inner = FakeChat("m1", [ClaimList(claims=["a"])])
    CachedChat(inner, tmp_path).structured("p", [], ClaimList)

    cold = FakeChat("m1", [])  # empty script: any model call now fails loudly
    assert CachedChat(cold, tmp_path).structured("p", [], ClaimList).claims[0].text == "a"


def test_a_different_image_misses(tmp_path):
    """Same question, different page, must not reuse the answer."""
    one = tmp_path / "one.png"
    two = tmp_path / "two.png"
    one.write_bytes(b"page-one-bytes")
    two.write_bytes(b"page-two-bytes")

    inner = FakeChat("m1", [ClaimList(claims=["a"]), ClaimList(claims=["b"])])
    chat = CachedChat(inner, tmp_path / "cache")

    assert chat.structured("p", [one], ClaimList).claims[0].text == "a"
    assert chat.structured("p", [two], ClaimList).claims[0].text == "b"


def test_the_key_is_unambiguous_across_part_boundaries(tmp_path):
    """A separator-joined key is not injective: ("X\\0Y", "Z") and ("X", "Y\\0Z")
    would hash the same, so two different calls would share a cache entry and
    one would silently return the other's answer."""
    from visual_verify.agent.cache import _digest

    assert _digest("X\x00ClaimList", "Y", [], "ClaimList") != _digest(
        "X", "ClaimList\x00Y", [], "ClaimList"
    )


def test_two_different_page_sets_do_not_share_a_cache_entry(tmp_path):
    """THE test of this change. The digest used to cover ONE image, so a call
    over pages [7, 8, 9] and a call over pages [7, 8] hashed identically and
    the second silently returned the first's answer. A cached wrong answer is
    the worst failure this module can produce, because it is reproducible."""
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    inner = FakeChat("m", [claim_list("from both"), claim_list("from one")])
    chat = CachedChat(inner, tmp_path / "cache")

    assert chat.structured("p", [a, b], ClaimList).claims[0].text == "from both"
    assert chat.structured("p", [a], ClaimList).claims[0].text == "from one"


def test_a_text_only_call_and_a_one_image_call_do_not_share_a_cache_entry(tmp_path):
    """`image_paths` used to be `Path | None`, and `[]` now stands in for the
    text-only `None` case. That is exactly the boundary the change moved: a
    call with no pages attached and a call with one page, same prompt, must
    not collide just because an empty list could hash away to nothing."""
    a = tmp_path / "a.png"
    a.write_bytes(b"a")
    inner = FakeChat("m", [claim_list("text only"), claim_list("with page")])
    chat = CachedChat(inner, tmp_path / "cache")

    assert chat.structured("p", [], ClaimList).claims[0].text == "text only"
    assert chat.structured("p", [a], ClaimList).claims[0].text == "with page"


def test_image_order_changes_the_cache_key(tmp_path):
    """[a, b] and [b, a] are different prompts to a vision model: the pages are
    numbered in the text and the model reads them in order."""
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    inner = FakeChat("m", [claim_list("ab"), claim_list("ba")])
    chat = CachedChat(inner, tmp_path / "cache")

    assert chat.structured("p", [a, b], ClaimList).claims[0].text == "ab"
    assert chat.structured("p", [b, a], ClaimList).claims[0].text == "ba"
