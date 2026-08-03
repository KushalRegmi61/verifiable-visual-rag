import pytest

from visual_verify.retrieval.provenance import EmbedProvenance, ProvenanceMismatch

BASE = EmbedProvenance(
    model_id="vidore/colqwen2-v1.0",
    model_revision="abc123",
    quantization="nf4-skipvis",
    dtype="float16",
    render_dpi=150,
    embed_version=1,
)


def test_round_trips_through_payload():
    assert EmbedProvenance.from_payload(BASE.to_payload()) == BASE


def test_identical_provenance_is_compatible():
    BASE.require_compatible(BASE)  # must not raise


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_id", "vidore/colSmol-500M"),
        ("model_revision", "def456"),
        ("quantization", "none"),
        ("dtype", "bfloat16"),
        ("render_dpi", 300),
        ("embed_version", 2),
    ],
)
def test_any_difference_is_a_mismatch(field, value):
    """Every field is load-bearing: differing vectors are not comparable."""
    other = EmbedProvenance(**{**BASE.to_payload(), field: value})
    with pytest.raises(ProvenanceMismatch, match=field):
        BASE.require_compatible(other)


def test_mismatch_message_names_both_values():
    other = EmbedProvenance(**{**BASE.to_payload(), "render_dpi": 300})
    with pytest.raises(ProvenanceMismatch) as exc:
        BASE.require_compatible(other)
    assert "150" in str(exc.value) and "300" in str(exc.value)


def test_payload_keys_are_flat_scalars():
    """Qdrant payloads must be JSON scalars, not nested objects."""
    for value in BASE.to_payload().values():
        assert isinstance(value, (str, int))


def test_from_payload_tolerates_extra_keys():
    """The real Qdrant payload also holds doc_sha, page_no, image_path,
    n_patches_x, etc. from_payload is called on that whole dict, not on a
    payload containing only provenance fields, so it must ignore the rest
    rather than blowing up on an unexpected keyword argument.
    """
    payload = {
        **BASE.to_payload(),
        "doc_sha": "deadbeef",
        "page_no": 3,
        "image_path": "/tmp/page-3.png",
        "n_patches_x": 32,
    }
    assert EmbedProvenance.from_payload(payload) == BASE
