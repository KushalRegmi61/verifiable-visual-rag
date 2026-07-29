import fitz
from PIL import Image

from visual_verify.ingest.render import RenderedPage, render_page


def test_render_writes_png_at_expected_size(born_digital_pdf, tmp_path):
    doc = fitz.open(born_digital_pdf)
    out = render_page(doc[0], tmp_path / "p0001.png", dpi=150)
    doc.close()

    assert isinstance(out, RenderedPage)
    assert out.path.exists()
    # 612 x 792 points at 150 dpi = 1275 x 1650 px
    assert (out.width_px, out.height_px) == (1275, 1650)
    assert Image.open(out.path).size == (1275, 1650)


def test_render_scales_with_dpi(born_digital_pdf, tmp_path):
    doc = fitz.open(born_digital_pdf)
    low = render_page(doc[0], tmp_path / "low.png", dpi=72)
    high = render_page(doc[0], tmp_path / "high.png", dpi=144)
    doc.close()

    assert (low.width_px, low.height_px) == (612, 792)
    assert high.width_px == low.width_px * 2
    assert high.height_px == low.height_px * 2


def test_render_uses_rotated_dimensions(rotated_pdf, tmp_path):
    """A 90-rotated portrait page must render landscape."""
    doc = fitz.open(rotated_pdf)
    out = render_page(doc[0], tmp_path / "rot.png", dpi=150)
    doc.close()

    assert (out.width_px, out.height_px) == (1650, 1275)


def test_render_records_the_rect_it_normalized_against(born_digital_pdf, tmp_path):
    doc = fitz.open(born_digital_pdf)
    out = render_page(doc[0], tmp_path / "p.png", dpi=150)
    doc.close()

    assert out.rect_width == 612.0
    assert out.rect_height == 792.0


def test_render_creates_parent_directories(born_digital_pdf, tmp_path):
    doc = fitz.open(born_digital_pdf)
    out = render_page(doc[0], tmp_path / "deep" / "nested" / "p.png", dpi=72)
    doc.close()

    assert out.path.exists()
