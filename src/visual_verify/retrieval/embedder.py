"""ColQwen2 page and query embedding.

Every non-obvious line here has a measured failure behind it. See
docs/superpowers/specs/2026-08-03-s3-retrieval-design.md section 4.
"""

import numpy as np
import torch
from PIL import Image

from visual_verify.retrieval.geometry import PatchGrid
from visual_verify.retrieval.provenance import EmbedProvenance
from visual_verify.retrieval.types import PageEmbedding

MODEL_ID = "vidore/colqwen2-v1.0"
EMBED_VERSION = 1

# GTX 1650 is Turing (sm_75) and has NO native bfloat16, which every ColPali
# example in the wild uses. Verified: fp16 embeddings on this card contain no
# NaN or Inf and stay unit-normalized.
DTYPE = torch.float16

# LOAD-BEARING. Blanket load_in_4bit also quantizes the vision tower and the
# projection head, which destroys patch-level geometry: measured known-item
# top-1 of 0.00 against 1.00 with this skip list, on identical code and
# hardware. The broken configuration emits no warning, raises nothing, produces
# no NaN, returns the correct shape, and yields unit-normalized vectors.
SKIP_MODULES = ["visual", "custom_text_proj"]


class AdapterLoadError(RuntimeError):
    """The LoRA adapter or projection head did not load."""


class ColQwen2Embedder:
    """Loads once, embeds many. Not thread-safe."""

    def __init__(self, render_dpi: int = 150, device: str = "cuda:0") -> None:
        from colpali_engine.models import ColQwen2, ColQwen2Processor
        from huggingface_hub import model_info
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=SKIP_MODULES,
        )
        self.model = ColQwen2.from_pretrained(
            MODEL_ID, torch_dtype=DTYPE, device_map=device, quantization_config=quant
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(MODEL_ID)
        self._check_adapter_loaded()

        self._image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        # Confirmed directly on the loaded ColQwen2 wrapper (measured value 2),
        # not on model.config, which does not carry this attribute; it lives on
        # model.config.vision_config instead. Using the top-level accessor
        # matches colpali_engine's own docstring for get_n_patches.
        self._merge_size = self.model.spatial_merge_size

        # Pin the resolved commit, not "main": a branch can move and would let
        # the weights change while provenance claimed they had not.
        revision = model_info(MODEL_ID).sha
        self._provenance = EmbedProvenance(
            model_id=MODEL_ID,
            model_revision=revision,
            quantization="nf4-skipvis",
            dtype="float16",
            render_dpi=render_dpi,
            embed_version=EMBED_VERSION,
        )

    def _check_adapter_loaded(self) -> None:
        """Fail loudly if the trained weights were silently randomized.

        colpali-engine 0.3.17 on transformers 5.x does exactly this: the adapter
        keys are written against the old submodule path (model.layers, since
        renamed to language_model.layers), so every LoRA weight and the
        projection head are newly initialized at random. transformers reports it
        as an informational table, not an error, and the resulting model scores
        at chance while looking entirely healthy.
        """
        names = [n for n, _ in self.model.named_parameters()]
        if not any("lora" in n for n in names):
            raise AdapterLoadError(
                "no LoRA parameters found on the loaded model. The installed "
                "colpali-engine and transformers versions are incompatible; see "
                "the retrieval extra pins in pyproject.toml."
            )
        proj = getattr(self.model, "custom_text_proj", None)
        if proj is None:
            raise AdapterLoadError("custom_text_proj missing; projection head did not load")

    @property
    def provenance(self) -> EmbedProvenance:
        return self._provenance

    def _grid_for(self, image_size: tuple[int, int], seq_len: int, input_ids) -> PatchGrid:
        n_x, n_y = self.processor.get_n_patches(image_size, spatial_merge_size=self._merge_size)
        # Derive the offset from the token stream rather than hardcoding 4. The
        # prompt template is model-version dependent, and an off-by-N offset
        # shifts every patch box without raising anything.
        positions = (input_ids == self._image_token_id).nonzero(as_tuple=True)[0]
        if positions.numel() != n_x * n_y:
            raise ValueError(
                f"processor reports a {n_x}x{n_y} grid ({n_x * n_y} patches) but the "
                f"token stream holds {positions.numel()} image tokens"
            )
        return PatchGrid(n_x=n_x, n_y=n_y, offset=int(positions[0]), n_vectors=seq_len)

    @torch.no_grad()
    def embed_page(self, image_path: str, image_size: tuple[int, int]) -> PageEmbedding:
        # Context manager: a 300-page ingest that never closes its image file
        # handle leaks a file descriptor per page.
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        # Batch of one: measured faster than batch 2 on this card (21.4 s vs
        # 24.6 s per page) because it is memory-bound, and with no padding there
        # is no left-padding trap to fall into.
        batch = self.processor.process_images([image]).to(self.model.device)
        out = self.model(**batch)

        mask = batch["attention_mask"][0].bool()
        vectors = out[0][mask].to(torch.float32).cpu().numpy()
        ids = batch["input_ids"][0][mask]
        grid = self._grid_for(image_size, vectors.shape[0], ids)
        return PageEmbedding(vectors=vectors, grid=grid)

    @torch.no_grad()
    def embed_query(self, text: str) -> np.ndarray:
        batch = self.processor.process_queries([text]).to(self.model.device)
        out = self.model(**batch)
        # Select BY MASK, never out[0, :n]. Qwen2-VL's processor pads on the
        # LEFT, so prefix slicing reads padding. MaxSim sums over query tokens,
        # so a pad vector adds a spurious maximum to every page's score.
        mask = batch["attention_mask"][0].bool()
        return out[0][mask].to(torch.float32).cpu().numpy()
