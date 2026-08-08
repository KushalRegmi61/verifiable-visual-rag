"""The only module in this package allowed to import transformers and torch.

Everything here is lazy by construction: importing this module imports
torch. The core and backends never import it; the local backends do, at
call time.

The model class is chosen per model id: the wrong architecture class
silently builds a mismatched state dict and fails with a "size mismatch"
traceback that looks like cache corruption. Qwen2-VL and Qwen2.5-VL are
different architectures and get different classes.
"""

from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

_MODEL_CLASSES = {
    "Qwen/Qwen2-VL-2B-Instruct": Qwen2VLForConditionalGeneration,
    "Qwen/Qwen2.5-VL-3B-Instruct": Qwen2_5_VLForConditionalGeneration,
    "Qwen/Qwen2.5-VL-7B-Instruct": Qwen2_5_VLForConditionalGeneration,
}


def generate_json(
    model_id: str, device: str, prompt: str, image: Image.Image | None
) -> str:
    """One greedy generation, returned as raw text for the caller to parse."""
    import torch

    model_cls = _MODEL_CLASSES.get(model_id, AutoModelForImageTextToText)
    model = model_cls.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_id)

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if image is not None:
        messages[0]["content"].insert(0, {"type": "image", "image": image})

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image] if image is not None else None,
        return_tensors="pt",
    ).to(model.device)

    out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    prompt_len = inputs["input_ids"].shape[1]
    answer = processor.batch_decode(
        out[:, prompt_len:], skip_special_tokens=True
    )[0]

    del model, inputs
    torch.cuda.empty_cache()
    return answer.strip()
