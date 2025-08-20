# core/assembler.py
from typing import Dict, Any, List
from core.synth_text import synth_string
from core.synth_image import synth_base64_image

def assemble_output(spec: Dict[str, Any], constraints: Dict[str, Any]):
    wants_img = constraints.get("needs_image", False)
    image_slot = constraints.get("image_slot")  # either index for array or key name for object

    if spec["kind"] == "array":
        n = spec["length"]
        out: List[Any] = []
        for i in range(n):
            if wants_img and image_slot == i:
                out.append(synth_base64_image(constraints))
            else:
                out.append(synth_string(i+1))
        # If needs image but slot unspecified, put it in the last item
        if wants_img and image_slot is None and n > 0:
            out[-1] = synth_base64_image(constraints)
        return out

    if spec["kind"] == "object":
        keys = spec["keys"]
        out: Dict[str, Any] = {}
        for k in keys:
            if wants_img and (image_slot == k or (image_slot is None and k.lower() == "image")):
                out[k] = synth_base64_image(constraints)
            else:
                out[k] = synth_string(k)
        return out

    # Fallback
    return ["Synthetic answer 1"]
