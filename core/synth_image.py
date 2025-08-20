# core/synth_image.py
import io, base64
from PIL import Image, ImageDraw

def synth_base64_image(constraints):
    w = int(constraints.get("img_width", 640))
    h = int(constraints.get("img_height", 400))
    caption = constraints.get("img_caption", "Synthetic image (dotted red line)")
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # dotted red line
    for x in range(20, w-20, 6):
        y = int(h/2 + 0.25*(x - w/2) / (w/2) * h/2)
        if 0 <= y < h:
            d.point((x, y), fill=(220, 0, 0))
    d.text((10, 10), caption, fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
