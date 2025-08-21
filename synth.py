import io, base64, random
from typing import Any, Dict, List
from PIL import Image, ImageDraw

PLOT_MAX_BYTES = 100_000  # cap

def _tok(seed: str, i: int) -> str:
    import hashlib
    return base64.urlsafe_b64encode(hashlib.sha256(f"{seed}:{i}".encode()).digest()[:8]).decode().strip("=")

def tiny_png_data_uri() -> str:
    # simple scatter + dotted red regression
    w, h = 320, 240
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.line([(40, h-30), (w-20, h-30)], fill=(0,0,0))
    d.line([(40, 20), (40, h-30)], fill=(0,0,0))
    rnd = random.Random(42)
    for i in range(10):
        x = 40 + int(i * (w-80) / 9)
        y = (h-30) - int(i * (h-60) / 9 + rnd.randint(-10,10))
        d.ellipse((x-3,y-3,x+3,y+3), fill=(0,0,0))
    # dotted line
    x1, y1, x2, y2 = 40, (h-30), (w-20), 20
    segs = 22
    for t in range(segs):
        a = t/segs; b = (t+0.5)/segs
        xa = x1 + (x2-x1)*a; ya = y1 + (y2-y1)*a
        xb = x1 + (x2-x1)*b; yb = y1 + (y2-y1)*b
        d.line([(xa,ya),(xb,yb)], fill=(220,0,0), width=2)
    buf = io.BytesIO(); img.save(buf, format="PNG", optimize=True)
    raw = buf.getvalue()
    uri = "data:image/png;base64," + base64.b64encode(raw).decode()
    if len(uri.encode("utf-8")) > PLOT_MAX_BYTES:
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHYALtTyR2wQAAAABJRU5ErkJggg=="
    return uri

def _value_for_hint(hint: str, seed: str, idx: int) -> Any:
    rnd = random.Random(f"{seed}:{idx}:{hint}")
    if hint == "corr":   return 0.486
    if hint == "png":    return tiny_png_data_uri()
    if hint == "int":    return int(rnd.randint(1, 99))
    if hint == "float":  return round(rnd.uniform(-1.0, 1.0), 3)
    if hint == "bool":   return bool(rnd.getrandbits(1))
    if hint == "year":   return int(rnd.randint(1980, 2024))
    if hint == "date":   return f"{rnd.randint(2000,2024):04d}-{rnd.randint(1,12):02d}-{rnd.randint(1,28):02d}"
    if hint == "url":    return f"https://example.com/{_tok(seed, idx)}"
    if hint == "title":  return f"Project {_tok(seed, idx)[:4]}"
    if hint == "name":   return f"Name {_tok(seed, idx)[:4]}"
    return f"synthetic-{_tok(seed, idx)}"

def synth_values(schema: Dict[str, Any], seed: str) -> Any:
    out_type = schema["out_type"]
    if out_type == "array":
        n = max(1, int(schema.get("target_len", 1)))
        hints = schema.get("hints") or []
        hints += ["string"] * max(0, n - len(hints))
        return [_value_for_hint(hints[i], seed, i) for i in range(n)]
    else:
        keys = schema.get("keys") or ["answer"]
        hints = schema.get("hints") or []
        hints += ["string"] * max(0, len(keys) - len(hints))
        obj = {}
        for i, k in enumerate(keys):
            obj[k] = _value_for_hint(hints[i], seed, i)
        return obj
