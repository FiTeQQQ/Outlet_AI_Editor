"""
Outlet Photo Editor - Flask backend
Napojeno na Replicate.com API pro AI operace
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import replicate
import base64
import os
import io
import requests as req_lib
from PIL import Image

app = Flask(__name__)

# Výchozí složka s pozadími (vedle app.py)
DEFAULT_BG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backgrounds")
app.config["BACKGROUNDS_FOLDER"] = os.environ.get("BACKGROUNDS_FOLDER", DEFAULT_BG_FOLDER)


# ─── Pomocné funkce ──────────────────────────────────────────────────────────

def get_token():
    return os.environ.get("REPLICATE_API_TOKEN", "")

def image_to_dataurl(img_bytes: bytes, mime: str = None) -> str:
    if mime is None:
        mime = "image/png" if img_bytes[:4] == b"\x89PNG" else "image/jpeg"
    b64 = base64.b64encode(img_bytes).decode()
    return f"data:{mime};base64,{b64}"

def fetch_replicate_output(output) -> bytes:
    """Stáhne výsledek z Replicate (FileOutput nebo URL string)."""
    if isinstance(output, list):
        output = output[0]
    url = output.url if hasattr(output, "url") else str(output)
    r = req_lib.get(url, timeout=60)
    r.raise_for_status()
    return r.content

def resize_for_inpaint(img_bytes: bytes, max_size: int = 1024) -> bytes:
    """Zmenší obrázek na max_size pro inpainting, zachová poměr stran."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─── Nastavení ───────────────────────────────────────────────────────────────

@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        data = request.json or {}
        if "api_token" in data and data["api_token"].strip():
            os.environ["REPLICATE_API_TOKEN"] = data["api_token"].strip()
        if "backgrounds_folder" in data and data["backgrounds_folder"].strip():
            folder = data["backgrounds_folder"].strip()
            if os.path.isdir(folder):
                app.config["BACKGROUNDS_FOLDER"] = folder
            else:
                return jsonify({"error": f"Složka neexistuje: {folder}"}), 400
        return jsonify({"ok": True})

    return jsonify({
        "has_token": bool(get_token()),
        "token_preview": ("***" + get_token()[-4:]) if len(get_token()) > 6 else "",
        "backgrounds_folder": app.config["BACKGROUNDS_FOLDER"],
    })


# ─── Pozadí ──────────────────────────────────────────────────────────────────

@app.route("/api/backgrounds")
def list_backgrounds():
    folder = app.config["BACKGROUNDS_FOLDER"]
    os.makedirs(folder, exist_ok=True)
    items = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            items.append({"name": f, "url": f"/api/bg/{f}"})
    return jsonify({"items": items, "folder": folder})

@app.route("/api/bg/<path:filename>")
def get_bg(filename):
    return send_from_directory(app.config["BACKGROUNDS_FOLDER"], filename)


# ─── Odebrání pozadí ─────────────────────────────────────────────────────────

@app.route("/api/remove-bg", methods=["POST"])
def remove_bg():
    token = get_token()
    if not token:
        return jsonify({"error": "API token není nastaven. Otevři Nastavení."}), 400

    f = request.files.get("image")
    if not f:
        return jsonify({"error": "Žádný obrázek"}), 400

    img_bytes = f.read()
    data_url = image_to_dataurl(img_bytes)

    client = replicate.Client(api_token=token)
    try:
        output = client.run(
            "cjwbw/rembg",
            input={"image": data_url}
        )
        result = fetch_replicate_output(output)
        return jsonify({"image": image_to_dataurl(result, "image/png")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Inpainting (odstranění ukazovátka) ──────────────────────────────────────

@app.route("/api/inpaint", methods=["POST"])
def inpaint():
    token = get_token()
    if not token:
        return jsonify({"error": "API token není nastaven"}), 400

    body = request.json or {}
    image_data = body.get("image")   # data URL kompozitu
    mask_data  = body.get("mask")    # data URL masky (bílá = opravit)
    prompt     = body.get("prompt", "product clothing on clean studio background, seamless texture")
    neg_prompt = body.get("neg_prompt", "hand, finger, pointer, stick, ruler, arm, pen, blur, distortion, artifacts")
    model      = body.get("model", "flux")  # "flux" nebo "sd2"

    if not image_data or not mask_data:
        return jsonify({"error": "Chybí image nebo mask"}), 400

    # Dekóduj → zmenši na max 1024 → zakóduj zpět
    def decode_dataurl(data_url):
        header, b64 = data_url.split(",", 1)
        return base64.b64decode(b64)

    img_bytes  = decode_dataurl(image_data)
    mask_bytes = decode_dataurl(mask_data)

    img_small  = resize_for_inpaint(img_bytes, 1024)
    mask_small = resize_for_inpaint(mask_bytes, 1024)

    img_url  = image_to_dataurl(img_small,  "image/png")
    mask_url = image_to_dataurl(mask_small, "image/png")

    client = replicate.Client(api_token=token)

    try:
        if model == "flux":
            output = client.run(
                "black-forest-labs/flux-fill-pro",
                input={
                    "image":    img_url,
                    "mask":     mask_url,
                    "prompt":   prompt,
                    "steps":    25,
                    "guidance": 30,
                }
            )
        else:  # výchozí: stable-diffusion-2-inpainting
            output = client.run(
                "stability-ai/stable-diffusion-2-inpainting",
                input={
                    "image":            img_url,
                    "mask":             mask_url,
                    "prompt":           prompt,
                    "negative_prompt":  neg_prompt,
                    "num_inference_steps": 30,
                    "guidance_scale":   7.5,
                    "num_outputs":      1,
                }
            )

        result = fetch_replicate_output(output)
        return jsonify({"image": image_to_dataurl(result, "image/png")})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Hlavní stránka ───────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    os.makedirs(app.config["BACKGROUNDS_FOLDER"], exist_ok=True)
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    print(f"\n🚀 Outlet Photo Editor běží na http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
