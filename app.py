"""
Outlet Photo Editor - Flask backend
Ověřené modely s fixními version hashy (May 2026)
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import base64
import os
import io
import requests as req_lib
from PIL import Image

app = Flask(__name__)

DEFAULT_BG_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backgrounds")
app.config["BACKGROUNDS_FOLDER"] = os.environ.get("BACKGROUNDS_FOLDER", DEFAULT_BG_FOLDER)

# ── Ověřené modely (version hashe zkontrolovány May 2026) ────────────────────
MODEL_REMBG   = "851-labs/background-remover:a029dff38972b5fda4ec5d75d7d1cd25aeff621d2cf4946a41055d7db66b80bc"
MODEL_FLUX    = "black-forest-labs/flux-fill-pro:9609ba7331ed872c99f81c92c69a9ee52a50d8aba99f636173e0674d997efd0c"
MODEL_SD2     = "stability-ai/stable-diffusion-2-inpainting:e490d072a34a94a11e9711ed5a6ba621c3fab884eda1665d9d3a282d65a21180"

REPLICATE_API = "https://api.replicate.com/v1"


# ── Pomocné funkce ────────────────────────────────────────────────────────────

def get_token():
    return os.environ.get("REPLICATE_API_TOKEN", "")

def image_to_dataurl(img_bytes, mime=None):
    if mime is None:
        mime = "image/png" if img_bytes[:4] == b"\x89PNG" else "image/jpeg"
    b64 = base64.b64encode(img_bytes).decode()
    return f"data:{mime};base64,{b64}"

def decode_dataurl(data_url):
    header, b64 = data_url.split(",", 1)
    return base64.b64decode(b64)

def resize_img(img_bytes, max_px=1024):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def replicate_run(token, model_version, input_data):
    """
    Spustí Replicate model přes HTTP API a počká na výsledek.
    model_version: "owner/model:hash" nebo "owner/model" pro latest
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "wait"   # synchronní čekání až 60s
    }

    # Rozdělení na owner/model a verzi
    if ":" in model_version:
        model_part, version_hash = model_version.rsplit(":", 1)
        url = f"{REPLICATE_API}/predictions"
        payload = {"version": version_hash, "input": input_data}
    else:
        owner, name = model_version.split("/", 1)
        url = f"{REPLICATE_API}/models/{owner}/{name}/predictions"
        payload = {"input": input_data}

    resp = req_lib.post(url, json=payload, headers=headers, timeout=120)

    if not resp.ok:
        raise RuntimeError(f"Replicate {resp.status_code}: {resp.text}")

    result = resp.json()

    # Pokud ještě není hotovo, polluj
    while result.get("status") not in ("succeeded", "failed", "canceled"):
        poll_url = result.get("urls", {}).get("get") or f"{REPLICATE_API}/predictions/{result['id']}"
        r2 = req_lib.get(poll_url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
        result = r2.json()

    if result.get("status") != "succeeded":
        raise RuntimeError(f"Model selhal: {result.get('error', 'neznámá chyba')}")

    output = result.get("output")
    if isinstance(output, list):
        output = output[0]
    if not output:
        raise RuntimeError("Model nevrátil žádný výstup")

    # Stáhni výsledek jako bytes
    r3 = req_lib.get(output, timeout=60)
    r3.raise_for_status()
    return r3.content


# ── Debug endpoint ────────────────────────────────────────────────────────────

@app.route("/api/test")
def test():
    token = get_token()
    return jsonify({
        "status": "ok",
        "version": "2026-05-26-v3",
        "has_token": bool(token),
        "token_preview": ("***" + token[-4:]) if len(token) > 6 else "není nastaven",
        "model_rembg": MODEL_REMBG,
        "model_flux": MODEL_FLUX,
    })


# ── Nastavení ─────────────────────────────────────────────────────────────────

@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        data = request.json or {}
        if data.get("api_token", "").strip():
            os.environ["REPLICATE_API_TOKEN"] = data["api_token"].strip()
        return jsonify({"ok": True})
    token = get_token()
    return jsonify({
        "has_token": bool(token),
        "token_preview": ("***" + token[-4:]) if len(token) > 6 else "",
    })


# ── Pozadí ────────────────────────────────────────────────────────────────────

@app.route("/api/backgrounds")
def list_backgrounds():
    folder = app.config["BACKGROUNDS_FOLDER"]
    os.makedirs(folder, exist_ok=True)
    items = [
        {"name": f, "url": f"/api/bg/{f}"}
        for f in sorted(os.listdir(folder))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    return jsonify({"items": items, "folder": folder})

@app.route("/api/bg/<path:filename>")
def get_bg(filename):
    return send_from_directory(app.config["BACKGROUNDS_FOLDER"], filename)


# ── Odebrání pozadí ───────────────────────────────────────────────────────────

@app.route("/api/remove-bg", methods=["POST"])
def remove_bg():
    token = get_token()
    if not token:
        return jsonify({"error": "API token není nastaven. Otevři Nastavení ⚙️"}), 400

    f = request.files.get("image")
    if not f:
        return jsonify({"error": "Žádný obrázek"}), 400

    img_bytes = f.read()
    data_url = image_to_dataurl(img_bytes)

    try:
        result = replicate_run(token, MODEL_REMBG, {
            "image": data_url,
            "background_type": "rgba",
            "format": "png"
        })
        return jsonify({"image": image_to_dataurl(result, "image/png")})
    except Exception as e:
        print(f"[remove-bg] Chyba: {e}")
        return jsonify({"error": str(e)}), 500


# ── Inpainting ────────────────────────────────────────────────────────────────

@app.route("/api/inpaint", methods=["POST"])
def inpaint():
    token = get_token()
    if not token:
        return jsonify({"error": "API token není nastaven"}), 400

    body = request.json or {}
    image_data = body.get("image")
    mask_data  = body.get("mask")
    prompt     = body.get("prompt", "product clothing on clean studio background, seamless texture")
    model      = body.get("model", "flux")

    if not image_data or not mask_data:
        return jsonify({"error": "Chybí image nebo mask"}), 400

    img_bytes  = decode_dataurl(image_data)
    mask_bytes = decode_dataurl(mask_data)
    img_small  = resize_img(img_bytes, 1024)
    mask_small = resize_img(mask_bytes, 1024)
    img_url    = image_to_dataurl(img_small, "image/png")
    mask_url   = image_to_dataurl(mask_small, "image/png")

    try:
        if model == "flux":
            result = replicate_run(token, MODEL_FLUX, {
                "image":    img_url,
                "mask":     mask_url,
                "prompt":   prompt,
                "steps":    25,
                "guidance": 30,
                "output_format": "png",
                "safety_tolerance": 6,
            })
        else:
            result = replicate_run(token, MODEL_SD2, {
                "image":            img_url,
                "mask":             mask_url,
                "prompt":           prompt,
                "negative_prompt":  "hand, finger, pointer, stick, ruler, arm, pen, blur, artifacts",
                "num_inference_steps": 30,
                "guidance_scale":   7.5,
                "num_outputs":      1,
            })

        return jsonify({"image": image_to_dataurl(result, "image/png")})

    except Exception as e:
        print(f"[inpaint] Chyba: {e}")
        return jsonify({"error": str(e)}), 500


# ── Hlavní stránka ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    os.makedirs(app.config["BACKGROUNDS_FOLDER"], exist_ok=True)
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    print(f"\n Outlet Photo Editor bezi na http://localhost:{port}")
    print(f" Modely: rembg={MODEL_REMBG[:40]}...")
    app.run(host="0.0.0.0", port=port, debug=debug)
