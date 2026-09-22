"""
Handwritten Digit Recognition - Flask Deployment App
======================================================
Serves a web UI with two input modes:
  1. A drawable canvas (draw a digit with mouse/touch)
  2. A file upload (anyone can upload a photo/scan of a handwritten digit)

Both paths are preprocessed to match the MNIST training format (28x28,
grayscale, normalized) and classified by a trained MLP neural network.
"""
import os
import io
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps, ImageFilter, UnidentifiedImageError
import joblib

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model", "digit_model.pkl")
META_PATH = os.path.join(APP_DIR, "model", "scaler.pkl")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif"}
MAX_UPLOAD_MB = 5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024  # protect a public endpoint from abuse

# Load model once at startup
print("Loading model...")
model = joblib.load(MODEL_PATH)
meta = joblib.load(META_PATH)
print(f"Model loaded. Classes: {meta['classes']}")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_features(img: Image.Image) -> np.ndarray:
    """
    Convert ANY input image (canvas drawing OR an arbitrary uploaded
    photo/scan) into a 784-length normalized feature vector matching
    the Train.csv format (28x28 grayscale, digit=bright on dark
    background, pixel values 0-1).

    Uploaded photos are messier than canvas strokes: uneven lighting,
    paper texture, shadows, JPEG noise. This pipeline is written to be
    robust to that, not just to clean canvas input.
    """
    img = img.convert("L")  # grayscale
    arr = np.array(img, dtype=np.float32)

    # 1. Figure out polarity: is the background light with dark ink
    #    (typical photo of pencil/pen on paper), or already dark with
    #    light strokes (our canvas)? Sample the border pixels, which
    #    are almost always background, not digit.
    border = np.concatenate([
        arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]
    ])
    if np.median(border) > 127:
        # Light background -> invert so the digit becomes bright on dark,
        # matching MNIST's convention.
        img = ImageOps.invert(img)

    # 2. Mild denoising for photo artifacts (JPEG blocking, paper grain),
    #    then stretch contrast so the digit stands out clearly.
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img = ImageOps.autocontrast(img, cutoff=2)
    arr = np.array(img, dtype=np.float32)

    # 3. Bounding-box crop around the digit using a data-driven threshold
    #    (mean + a fraction of the remaining headroom) rather than a
    #    fixed constant, since photo brightness varies a lot.
    threshold = arr.mean() + 0.25 * (arr.max() - arr.mean())
    coords = np.argwhere(arr > max(threshold, 30))
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        # small margin so we don't clip anti-aliased edges
        pad_px = max(2, int(0.03 * max(arr.shape)))
        y0, x0 = max(0, y0 - pad_px), max(0, x0 - pad_px)
        y1, x1 = min(arr.shape[0], y1 + pad_px), min(arr.shape[1], x1 + pad_px)
        cropped = img.crop((x0, y0, x1, y1))
    else:
        cropped = img

    # 4. Pad to a square with generous border (MNIST digits are centered
    #    with ~20% margin), then resize to 28x28.
    w, h = cropped.size
    side = max(w, h, 1)
    pad = int(side * 0.4)
    canvas_size = side + pad * 2
    padded = Image.new("L", (canvas_size, canvas_size), color=0)
    padded.paste(cropped, ((canvas_size - w) // 2, (canvas_size - h) // 2))

    resized = padded.resize((28, 28), Image.LANCZOS)
    pixels = np.array(resized).astype(np.float32) / 255.0
    return pixels.reshape(1, -1)


def preprocess_data_url(image_data_url: str) -> np.ndarray:
    """Canvas path: base64 PNG data URL -> feature vector."""
    header, encoded = image_data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes))
    return image_to_features(img)


def preprocess_upload(file_storage) -> np.ndarray:
    """Upload path: a werkzeug FileStorage -> feature vector."""
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)  # respect phone-camera orientation
    return image_to_features(img)


def build_prediction_response(features: np.ndarray) -> dict:
    probs = model.predict_proba(features)[0]
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [{"digit": int(i), "prob": float(probs[i])} for i in top3_idx]
    return {
        "prediction": pred,
        "confidence": confidence,
        "top3": top3,
        "probabilities": [float(p) for p in probs],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Canvas-drawing path: expects JSON { "image": "<base64 data URL>" }."""
    try:
        data = request.get_json(force=True)
        image_data_url = data.get("image")
        if not image_data_url:
            return jsonify({"error": "No image data provided"}), 400

        features = preprocess_data_url(image_data_url)
        return jsonify(build_prediction_response(features))
    except (UnidentifiedImageError, ValueError):
        return jsonify({"error": "Could not read that image. Try a different one."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict_upload", methods=["POST"])
def predict_upload():
    """
    File-upload path: anyone can POST a photo/scan of a handwritten
    digit as multipart/form-data under the "file" field, e.g.:

        curl -F "file=@digit.jpg" https://your-app.example.com/predict_upload
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            }), 400

        features = preprocess_upload(file)
        return jsonify(build_prediction_response(features))
    except (UnidentifiedImageError, ValueError):
        return jsonify({"error": "Could not read that image. Make sure it's a valid photo of a single digit."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": f"File too large. Max size is {MAX_UPLOAD_MB} MB."}), 413


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
