# ✍️ Handwritten Digit Recognition
App link :https://handwritten-digit-recognition-8kgm.onrender.com

A full deploy-ready web app that recognizes handwritten digits (0–9),
using a neural network (MLP) trained on 42,000 MNIST samples from
`Train.csv`. **Test accuracy: 97.6%**

Two ways to give it a digit:
- **Draw** on an HTML canvas
- **Upload an image** — anyone can upload a photo or scan of a
  handwritten digit and get a prediction back. This works as a public
  endpoint once deployed, so it's not limited to you locally.

```
digit_recognizer/
├── app.py                 # Flask backend (serves UI + /predict API)
├── train_model.py         # Script used to train the neural network
├── requirements.txt       # Python dependencies
├── Procfile                # For Heroku / Render process definition
├── runtime.txt             # Python version pin
├── model/
│   ├── digit_model.pkl     # Trained MLPClassifier (already trained)
│   └── scaler.pkl          # Model metadata
├── templates/
│   └── index.html          # Drawing UI
└── static/
    ├── style.css
    └── script.js
```

## 1. Run locally

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app
python app.py
```

Open **http://localhost:5000**, draw a digit, click **Predict**.

The trained model (`model/digit_model.pkl`) is already included, so you
do **not** need to retrain before running the app.

## 2. Retrain the model (optional)

If you want to retrain on your own copy of `Train.csv` (place it in the
project root):

```bash
python train_model.py
```

This overwrites `model/digit_model.pkl` and `model/scaler.pkl`. Training
takes under a minute on CPU.

## 3. How it works

- **Model**: a `scikit-learn` `MLPClassifier` (feed-forward neural
  network) with two hidden layers (256 → 128 neurons), ReLU activations,
  Adam optimizer, and early stopping. Trained on the 784 pixel columns
  (28×28 grayscale images) of `Train.csv`.
- **Preprocessing**: the canvas drawing (white strokes on black,
  matching MNIST) is base64-decoded, cropped to the digit's bounding
  box, padded, and resized to 28×28 to match the training distribution
  — the same pipeline classic MNIST demos use.
- **APIs**:
  - `POST /predict` — canvas drawings. Accepts
    `{ "image": "<base64 PNG data URL>" }`.
  - `POST /predict_upload` — arbitrary uploaded images. Accepts
    `multipart/form-data` with a `file` field (PNG/JPG/WEBP/BMP/GIF, up
    to 5 MB). Example:
    ```bash
    curl -F "file=@digit.jpg" https://your-app.example.com/predict_upload
    ```
  - Both return the same JSON shape: `prediction`, `confidence`,
    `top3`, `probabilities`.
- **Upload robustness**: uploaded photos are messier than clean canvas
  strokes (uneven lighting, shadows, paper texture, JPEG noise, phone
  camera EXIF rotation, color images). The upload pipeline handles this
  by auto-detecting ink-vs-background polarity from border pixels,
  denoising, auto-contrasting, and using a data-driven threshold for the
  bounding-box crop — rather than assuming clean black-on-white input.
  It's tested at ~90%+ accuracy on simulated photo conditions, but for
  best results the image should contain a single digit with reasonable
  contrast against its background.
- **Abuse protection on the public upload endpoint**: file size is
  capped at 5 MB (`MAX_CONTENT_LENGTH`), file extension is validated,
  and unreadable/corrupt images return a clean 400 error instead of
  crashing the server. If you deploy this publicly and expect real
  traffic, consider adding rate limiting (e.g. `Flask-Limiter`) on top.

## 4. Deploy

### Option A — Render / Railway / Heroku (Procfile-based)
1. Push this folder to a GitHub repo.
2. Create a new **Web Service** and point it at the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command (already in `Procfile`): `gunicorn app:app`
5. Deploy — the platform will assign a public URL.

### Option B — Docker
Create a `Dockerfile` (not included by default) with:
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
```
Then:
```bash
docker build -t digit-recognizer .
docker run -p 5000:5000 digit-recognizer
```

### Option C — Any VPS
```bash
pip install -r requirements.txt
gunicorn -w 2 -b 0.0.0.0:8000 app:app
```
Put behind Nginx/Caddy for TLS in production.

## 5. Health check

`GET /health` → `{"status": "ok", "model_loaded": true}` — useful for
uptime monitors and platform health checks (Render/Railway ping this
automatically if configured).

## 6. Notes / next steps

- The included model reaches ~97.6% accuracy on a held-out 10% split of
  `Train.csv`. For higher accuracy, swap in a CNN (e.g., with
  TensorFlow/PyTorch) — the `/predict` preprocessing already outputs a
  clean 28×28 normalized array, so only `app.py`'s model-loading and
  inference lines would need to change.
- `model/digit_model.pkl` is ~2.8 MB — small enough to commit directly
  to git and deploy on free hosting tiers without Git LFS.
