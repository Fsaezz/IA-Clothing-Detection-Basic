from flask import Flask, render_template, request
import logging
import json, io, os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
from keras.saving import register_keras_serializable

# ---------------- 1) Crear la app ANTES de usar app.logger/config ----------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB

# CPU pequeña en Render: menos hilos para TF
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ---------------- 2) Parche softmax_v2 para Keras 3 ----------------
@register_keras_serializable(package="keras.activations", name="softmax_v2")
def softmax_v2(x):
    return tf.keras.activations.softmax(x)

custom_objects = {"softmax_v2": softmax_v2}

# ---------------- 3) Cargar modelo y clases UNA SOLA VEZ ----------------
MODELO_PATH = "modelo_fashion.keras"
CLASES_PATH = "clases.json"

modelo = tf.keras.models.load_model(
    MODELO_PATH,
    custom_objects=custom_objects,
    compile=False,
    safe_mode=False
)

def _warmup_model():
    try:
        dummy = np.zeros((1, 28, 28, 1), dtype=np.float32)
        _ = modelo.predict(dummy)
        app.logger.info("✅ Warmup del modelo completado")
    except Exception as e:
        app.logger.warning(f"⚠️ Warmup falló: {e}")

_warmup_model()

if os.path.exists(CLASES_PATH):
    with open(CLASES_PATH, "r", encoding="utf-8") as f:
        nombre_clases = json.load(f)
else:
    nombre_clases = [
        "T-shirt/top","Trouser","Pullover","Dress","Coat",
        "Sandal","Shirt","Sneaker","Bag","Ankle boot"
    ]

# ---------------- 4) Preprocesado ----------------
def preprocess(img_pil: Image.Image) -> np.ndarray:
    img = img_pil.convert("L")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((28, 28), Image.BILINEAR)
    img = ImageOps.autocontrast(img, cutoff=2)
    img = img.filter(ImageFilter.SMOOTH)
    arr = np.asarray(img).astype("float32") / 255.0
    if arr.mean() > 0.55:
        arr = 1.0 - arr
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr

# ---------------- 5) Rutas ----------------
@app.get("/health")
def health():
    return "ok", 200

@app.route("/", methods=["GET", "POST"])
def index():
    pred_text, top3 = None, None
    if request.method == "POST":
        app.logger.info("POST / recibido. files=%s form=%s",
                        list(request.files.keys()), dict(request.form))
        file = request.files.get("imagen")
        if file and file.filename:
            img = Image.open(io.BytesIO(file.read()))
            x = preprocess(img)
            proba = modelo.predict(x, verbose=0)[0]
            idx = int(np.argmax(proba))
            pred = nombre_clases[idx]
            conf = float(proba[idx]) * 100
            pred_text = f"Predicción: {pred} ({conf:.1f}%)"
            top_idx = np.argsort(proba)[-3:][::-1]
            top3 = [(nombre_clases[i], float(proba[i]) * 100) for i in top_idx]
    return render_template("index.html", pred_text=pred_text, top3=top3)

# ---------------- 6) Modo local (Render lo ignora) ----------------
if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"\n🌐 Local: http://127.0.0.1:5000")
    print(f"📱 WiFi : http://{local_ip}:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)

