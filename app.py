from flask import Flask, render_template, request
import tensorflow as tf

import logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# Render/CPU pequeña: evitar contención
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Evitar cuelgues por fotos enormes (5 MB)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

import numpy as np
from PIL import Image
import json, io, os
from PIL import Image, ImageOps, ImageFilter


# ---- Parche: registrar softmax_v2 para Keras 3 / Python 3.13 ----
from keras.saving import register_keras_serializable

@register_keras_serializable(package="keras.activations", name="softmax_v2")
def softmax_v2(x):
    return tf.keras.activations.softmax(x)

custom_objects = {"softmax_v2": softmax_v2}
# -----------------------------------------------------------------

app = Flask(__name__)

MODELO_PATH = "modelo_fashion.keras"
CLASES_PATH = "clases.json"

# Cargar modelo y clases
modelo = tf.keras.models.load_model(
    MODELO_PATH,
    custom_objects=custom_objects,
    compile=False,   # para inferencia no necesitamos deserializar optimizer/loss
    safe_mode=False  # habilita objetos custom registrados
)
import numpy as np

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

def preprocess(img_pil: Image.Image) -> np.ndarray:
    """
    Prepara cualquier foto para el modelo Fashion-MNIST:
    - grises
    - recorte centrado a cuadrado
    - resize 28x28
    - autocontraste + leve suavizado
    - inversión si el fondo es claro
    - normaliza [0,1] y da shape (1,28,28,1)
    """
    # 1) A grises
    img = img_pil.convert("L")

    # 2) Crop centrado a cuadrado (fit center)
    w, h = img.size
    side = min(w, h)
    left   = (w - side) // 2
    top    = (h - side) // 2
    right  = left + side
    bottom = top + side
    img = img.crop((left, top, right, bottom))

    # 3) Redimensionar a 28x28
    img = img.resize((28, 28), Image.BILINEAR)

    # 4) Autocontraste para resaltar figura (reduce extremos del histograma)
    #    cutoff=2 evita quemar blancos/negros por ruido
    img = ImageOps.autocontrast(img, cutoff=2)

    # 5) Suavizado muy leve para sacar grano/ruido de fotos
    img = img.filter(ImageFilter.SMOOTH)

    # 6) Convertir a array y normalizar
    arr = np.asarray(img).astype("float32") / 255.0

    # 7) Heurística de inversión:
    #    Si el promedio es alto, asumimos fondo claro → invertimos
    if arr.mean() > 0.55:
        arr = 1.0 - arr

    # 8) (Opcional) recentrado por intensidad:
    #    Si querés, podés binarizar suave para resaltar la prenda
    #    thr = arr.mean() * 0.9
    #    arr = np.where(arr > thr, arr, arr * 0.5)

    # 9) Añadir ejes: (1, 28, 28, 1)
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr

@app.route("/", methods=["GET", "POST"])
def index():
    pred_text = None
    top3 = None
    if request.method == "POST":
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

# if __name__ == "__main__":
#     # host=0.0.0.0 si querés acceder desde otra máquina en tu red local
#     app.run(host="0.0.0.0", port=5000, debug=True)

import socket

if __name__ == "__main__":
    # Detectar la IP local de tu máquina
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"\n🌐 Acceso desde esta PC:   http://127.0.0.1:5000")
    print(f"📱 Acceso desde tu red WiFi: http://{local_ip}:5000\n")

    # Correr Flask abierto a la red local
    app.run(host="0.0.0.0", port=5000, debug=True)


@app.get("/health")
def health():
    return "ok", 200

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # 👇 este log te confirma que el navegador envió el POST
        app.logger.info(
            "POST / recibido. files=%s form=%s",
            list(request.files.keys()), dict(request.form)
        )
        # ... tu preprocesado y predicción con 'modelo' ...
        # pred = modelo.predict(...)
        # return render_template("index.html", ...)

    # GET:
    return render_template("index.html")


