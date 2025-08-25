import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask app
app = Flask(__name__)

# Load trained model
MODEL_PATH = "saved_models/best_model.h5"
model = load_model(MODEL_PATH)

# Class labels (adjust based on your dataset)
class_names = ["Bottle", "Cup"]

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def predict_image(img_path):
    """Load and preprocess image, then predict."""
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    return class_names[0] if prediction < 0.5 else class_names[1]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Save uploaded file
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            # Predict
            result = predict_image(filepath)

            return render_template("index.html", prediction=result, image_path=filepath)

    return render_template("index.html", prediction=None, image_path=None)


if __name__ == "__main__":
    app.run(debug=True)
