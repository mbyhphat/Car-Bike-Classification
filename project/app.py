from flask import Flask, render_template, jsonify, request
import cv2
from flask_cors import CORS
from skimage.feature import hog
import joblib
from io import BytesIO
import requests
import numpy as np
from PIL import Image
import base64

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualize=vis, feature_vector=feature_vec)
        return features

def read_image(url_image):
    response = requests.get(url_image)
    image_bytes = BytesIO(response.content)

    # Đọc hình ảnh từ byte data
    image = cv2.imdecode(np.frombuffer(image_bytes.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))

    # Chuyển đổi ảnh màu thành ảnh xám (grayscale)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray_image

class_name = {0: "Motorbike", 1: "Car"}
model_svc = joblib.load("project/models/best_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    files = request.get_json()
    if not files:  # Nếu queue không có ảnh nào
        return jsonify({"error": "Queue is empty"}), 400
    results = {
        "labels": [],
        "hog_image": []
    }
    for file in files['queue']:
        img_gray = read_image(file['src'])
        hog_feature, hog_image = get_hog_features(img_gray, vis=True)
        y_pred = model_svc.predict(hog_feature.reshape(1, -1))
        print(y_pred)

        hog_image_pil = Image.fromarray((hog_image * 255).astype(np.uint8))  # Chuyển đổi ảnh về dạng uint8
        buffer = BytesIO()
        hog_image_pil.save(buffer, format="JPEG")
        hog_image_base64 = base64.b64encode(buffer.getvalue()).decode()

        results["labels"].append(class_name[y_pred[0]])
        results['hog_image'].append(hog_image_base64)

    return jsonify({"ok": True, "results": results})

if __name__ == "__main__":
    app.run(debug=True)