
from flask import Flask, request, jsonify, send_from_directory
from deepface import DeepFace
import os
import pandas as pd
import base64
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="../frontend", static_url_path="")

# Configure paths
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
DB_PATH = os.path.join(os.getcwd(), "media", "db")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Ensure required folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

# Check if a file is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to serve the frontend
@app.route("/")
def serve_frontend():
    return send_from_directory("../frontend", "index.html")

# Route for file uploads (Image Upload)
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file!"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        return recognize_face(filepath)

    return jsonify({"error": "Invalid file format. Allowed: png, jpg, jpeg"})

# Route for camera-based face recognition
@app.route("/camera_recognition", methods=["POST"])
def camera_recognition():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image data received!"})

    image_data = data["image"].split(",")[1]  # Remove Base64 prefix
    image_bytes = base64.b64decode(image_data)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # Save the captured image temporarily
    temp_image_path = os.path.join(UPLOAD_FOLDER, "camera_capture.jpg")
    cv2.imwrite(temp_image_path, img)

    return recognize_face(temp_image_path)

# Face recognition function
def recognize_face(image_path):
    try:
        result = DeepFace.find(
            img_path=image_path,
            db_path=DB_PATH,
            model_name="Facenet512",
            distance_metric="cosine",
            threshold=0.8,
            detector_backend="mtcnn",
            enforce_detection=False,
        )

        if isinstance(result, list) and len(result) > 0:
            df = result[0]
            if isinstance(df, pd.DataFrame) and not df.empty:
                file_names = df["identity"].tolist()
                return jsonify({"success": True, "matches": file_names})
            else:
                return jsonify({"success": True, "matches": []})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
