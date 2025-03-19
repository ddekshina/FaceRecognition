from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
from deepface import DeepFace
import base64
import cv2
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__, static_folder="frontend", static_url_path="/")

# Disable TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# MongoDB Atlas connection
MONGO_URI = "mongodb+srv://devidekshina7:krXjjSaC8AwYYBxu@cluster0.qpkff.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["FaceRecognitionDB"]
reference_collection = db["reference_images"]

try:
    client.admin.command("ping")  # Check MongoDB connection
    print("✅ Connected to MongoDB Atlas successfully!")
except Exception as e:
    print(f"❌ Failed to connect to MongoDB Atlas: {e}")

# Upload folder setup
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------- Reference Image Upload ----------------------
@app.route("/upload_reference", methods=["POST"])
def upload_reference():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files["file"]
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    print(f"📸 Reference image received: {filename}")

    # Convert image to Base64 and store in MongoDB
    with open(filepath, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    result = reference_collection.insert_one({"name": filename, "image": image_data})
    print(f"✅ Reference image {filename} stored in MongoDB!")

    return jsonify({"message": "Reference image uploaded successfully!", "filename": filename})


# ---------------------- Face Matching Endpoint ----------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files["file"]
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    print(f"📸 Uploaded image received: {filename}")

    # Fetch stored reference images from MongoDB
    reference_faces = reference_collection.find({}, {"_id": 0, "image": 1, "name": 1})

    best_match = None
    for ref_face in reference_faces:
        ref_name = ref_face["name"]
        ref_image_data = base64.b64decode(ref_face["image"])

        # Convert Base64 to NumPy array and decode
        ref_np_arr = np.frombuffer(ref_image_data, np.uint8)
        ref_img = cv2.imdecode(ref_np_arr, cv2.IMREAD_COLOR)

        # Save temp reference image
        ref_image_path = os.path.join(UPLOAD_FOLDER, ref_name)
        cv2.imwrite(ref_image_path, ref_img)

        # Face detection check before verification
        try:
            detected_faces = DeepFace.extract_faces(filepath, detector_backend="mtcnn")
            if detected_faces is None:
                print(f"❌ No face detected in uploaded image: {filename}")
                return jsonify({"error": "No face detected in the image!"}), 400
        except Exception as e:
            print(f"❌ Error detecting face in {filename}: {e}")
            return jsonify({"error": "Face detection failed!"}), 500

        # Perform face verification using DeepFace
        try:
            print(f"🔍 Comparing {filename} with reference: {ref_name}")
            result = DeepFace.verify(img1_path=filepath, img2_path=ref_image_path, model_name="Facenet512", distance_metric="cosine", detector_backend="mtcnn")

            print(f"✅ Face verification result: {result}")

            if result["verified"]:
                best_match = ref_name
                break  # Stop after finding the first match
        except Exception as e:
            print(f"❌ Error comparing with {ref_name}: {e}")

    if best_match:
        return jsonify({"message": "Match found!", "matched_with": best_match})
    else:
        return jsonify({"message": "No match found!"})


# ---------------------- Serve Frontend ----------------------
@app.route("/")
def serve_frontend():
    frontend_path = os.path.join(os.getcwd(), "..", "frontend", "index.html")
    return send_file(frontend_path)


if __name__ == "__main__":
    app.run(debug=True)
