from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
from deepface import DeepFace
import base64
import cv2
import numpy as np
import os
import tensorflow as tf
import tempfile

app = Flask(__name__, static_folder="frontend", static_url_path="/")

# Disable TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# MongoDB Atlas connection
MONGO_URI = "mongodb+srv://devidekshina7:krXjjSaC8AwYYBxu@cluster0.qpkff.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["FaceRecognitionDB"]
faces_collection = db["faces"]  # Use "faces" collection instead of storing locally

try:
    client.admin.command("ping")  # Check MongoDB connection
    print("✅ Connected to MongoDB Atlas successfully!")
except Exception as e:
    print(f"❌ Failed to connect to MongoDB Atlas: {e}")

# ---------------------- Upload Reference Image to MongoDB ----------------------
@app.route("/upload_reference", methods=["POST"])
def upload_reference():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files["file"]
    filename = file.filename
    image_data = file.read()  # Read binary image data

    # Convert image to Base64 for storage in MongoDB
    encoded_image = base64.b64encode(image_data).decode("utf-8")

    # Store in MongoDB
    faces_collection.insert_one({"name": filename, "image": encoded_image})
    print(f"✅ Reference image {filename} stored in MongoDB!")

    return jsonify({"message": "Reference image uploaded successfully!", "filename": filename})


# ---------------------- Camera Captured Image Upload ----------------------
import tempfile  # Add this import

@app.route("/camera_recognition", methods=["POST"])
def camera_recognition():
    data = request.json
    if "image" not in data:
        return jsonify({"error": "No image received!"}), 400

    try:
        # Decode Base64 string
        image_str = data["image"].split(",")[1]  # Remove "data:image/jpeg;base64," part
        image_data = base64.b64decode(image_str)
        image_np = np.frombuffer(image_data, np.uint8)

        if image_np.size == 0:
            return jsonify({"error": "Failed to decode image!"}), 400

        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Failed to process image!"}), 400

        # Convert to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print("📸 Captured image processed.")

        # Save the image as a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, image)  # Save as a file

        # Perform face recognition against stored faces in MongoDB
        best_match = None
        reference_faces = faces_collection.find({}, {"_id": 0, "image": 1, "name": 1})

        for ref_face in reference_faces:
            ref_name = ref_face["name"]
            ref_image_data = base64.b64decode(ref_face["image"])
            ref_np_arr = np.frombuffer(ref_image_data, np.uint8)
            ref_img = cv2.imdecode(ref_np_arr, cv2.IMREAD_COLOR)

            # Save reference image as a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as ref_temp_file:
                ref_temp_path = ref_temp_file.name
                cv2.imwrite(ref_temp_path, ref_img)

            try:
                print(f"🔍 Comparing captured image with reference: {ref_name}")
                result = DeepFace.verify(img1_path=temp_path, img2_path=ref_temp_path, model_name="Facenet512", distance_metric="cosine", detector_backend="mtcnn")

                if result["verified"]:
                    best_match = ref_name
                    break  # Stop after finding a match
            except Exception as e:
                print(f"❌ Error comparing with {ref_name}: {e}")

        # Cleanup temporary files
        os.remove(temp_path)
        if best_match:
            return jsonify({"message": "Match found!", "matched_with": best_match})
        else:
            return jsonify({"message": "No match found!"})

    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return jsonify({"error": "Error processing image!"}), 500

# ---------------------- Face Matching Endpoint ----------------------
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files["file"]
    filename = file.filename
    image_data = file.read()

    # Convert image to Base64 and store in MongoDB
    encoded_image = base64.b64encode(image_data).decode("utf-8")

    # Store in MongoDB
    faces_collection.insert_one({"name": filename, "image": encoded_image})
    print(f"✅ Uploaded image {filename} stored in MongoDB!")

    # Perform face matching
    best_match = None
    reference_faces = faces_collection.find({}, {"_id": 0, "image": 1, "name": 1})

    for ref_face in reference_faces:
        ref_name = ref_face["name"]
        ref_image_data = base64.b64decode(ref_face["image"])
        ref_np_arr = np.frombuffer(ref_image_data, np.uint8)
        ref_img = cv2.imdecode(ref_np_arr, cv2.IMREAD_COLOR)

        try:
            print(f"🔍 Comparing {filename} with reference: {ref_name}")
            result = DeepFace.verify(img1_path=image_data, img2_path=ref_img, model_name="Facenet512", distance_metric="cosine", detector_backend="mtcnn")

            print(f"✅ Face verification result: {result}")

            if result["verified"]:
                best_match = ref_name
                break  # Stop after finding the first match
        except Exception as e:
            print(f"❌ Error comparing with {ref_name}: {e}")

    return jsonify({"message": "Match found!", "matched_with": best_match}) if best_match else jsonify({"message": "No match found!"})


# ---------------------- Serve Frontend ----------------------
@app.route("/")
def serve_frontend():
    frontend_path = os.path.join(os.getcwd(), "..", "frontend", "index.html")
    return send_file(frontend_path)


if __name__ == "__main__":
    app.run(debug=True)
