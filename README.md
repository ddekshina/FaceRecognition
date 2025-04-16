# FaceRecognition
This project develops an automated face identification system allowing users to upload reference faces and then identify matches in uploaded or webcam-captured images through a user-friendly interface.

**Flow of Execution**

1.  **User Interaction (Frontend):** The user interacts with the frontend, either uploading an image file or capturing an image using the device's camera.
2.  **Image Submission (Frontend):** The frontend sends the image data to the backend API endpoint. For camera images, this is likely done as a Base64 encoded string.
3.  **Image Processing (Backend):**
    *   The backend API receives the image data.
    *   The backend utilizes the `deepface` library to perform face detection and recognition.
    *   The backend interacts with the MongoDB database to access stored reference images.
4.  **Result Return (Backend):** The backend returns the recognition result (match found, no match, error) as a JSON response.
5.  **Result Display (Frontend):**  The frontend receives the JSON response from the backend and displays the results to the user.

   ## Tech Stack Used

**Languages:**

*   Python
*   HTML
*   JavaScript

**Frameworks and Libraries:**

*   **Backend:**
    *   Flask: Web framework for creating the API and serving the frontend.
    *   pymongo: MongoDB driver for Python.
    *   deepface: Face detection and recognition library.
    *   cv2 (OpenCV): Image processing library.
    *   numpy: Numerical operations (used with `cv2`).
    *   tensorflow: DeepFace's underlying deep learning framework.
    *   base64: Encoding/decoding images.
    *   tempfile: Temporary file management.
    *   os: Operating system interactions (file operations).
*   **Frontend:**
    *   Vanilla JavaScript:  Core scripting language for the user interface.
    *   `fetch` API:  For making HTTP requests to the backend.
    *   `navigator.mediaDevices.getUserMedia`: For camera access.

**Data Storage:**

*   MongoDB Atlas: Database for storing reference images.
