import pandas as pd
from deepface import DeepFace  

# Run face recognition
result = DeepFace.find(
    img_path="./media/t1.jpg",  
    db_path="./media/db",  
    model_name="Facenet512",  
    distance_metric="cosine",  
    threshold=0.8,  
    detector_backend="mtcnn",  
    enforce_detection=False  
)  

# ✅ Extract and print file names
if isinstance(result, list) and len(result) > 0:
    df = result[0]  # Extract the first DataFrame
    if isinstance(df, pd.DataFrame) and not df.empty:
        file_names = df["identity"].tolist()  # Extract file names
        print("✅ Detected face(s) in:", file_names)
    else:
        print("❌ No match found.")
else:
    print("⚠️ DeepFace returned an empty or invalid response:", result)
