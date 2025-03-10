from flask import Flask, request, render_template

import os
import cv2
import numpy as np
from deepface import DeepFace
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
DATABASE_FOLDER = r"archive (3)\lfw-deepfunneled\lfw-deepfunneled"  # Update to the LFW dataset folder

def verify_face(image1, image2):
    """Compares two images using DeepFace and returns similarity."""
    try:
        result = DeepFace.verify(image1, image2, model_name="Facenet")
        return result['verified'], result['distance']
    except Exception as e:
        print("Error during face verification:", e)
        return False, None
    
def identify_missing_person(missing_person_img, database_path):
    """Compares a missing person's image against a database of found persons."""
    for person_folder in os.listdir(database_path):
        person_folder_path = os.path.join(database_path, person_folder)
        print(person_folder_path)
        if os.path.isdir(person_folder_path):  # Check if it's a directory
            for person_img in os.listdir(person_folder_path):
                person_img_path = os.path.join(person_folder_path, person_img)
                match, distance = verify_face(missing_person_img, person_img_path)

                if match:
                    print(f"Match Found: {person_img_path} (Distance: {distance:.4f})")
                    return person_img_path  # Return the matched image path
    print("No match found.")
    return None

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    matched_image = identify_missing_person(file_path, DATABASE_FOLDER)

    if matched_image:
        return render_template("result.html", match=matched_image)
    else:
        return render_template("result.html", match=None)


if __name__ == "__main__":
    app.run(debug=False)