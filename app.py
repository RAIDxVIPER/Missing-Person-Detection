from flask import Flask, request, render_template
import os
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
DATABASE_FOLDER = "database"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)

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

    return render_template("result.html", match=matched_image)

def verify_face(image1, image2):
    """Compare two images using DeepFace and return similarity."""
    try:
        result = DeepFace.verify(image1, image2, model_name="Facenet")
        return result.get('verified', False)
    except Exception as e:
        print(f"Error verifying face: {e}")
        return False

def identify_missing_person(missing_person_img, database_path):
    """Compare uploaded image with database images."""
    for person_img in os.listdir(database_path):
        person_img_path = os.path.join(database_path, person_img)
        if verify_face(missing_person_img, person_img_path):
            return person_img_path
    return None

if __name__ == "__main__":
    app.run(debug=True)
