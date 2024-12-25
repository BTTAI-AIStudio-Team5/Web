from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import torch
import joblib
from CV.cvmodel import load_model, process_img
from STRIPS_planner.strips import final_main

from LLM.CVtoSTRIPS import get_initial_state, generate_strips_plan
from LLM.llm_openai import strips_to_NL

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    if file:
        print(f"Receiving file {file.filename}...")
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        print(f"File saved to {filepath}")

        # Load model and process image
        print("Loading model...")
        try:
            model = load_model()
            label_encoder = joblib.load("label_encoder.pkl")
            print("Processing image...")
            result_img, info, ROI_number, imgs= process_img(filepath, model)
            print("Detection info:", info)

            # Format detection results
            results = []
            # Save and encode the result image
            output_path = os.path.join(
                app.config["UPLOAD_FOLDER"], "result_" + secure_filename(file.filename)
            )
            cv2.imwrite(output_path, result_img)
            # Add the image path to results
            results.append(
                {
                    "message": f'Step 1: Detection of Objects.\n<img src="../uploads/result_{secure_filename(file.filename)}" class="img-fluid" alt="Processed Image">',
                    "type": "processing"
                }
            )
            for i, detection in enumerate(info):
                results.append(
                    {
                        "message": f"Object detected! Object {i+1}: {detection['category']} at coordinates {detection['bbox']}"
                    }
                )

            initial_state = get_initial_state(info)
            print("Initial state:", initial_state)
            results.append({"message": f"Step 2: STRIPS Problem Definition.<br/>{initial_state}", "type": "processing"})

            print("Generating STRIPS plan...")
            filename = "strips_problem.txt"
            generate_strips_plan(initial_state, filename)

            # Run STRIPS planner TODO: fix this because the import is not working because the directory name has a dash in it
            print("Running STRIPS planner...")
            strips_solution = final_main(filename="strips_problem.txt")
            # This line formats the solution to be a string
            strips_solution = "Plan: {0}".format(
                " -> ".join([x.simple_str() for x in strips_solution])
            )
            results.append(
                {"message": f"Step 3: STRIPS Solution Found!<br/><b>{strips_solution}</b>", "type": "processing"}
            )

            # TODO: Add LLM here
            print("Running LLM")
            nl_solution = strips_to_NL(strips_solution)
            results.append({"message": f'Step 4: Natural Language Solution.<br/><b>{nl_solution}</b><br/><img src="../uploads/{secure_filename(file.filename)}" class="img-fluid" alt="Original Image">'})

            print("Returning results...")
            return jsonify({"messages": results})
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
