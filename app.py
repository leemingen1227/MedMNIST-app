import os

from flask import Flask, render_template, request
from deploy import predict
from inference import evaluate
app = Flask(__name__)
UPLOAD_FOLDER = "./static"
TEST_FOLDER = "./test_image"

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        file_location = os.path.join(
                TEST_FOLDER,
                image_file.filename
            )
        if str(file_location).endswith('npz'):
            file_location = os.path.join(
                TEST_FOLDER,
                image_file.filename
            )
            image_file.save(file_location)
            auc, acc = evaluate(TEST_FOLDER)
            return render_template("test.html", prediction = None,
            auc = auc, acc = acc, image_loc = None, file_loc = file_location )
            
        else:
            image_file = request.files["image"]
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location)
            return render_template("test.html", prediction = pred, 
                    auc = None, acc = None, image_loc = image_file.filename, file_loc = None)
    return render_template("test.html", prediction = None,
            auc = None, acc = None, image_loc = None, file_loc = None)

if __name__ == "__main__":
    app.run(host = '0.0.0.0', debug=True)