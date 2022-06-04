import os

from flask import Flask, render_template, request
from deploy import predict
app = Flask(__name__)
UPLOAD_FOLDER = "./static"

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            pred = predict(image_location)
            return render_template("test.html", prediction = pred, image_loc = image_file.filename)
    return render_template("test.html", prediction = None, image_loc = None)

if __name__ == "__main__":
    app.run(port=12000, debug=True)