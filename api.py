import os
from flask import Flask, jsonify, request, render_template

from predict import predict_image
from config import config

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(config.UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_location)
            pred = predict_image(image_path=image_location)
            print(pred)
            return render_template("index.html", prediction=pred, image_loc=image_file.filename)
    return render_template("index.html", prediction=0, image_loc=None)


if __name__ == '__main__':
  app.run(host="0.0.0.0", debug=True)
