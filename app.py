from flask import Flask, redirect, url_for, render_template, request
from detect_plate import main
import os
from flask_cors import CORS
UPLOAD_PATH = "./static"


app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
  uploaded_file = request.files['file']
  if uploaded_file.filename != '':
      # print uploaded_file.filename
      uploaded_file.save(os.path.join(UPLOAD_PATH,"test.mp4"))
  return redirect(url_for('plate_recognize'))

@app.route("/plate")
def plate_recognize():
  return render_template('plate.html')

@app.route("/plate", methods=['GET'])
def run():
  print("startttttt")
  main("/static/test.mp4")
  return "ok"


if __name__ == "__main__":
	app.run(debug=False)