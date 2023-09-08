import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model

app = Flask(__name__)
model = load_model('HSC_model.h5')

def predict(img_path):
    img = tf.io.read_file(img_path)  # Read the image file
    img = tf.image.decode_image(img, channels=3)  # Ensure it has 3 channels (color image)
    img = tf.image.resize(img, (256, 256))  # Resize the image
    img = tf.cast(img, tf.float32) / 255.0  # Normalize the image to [0, 1]

    x = model.predict(tf.expand_dims(img, 0), verbose=0)  # Predict
    if x > 0.5:
        return 'sad'
    else:
        return 'happy'
    

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)