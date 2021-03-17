from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import numpy as np
import json
from keras.models import load_model
from PIL import Image
import traceback
import base64
from io import BytesIO
import re

app = Flask(__name__)

dog_categories = ['beagle', 'chihuahua', 'doberman', 'french_bulldog', 'golden_retriever', 'malamute', 'pug', 'saint_bernard', 'scottish_deerhound', 'tibetan_mastiff']

@app.route('/', methods=['GET'])
def home():
    return "WELCOME"

 
def base64_to_image(base64_str, image_path=None):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    if image_path:
        img.save(image_path)
    return img

def classify(image, l_model):
    img = image.resize((512, 512))
    img = np.array(img)
    img = img.reshape(-1, 512, 512, 3)
    probs = l_model.predict(img)
    y_pred = np.argmax(probs) 
    probs = [item for sublist in probs for item in sublist] # flatten the 2d list to 1d list
    score = (probs[y_pred]/np.sum(probs)) * 100 # calculate the percentage scores from probabilities
    score = "%.2f" % score # limit upto two decimal points
    return y_pred, score

@app.route('/predict', methods=['POST'])
def predict():
    try:
        modelfile = 'final_model.h5'
        model = load_model(modelfile)
        content = request.get_json()
        image_str = content['image']
        img = base64_to_image(image_str)
        pred, score = classify(img, model)
        pred = dog_categories[pred]
        print(f"breed: {pred}, score:{score}")
        return jsonify({'breed': str(pred), 'score': str(score)})
    except:
        print("Error")
        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    port = 5001
    app.run(host='0.0.0.0', port=port, debug=True)
