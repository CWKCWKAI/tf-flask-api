# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 02:13:55 2020

@author: Christoph
"""

#%% import packages
import os
import inference_functions
from flask import Flask, request, Response, render_template,jsonify
import numpy as np
from PIL import Image
import io
import base64
from io import BytesIO
import tensorflow as tf
import numpy as np
from inference_functions import *


#%% BEGIN APP
app = Flask(__name__)


# HEALTH END POINT
@app.route('/health')
def hello_world():
    return 'API live'


# INFERENCE 
@app.route('/detect_signature', methods=['POST'])
def detect_signature():
    # read b64 image data and decode into numpy array
    img = request.data
    base64_decoded = base64.b64decode(img)
    np_img = np.array(Image.open(io.BytesIO(base64_decoded)))
    
    # run inference
    image_w_inference = show_inference(detection_model, np_img, 0.5)
    print(image_w_inference)
    
    # make np array back to image, b64 encode
    img_out = Image.fromarray(np.uint8(image_w_inference))
    buffer = BytesIO()
    img_out.save(buffer, format="JPEG")
    img_out_str = base64.b64encode(buffer.getvalue())
    
    # return b64 image with inference
    return jsonify({'image_response':img_out_str.decode('utf-8')})


#%% run app
if __name__ == '__main__':
    app.run(debug=True,
    use_reloader=False,
    host = '0.0.0.0',
    port = 8080)