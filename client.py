# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 09:00:33 2021

@author: Christoph
"""
import os
os.chdir(r'C:\Users\Christoph\Documents\Python Scripts\chequeclearance\tensorflow_flask')

import base64
import requests
import time 
from PIL import Image
import io
import numpy as np

#%% test health endpoint
health_url = 'http://192.168.99.1:8080/health'
response = requests.get(health_url, verify=False)
print(response.text)

#%% send image 
with open("images/TEST_2.jpg", "rb") as imageFile:
    img_in = base64.b64encode(imageFile.read())
    
url = 'http://192.168.99.103:8080/detect_signature'
try:
    response = requests.post(url, data=img_in)
    img_out_b64 = base64.b64decode(response.json()['image_response'])
    img_out = Image.open(io.BytesIO(img_out_b64))
    np_img_out2 = np.array(img_out)
    print(response.status_code)
    print(response.json())
except Exception as e:
    print(e)
    

