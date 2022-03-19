
from flask import Flask, render_template, url_for,request, redirect, send_file

from flask_ngrok import run_with_ngrok
from datetime import datetime
import re
import numpy as np
import h5py
import math
import io
from PIL import Image
import werkzeug
import os
from keras.models import Model
from keras.layers import Input, Activation, BatchNormalization, Conv2D, Conv3D
from keras.layers import Lambda, Concatenate, MaxPooling2D, Maximum, Add
from keras.initializers import RandomNormal
from tensorflow.keras.optimizers import schedules, SGD
from keras.callbacks import Callback
# from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from PIL import Image 

import cv2
import base64

import numpy as np

from PIL import Image
from io import BytesIO
def Guidedfilter(im,p,r,eps):
	mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
	mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
	mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
	cov_Ip = mean_Ip - mean_I*mean_p
	mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
	var_I   = mean_II - mean_I*mean_I
	a = cov_Ip/(var_I + eps)
	b = mean_p - a*mean_I
	mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
	mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))
	q = mean_a*im + mean_b
	return q

def TransmissionRefine(im,et):
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	gray = np.float64(gray)/255
	r = 60
	eps = 0.0001
	t = Guidedfilter(gray,et,r,eps)
	return t
def TransmissionModel(input_shape):
    
    X_input = Input(input_shape, name = 'input1')

    # CONV -> RELU Block applied to X
    X = Conv2D(16, (3, 3), strides = (1, 1), name = 'conv1')(X_input)
    X = Activation('relu', name = 'activation1')(X)

    # SLICE Block applied to X
    X1 = Lambda(lambda X: X[:,:,:,:4], name = 'slice1')(X)
    X2 = Lambda(lambda X: X[:,:,:,4:8], name = 'slice2')(X)
    X3 = Lambda(lambda X: X[:,:,:,8:12], name = 'slice3')(X)
    X4 = Lambda(lambda X: X[:,:,:,12:], name = 'slice4')(X)

    # MAXIMUM Block applied to 4 slices
    X = Maximum(name = 'merge1_maximum')([X1,X2,X3,X4])

    # CONV BLock for multi-scale mapping with filters of size 3x3, 5x5, 7x7
    X_3x3 = Conv2D(16, (3, 3), strides = (1, 1), padding = 'same', name = 'conv2_3x3')(X)
    X_5x5 = Conv2D(16, (5, 5), strides = (1, 1), padding = 'same', name = 'conv2_5x5')(X)
    X_7x7 = Conv2D(16, (7, 7), strides = (1, 1), padding = 'same', name = 'conv2_7x7')(X)

    # CONCATENATE Block to join 3 multi-scale layers
    X = Concatenate(name = 'merge2_concatenate')([X_3x3,X_5x5,X_7x7])

    # MAXPOOL layer of filter size 7x7
    X = MaxPooling2D((7, 7), strides = (1, 1), name = 'maxpool1')(X)
    
    # CONV -> RELU BLock
    X = Conv2D(1, (8, 8), strides = (1, 1), name = 'conv3')(X)
    X = Activation('relu', name = 'activation2')(X)

    # Create Keras model instance
    model = Model(inputs = X_input, outputs = X, name='TransmissionModel')

    return model

def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")


app = Flask(__name__)
run_with_ngrok(app) 

def ResidualBlock(X, iter):
    
    # Save the input value
    X_shortcut = X
    
    # BATCHNORMALIZATION -> CONV Block
    X = BatchNormalization(axis = 3, name = 'res_batchnorm' + str(iter))(X)
    X = Conv2D(1, (3, 3), strides = (1,1), padding = 'same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001), name = 'res_conv' + str(iter))(X)

    # Add shortcut value to main path, and pass it through a RELU activation
    X = Add(name = 'res_add'+ str(iter))([X,X_shortcut])
    X = Activation('relu', name = 'res_activation'+ str(iter))(X)
    
    return X

def ResidualModel(input_shape):
    
    X_input = Input(input_shape, name = 'input1')

    # CONV -> RELU Block applied to X
    X = Conv2D(16, (3, 3), strides = (1, 1), padding = 'same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001), name = 'conv1')(X_input)
    X = Activation('relu', name = 'activation1')(X)

    for i in range(17):
        X = ResidualBlock(X, i)
    
    # CONV BLock
    X = Conv2D(3, (3, 3), strides = (1, 1), padding = 'same', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001), name = 'conv2')(X)
    X = Activation('relu', name = 'activation2')(X)

    # Create Keras model instance
    model = Model(inputs = X_input, outputs = X, name='TransmissionModel')

    return model
def dehaze_image(img):
    input_image_orig = np.asarray(img)/255.0
    input_image = np.pad(input_image_orig,((7,8), (7,8), (0,0)),'symmetric') 

    model = TransmissionModel(input_image.shape)
    model.load_weights('/content/defoggy.ai/defoggy/transmodel_weights.h5')

    input_image = np.expand_dims(input_image, axis=0)
    trans_map_orig = model.predict(input_image)
    trans_map = trans_map_orig.reshape(input_image_orig.shape[:2])
    trans_map_refine = TransmissionRefine((input_image_orig*255.0).astype('uint8'),trans_map)

    res_map_input = input_image_orig/np.expand_dims(trans_map_refine, axis=(0,3))

    model = ResidualModel(res_map_input.shape[1:])
    model.load_weights('/content/defoggy.ai/defoggy/resmodel_weights.h5')
    res_map_output = model.predict(np.clip(res_map_input,0,1))

    haze_free_image = (res_map_input-res_map_output)
    haze_free_image = np.clip(haze_free_image,0,1)

    return haze_free_image[0]

@app.route('/', methods=['GET'])
def index():
    '''
    Render the main page
    '''
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''
    print("hello world")
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        out = dehaze_image(img)
        print(out.max(),out.min())
        print("hello world5")
        Image.fromarray((out*255).astype(np.uint8)).save("/content/defoggy.ai/defoggy/static/output.jpg")
        # Serialize the result, you can add additional fields

        return render_template('submit.html',user_img = "/content/defoggy.ai/defoggy/static/output.jpg")
    return None

@app.route('/show', methods=['GET'])
def show():
  if(os.path.exists('/content/defoggy.ai/defoggy/static/output.jpg')):
    # return send_file("/content/output.jpg")
    return render_template("show.html",filename="/static/output.jpg")
  else:
    return "Image not found"


if __name__ == "__main__":
    app.run()
