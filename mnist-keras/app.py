from flask import Flask ,render_template,request,redirect,url_for,send_from_directory

from scipy.misc import imsave,imread,imresize
from skimage import color

import base64
import numpy as np  

import keras.models

from keras.models import model_from_json
import os,re,sys

# Canvas image code adapted from: https://github.com/llSourcell/how_to_deploy_a_keras_model_to_production/blob/master/app.py



app=Flask(__name__)





@app.route('/about')
def about():
	return render_template("about.html")




@app.route('/predict',methods=["GET","POST"])
def predict():

	imgData=request.get_data()

	imgstr=re.search(b'base64,(.*)', imgData).group(1)

	with open('digit.png','wb') as output:
		output.write(base64.decodebytes(imgstr))


	img= imread('digit.png',mode='L')

	img=np.invert(img)

	img=imresize(img,(28,28))

	img=img.reshape(1,28,28,1)

	json_file = open('model/mnist.json','r')

	model_json = json_file.read()

	json_file.close()

	model =model_from_json(model_json)

	model.load_weights('model/mnist.h5')

	prediction = model.predict(img)

	response=np.array_str(np.argmax(prediction,axis=1))


	return response




@app.route('/')
def index():
	return render_template("index.html")










def main():
	app.run(debug=True)


if __name__=="__main__":
	main()



