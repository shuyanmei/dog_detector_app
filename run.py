import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras import backend as K
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import cv2
from flask import Flask, request, redirect,url_for
from flask import send_from_directory, render_template
from werkzeug.utils import secure_filename
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def path_to_tensor(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	return np.expand_dims(x, axis=0)

def face_detector(img_path):
	img = cv2.imread(img_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier('saved_model/haarcascade_frontalface_alt.xml')
	faces = face_cascade.detectMultiScale(gray)
	return len(faces) > 0

def load_dog_names():
	''''
	load dog names
	'''
	filename = 'saved_model/dog_names'
	dog_names = pickle.load(open(filename, 'rb'))
	return dog_names

def load_model():
	''''
	load pre-trained model
	'''
	filename = 'saved_model/model_Inception'
	model_Inception = pickle.load(open(filename, 'rb'))
	return model_Inception

def load_weight():
	''''
	load pre-trained model weight
	'''
	load_model().load_weights('saved_model/weights.best.Inception.hdf5')

def extract_InceptionV3(tensor):
	return InceptionV3(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def predict_breed_Inception(img_path):
	'''
		predict the breed with pre-trained model
	'''
	bottleneck_feature = extract_InceptionV3(path_to_tensor(img_path))
	predicted_vector = load_model().predict(bottleneck_feature)
	name_predictions = load_dog_names()[np.argmax(predicted_vector)]
	return name_predictions

from keras.applications.resnet50 import preprocess_input, decode_predictions


def ResNet50_predict_labels(img_path):
	img = preprocess_input(path_to_tensor(img_path))
	ResNet50_model = ResNet50(weights='imagenet')
	return np.argmax(ResNet50_model.predict(img))


def dog_detector(img_path):
	prediction = ResNet50_predict_labels(img_path)
	return ((prediction <= 268) & (prediction >= 151))

def human_or_dog(img_path):    
	breed = predict_breed_Inception(img_path)
	if (not dog_detector(img_path)) and (not face_detector(img_path)): 
		result =  "This is neither a dog or human!"
	if dog_detector(img_path):
		result = "This is a dog!. The breed is {}".format(breed)
	if face_detector(img_path):
		result = "This is human! You look like a: {}".format(breed)
	return result



@app.route('/upload/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'] ,filename)


@app.route('/', methods=['GET', 'POST'])
def upload_picture():
	if request.method == 'POST':
		file = request.files['file']
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			fileurl = url_for('uploaded_file', filename=filename)
			prediction =human_or_dog(filepath)
			return render_template('main.html',breed = prediction, fileurl =fileurl)			
	return render_template('base.html')


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5001, debug=True)