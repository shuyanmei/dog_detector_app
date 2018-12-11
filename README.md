# dog_breed_detector

## Project Overview
The goal of this project is to build a web app which can detect the human or dog face and corresponding dog breed when you upload an image. If a dog is detected, then the app will identify it as a dog and its breed. If human is detected, it will identify it as human and its most resembling breed. The input training dataset is provided by Udacity. There are 8351 dog images with 133 different dog categories.

## Problem Statement
To detect dog breed in less time without sacrificing accuracy, I used transfer learning to create a Convolutional Neural Network(CNN) using Inception bottleneck features. The model use RMSprop algorithm for gradient descent optimization. The loss function is categorical cross entropy.

## Metrics
I use 6680 images for training, 835 images for validation and 836 for testing . The metric to evaluate the model is prediction accuracy. The model achieved above 81\% accuracy on test dataset. 

## Analysis and model development
The details of analysis and model development are in the 'dog_app.ipynb'  jupyter notebook.

This repository contains following files:
  ```
dog_app.ipynb : The codes contain all the development process of the model.
requirement.txt : Python packages need to be installed
run.py : The python script to run the app
saved_model/ : Pre-trained Inception model and its weight.
upload/ : Images to upload in the web app to test
templates/ : Two html pages 
```

To run this app, first you need to clone the repository to your folder. 
  ```
  git clone https://github.com/shuyanmei/dog_detector_app.git
```

Start the virtual environment in python.
```  
virtualenv -p ~/bin/python3.6 app
source app/bin/activate
```

Install all the packages in the requirements.txt.
  ```
  pip install -r requirement.txt
```

The python packages required are:
```  
keras
Flask
numpy
scipy
pillow
h5py
werkzeug
virtualenv
tqdm
opencv-python
Theano
sklearn
warnings
```


Start the app
 ```
  python run.py
```

Open a browser and access the web from 0.0.0.0:5001

Upload an image from upload folder, wait a while for the results

