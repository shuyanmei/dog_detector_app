# dog detector app

## Project Overview
The goal of this project is to build a web app which can detect the human or dog face and corresponding dog breed when you upload an image. If a dog is detected, then the app will identify it as a dog and its breed. If human is detected, it will identify it as human and its most resembling breed. The input training dataset is provided by Udacity. There are 8351 dog images with 133 different dog categories.

## Problem Statement
To detect dog breed in less time without sacrificing accuracy, I used transfer learning to create a Convolutional Neural Network(CNN) using Inception bottleneck features. The model use RMSprop algorithm for gradient descent optimization. The loss function is categorical cross entropy.

## Metrics
I use 6680 images for training, 835 images for validation and 836 for testing . The metric to evaluate the model is prediction accuracy. 

## Analysis and model development
The details of analysis and model development are in the 'dog_app.ipynb'  jupyter notebook.

## Methodology
 First, I load the image and resizes it to a square image that is  224×224 pixels. Then convert the image to an array and resize it to a 4D tensor. In this case, the returned tensor will always have shape (1,224,224,3). The final step is to convert it to a 4D tensor with number of samples.
 
 For the model fitting parts, first I used the last convolutional layer of pre-trained VGG16 model as input of my model. Then add a global average pooling layer and a fully connected layer. It achieved accuracy of 41\%.
 
 The final model I use is CNN with pre-trained Inception model. Then similarly, add global average pooling layer and a softmax activation function for the multi-classification. It improved accuracy to 81\%.
 
 ## Results
 The development of final model also follows the previous step. A validation set was used to tune the hyperparameter and evaluate the model. The size of output of first layer is 2048 and 133 for  final layer output which corresponds to 133 different dog categories. The total number of parameters is 272,517. The value of cost function did not improve from 0.60570 after 3 echos. There is 6680 training samples and batch size I use is 20, thus the number of iterations is 334.  The final model achieved above 81\% accuracy on test dataset. 

## Instructions on running the app
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

