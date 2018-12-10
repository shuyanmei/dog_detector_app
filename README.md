# dog_breed_detector


The goal of this project is to build a web app which can detect the human or dog face and corresponding dog breed when you upload an image. 

The algorithm behind this is Convolutional Neural Network(CNN).

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

