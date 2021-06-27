import keras 
from keras.callbacks import ModelCheckpoint  
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras import utils as np_utils
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from keras.models import Sequential, Model
from sklearn.datasets import load_files      
from sklearn.model_selection import train_test_split
import numpy as np
from glob import glob
import random
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras import backend as K


from io import BytesIO
from PIL import Image, ImageFile
import requests

import os

def extract_Resnet50(tensor):
    K.clear_session()
    return ResNet50(weights='imagenet', include_top=False,pooling=None).predict(preprocess_input(tensor))

def path_to_tensor(img_path):
      # loads RGB image as PIL.Image.Image type
      img = image.load_img(img_path, target_size=(224, 224))
      # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
      x = image.img_to_array(img)
      # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
      return np.expand_dims(x, axis=0)

def dog_class_model(path_to_weights='\\wrangling_scripts\\weights.best.resnet50.hdf5'):

    model = Sequential()
    model.add(Flatten(input_shape=(1,1,2048)))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(133, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.load_weights(os.getcwd() + path_to_weights)
    return model



def loadImage(url):
    response = requests.get(url)
    img_bytes = BytesIO(response.content)
    img = Image.open(img_bytes)
    img = img.resize((224,224), Image.NEAREST)
    img = np.array(img).astype('float32')
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img
  
def predict_dog_breed(img_path):

    # extract bottleneck features
    img_res50_bottleneck_features = extract_Resnet50(loadImage(img_path))
    K.clear_session()
    # instantiate classification model 
    classification_model=dog_class_model()

    # obtain predicted vector
    predicted_vector = classification_model.predict(img_res50_bottleneck_features)
    K.clear_session()
    # return dog breed that is predicted by the model
    dog_names = list(np.load(os.getcwd()+'\\wrangling_scripts\\dog_names.npy'))
    print(f"confidence: {predicted_vector.max()}%")

    probability = predicted_vector.max()
    breed_name = dog_names[np.argmax(predicted_vector)].split('.')[1]
    return breed_name, probability

# def return_dog_breed(URL):
#     # this function accepts a url to an image
#     # returns the dog breed classification and certainty
#     # INPUT:
#     #     URL (string): url to image
#     # OUTPUT:
#     #     breed (string): what dog breed is shown in the image
#     #     detection_probability (string): certainty prbability 
#     #     the algorithm has een able to detect the correct breed

#     resnet50_model = dog_class_model()

#     breed, detection_probability = predict_dog_breed(URL,resnet50_model)

#     return breed, detection_probability

