# Dog breed classification project

### Description
This projcect uses convolutional neural network and transfer learning based on Resnet50 model to perform dog breed classification.
The project is uploaded to the web, and allows an easy interface to paste an image URL and perform the breed prediction.
The classification output:
- Image is human or dog
- What dog breed is claddified (regardless if image is human or dog)
- What confidence the algorithm has in the dog breed classification

### Dependencies
- See requirements.txt file for dependencies
- Python version: 3.6.13

### Instructions:
Go to: https://dog-class-gillan.herokuapp.com/

### Data flow overview
Dog or human classification was performed by OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images.
Then to obtain the dog beed classification, the image is preprocessed and passed through a Resnet50 network.
All stages were implamented using the Keras API, specifically:
keras.applications.resnet50:
  - ResNet50
  - preprocess_input

Once all the image features were extracted, the data is passed through a CNN:
![image](https://user-images.githubusercontent.com/69136925/123932307-fcd56800-d999-11eb-8f94-e9a26c5bd146.png)

These network hyper-parameters were selected by Using the scikit-learn GridSearchCV and KerasClassifier
wrappers to optimize model hyperparameters

### Image of the web app:
![image](https://user-images.githubusercontent.com/69136925/123932512-33ab7e00-d99a-11eb-947e-3945da69a1e9.png)
