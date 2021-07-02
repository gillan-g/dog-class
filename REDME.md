# Human Detector and Dog Breed Classifier

Udacity data scientist nano-degree capstone project 


### Project Overview:
This project is an attempt to utilize transfer lerning with convolutional neural network to tackle an interesting problem.
Provided an image, the algorithm will first classify if the image contains a human or a dog in it.
Then, it will provide an estimation to what dog breed is in the image, or what dog breed best resembles the human.
The application will accept a URL to an image and provide the sescribed out put.
Data used for this project:
- We use OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images.
OpenCV provides many pre-trained face detectors, stored as XML files on github.
https://github.com/opencv/opencv/tree/master/data/haarcascades
- For the dog detector, pre-trained ResNet-50 model with weights that have been trained on ImageNet,
http://www.image-net.org/
The Image data was provided as pre-processed bottlenck features by Udacity
This projcect uses convolutional neural network and transfer learning based on Resnet50 model to perform dog breed classification. The project is uploaded to the web, and allows an easy interface to paste an image URL and perform the breed prediction. The classification output:

-	Image is human or dog
-	What dog breed is claddified (regardless if image is human or dog)
-	What confidence the algorithm has in the dog breed classification
	
	
### The problem:
Given an image 
1. does the image contain a dog or a human and at what confidence
2. what dog breed best resembles the figure in the image and at what confidence
Here we attempt to use OpenCV pre-trained human image classifier to answer the first question.  
For the dog breed classifier, as stated above, we will implement a deep CNN utilizing transfer learning based on Resnet50 model, and adding a top layer that shall provide the fine-tuned dog breed classification result.	


### Metrics:
For both classification stages, 
For the first prediction we had only 2 classes - human vs. dog
The test dataset is well balanced (100 samples of each class - human, dog)
accuracy was used as a metric

For the second prediction we had 133 different classes - different possible dog breeds.
As this is both a multiclass dataset with imbalanced classes (as seen above, there might be up to 3X fold difference in class sizes), the F1 score metric was used with the 'weighted' seeting. From Sklearn documantation, this method is ideal for case.
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html


### EDA:
For the first classifier (human vs. dog) the default implementation within the assignment was used. No through EDA was performed besides evaluating the classifier accuracy metric on a dummy, pre classified, dataset. 

For the dog breed classification dataset, several metrics were evaluated:
- Total number of classes: 133 different dog breeds
- Per class, mean number of observations: ~50
- max number of observations per class: 77
- min number of observations per class: 26
We can conclude that a rather 'symmetric' number of observations per class is provided, however, we can argue that insufficient number of overall training datasets is available. This can be solved by data augmentation. however, due to time constraints, this approach wasn't addressed in this implementation.	
![image](https://user-images.githubusercontent.com/69136925/124169429-ea991e00-daae-11eb-891a-13d62e4267ab.png)


### Data preprocessing:

#### For OpenCV human classifier:
- images converted to grayscale using cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#### For resnet50 model:
- all images converted to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
- tensors were then pre-processed using Keras preprocess_input method	
Implamentation:
Resnet50 model was selected for the transfer learning process due to its relatively small size (98 MB) and due to experience.


### Hyper parameter tuning:
Data was split into train, validation, and test datasets:
- 6680 training dog images.
- 835 validation dog images.
- 836 test dog images. 

The model attached to the top layer of the resnet50 model was optimized using the scikit-learn GridSearchCV and KerasClassifier wrappers to optimize model hyperparameters.
Hyper-parameter test space:
- layers=[1,2]
- nodes=[50, 100, 150,200]
- dropout=[0,0.2,0.5]
- activation=['relu','sigmoid']

3 CV iterations.

The resulting optimal values, yielded 82% accuracy on the averaged cross validation score
{'activation': 'sigmoid', 'batch_size': 20, 'dropout': 0, 'epochs': 20, 'layers': 1, 'nodes': 200, 'optimizer': 'RMSprop'}

Final selected model:
![image](https://user-images.githubusercontent.com/69136925/123932307-fcd56800-d999-11eb-8f94-e9a26c5bd146.png)


### Results:
Achieved results on classifiers:
-	The overall testing accuracy for the OpenCV human classifier is 94.5% 
-	The overall testing F1 for the dog breed classifier is: 83%
Another approach that you could take would be to demonstrate that your optimized model is robust would be to perform a k-fold cross validation. In this case, you'd document how the model performs across each individual validation fold. If the validation performance is stable and doesn't fluctuate much, then you can argue that the model is robust against small perturbations in the training data.
0.800898 (0.013211)

### Conclusion and reflection
This was an awesome project!
Although seemed a bit overwhelming to begin with as this is my first experience with implementing a CNN, I have learned a lot along the way and was able to implement many concepts and see how almost magically create something that actually works.
For future improvements:
-	Add feature augmentation: as I have discussed before, an average of 50 samples is provided for each dog breed, this might be a bit to small to train a complex model with 100 different classes. Augmenting the input data might be able to add to the robustness of the model.
-	Add more training data: for the same reasons as provided above.
-	Attempt performing transfer learning wit a different base model such as VGG#, Inception and more. Looking for similar work with a background check in literature or online posts might help guide me in the right way.
-	More optimizations, I believe the hyperparameter CV grid is rather small, and many more permutations can be tested: different optimizers, different activation functions and more.


### Dependencies
- See requirements.txt file for dependencies
- Python version: 3.6.13


### Instructions:
Go to: https://dog-class-gillan.herokuapp.com/


### Image of the web app:
![image](https://user-images.githubusercontent.com/69136925/123932512-33ab7e00-d99a-11eb-947e-3945da69a1e9.png)

### Acknowledgements:
https://www.udacity.com/
