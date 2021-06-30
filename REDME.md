# Disaster Response Pipeline Project

### Description
During a disaster the need for fast and accurate response is essential to save lifes.
A pain during such unfotunate incidences is the need to process high volume of information passed through social media, 
filter relevant information and delegate the information to relevant teams.

This application is a dummy attempt to create a solution for the above. 

The empty field accepts a text message, classifies it as relevant to a disaster or not, and outputs to the user the message tag.   

### Dependencies
- See requirements.txt file for dependencies
- Python version: 3.6.13

### Instructions:
Go to: https://dog-class-gillan.herokuapp.com/

### File Descriptions

### Data flow overview
This project performs the following steps:
ETL pipeline:
- Read csv files as input
- Transform and clean data into useable format
- Load tables to SQLite DB

NLP ML pipeline
- Load input data from SQLite DB
- Tokenize text messages and use Tfidf transformation
- Split into train and test datasets 
- Apply MultiOutputClassifier on all classification targets using GradientBoostingClassifier  
- Optimize parameters using GridsearchCV
- Evaluate model classifiaction score
- Save model as a pickel file for future use

Finally, the application is called using Flask framework.

### Image of the web app:
![image](https://user-images.githubusercontent.com/69136925/123932307-fcd56800-d999-11eb-8f94-e9a26c5bd146.png)

![image](https://user-images.githubusercontent.com/69136925/123932512-33ab7e00-d99a-11eb-947e-3945da69a1e9.png)

### Licensing, Acknowledgements

