from myapp import app
from flask import render_template, request
# from wrangling_scripts.wrangle_data import return_figures
import wrangling_scripts.wrangle_data 
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')




@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    # use model to predict classification for query


    # classification_labels = model.predict([query])[0]
    # classification_results = dict(zip(df.columns[4:], classification_labels))
    breed, probability = wrangling_scripts.wrangle_data.predict_dog_breed(query)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=breed,
        classification_probability=probability
    )