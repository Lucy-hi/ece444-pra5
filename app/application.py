from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json

application = Flask(__name__)

@application.route("/")
def index():
    return "Your Flask App Works! V1.0"

def load_model(input_news):
    ###### model loading #####
    loaded_model = None
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)

    vectorizer = None
    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)

    #######################
    # how to use model to predict
    prediction = loaded_model.predict(vectorizer.transform([input_news]))[0]

    # output will be 'FAKE' if fake, 'REAL' if real
    return prediction

if __name__ == "__main__":
    application.run()