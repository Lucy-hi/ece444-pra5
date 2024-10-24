from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import json

application = Flask(__name__)

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

@application.route("/")
def index():
    return "Your Flask App Works! V1.0"

@application.route('/predict', methods=['POST'])
def predict():
    # Get text input from the request
    input_news = request.json.get('text', '')

    if input_news:
        # Get prediction from the model
        prediction = load_model(input_news)
        return jsonify({
            "input_news": input_news,
            "prediction": prediction
        })
    else:
        return jsonify({
            "message": "No input text provided"
        }), 400

if __name__ == "__main__":
    application.run()