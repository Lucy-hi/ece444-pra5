import pytest
import json
from pathlib import Path

from application import application

EC2_URL = "http://lucy-ece444-pra5-env.eba-zbcjgkt6.ca-central-1.elasticbeanstalk.com/predict"

@pytest.fixture
def client():
    with application.app_context():
        yield application.test_client()  # tests run here

def test_true_news_1(client):
    input_news = {
        "text": "Ontario draw for Taylor Swift tickets will also see winner rake in more than $100K"
    }

    prediction = client.post(
        EC2_URL,
        data=json.dumps(input_news),
        content_type='application/json'
    )

    predict_data = json.loads(prediction.data)
    print(f"Prediction for {predict_data}, the label is REAL")
    assert predict_data["prediction"] == 'REAL'

def test_true_news_2(client):
    input_news = {
        "text": "PM Trudeau emerged from a Liberal caucus meeting noting the strength of the party and, the Bank of Canada dropped the key interest rate."
    }

    prediction = client.post(
        EC2_URL,
        data=json.dumps(input_news),
        content_type='application/json'
    )

    predict_data = json.loads(prediction.data)
    print(f"Prediction for {predict_data}, the label is REAL")
    assert predict_data["prediction"] == 'REAL'
    
def test_fake_news_1(client):
    input_news = {
        "text": "Disney World was battling the Florida government in court to get a resort exemption, which would allow anyone 18 and older to drink on property."
    }

    prediction = client.post(
        EC2_URL,
        data=json.dumps(input_news),
        content_type='application/json'
    )

    predict_data = json.loads(prediction.data)
    print(f"Prediction for {predict_data}, the label is FAKE")
    assert predict_data["prediction"] == 'FAKE'

def test_fake_news_2(client):
    input_news = {
        "text": "A new CDC study found the majority of those infected with COVID-19 ‘always’ wore Masks."
    }

    prediction = client.post(
        EC2_URL,
        data=json.dumps(input_news),
        content_type='application/json'
    )

    predict_data = json.loads(prediction.data)
    print(f"Prediction for {predict_data}, the label is FAKE")
    assert predict_data["prediction"] == 'FAKE'
