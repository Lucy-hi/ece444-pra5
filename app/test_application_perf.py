import pytest
import json
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from application import application

EC2_URL = "http://lucy-ece444-pra5-env.eba-zbcjgkt6.ca-central-1.elasticbeanstalk.com/predict"
test_case_performance = "test_case_performance.csv"
test_cases = [
    ("true_news_1", {"text": "Ontario draw for Taylor Swift tickets will also see winner rake in more than $100K"}),
    ("true_news_2", {"text": "PM Trudeau emerged from a Liberal caucus meeting noting the strength of the party and, the Bank of Canada dropped the key interest rate."}),
    ("fake_news_1", {"text": "Disney World was battling the Florida government in court to get a resort exemption, which would allow anyone 18 and older to drink on property."}),
    ("fake_news_2", {"text": "A new CDC study found the majority of those infected with COVID-19 ‘always’ wore Masks."}),
]

@pytest.fixture
def client():
    with application.app_context():
        yield application.test_client()  # tests run here

def write_to_csv(csv_path, timestamps):
    with open(csv_path, 'w+', newline='') as file:
        wr = csv.writer(file)
        wr.writerow(["Case", "Request#", "Start Time", "End Time", "Response Time"])
        for test_case, perf in timestamps.items():
            for idx, (start, end, response) in enumerate(perf):
                wr.writerow([test_case, idx, start, end, response])

def plot_performance(csv_path):
    df = pd.read_csv(csv_path)

    average_performance = df.groupby("Case")["Response Time"].mean()
    print("\nAverage Performance per Test Case")
    print(average_performance)

    plt.figure(figsize=(10,6))
    avg_perf = sns.boxplot(x="Case", y="Response Time", data=df)
    for idx, avg in enumerate(average_performance):
        avg_perf.text(idx, avg, f'Average: {avg:.2f}', horizontalalignment='center', verticalalignment='center',color='black')

    plt.title("Performance for Each Test Case")
    plt.xlabel("Test Case")
    plt.ylabel("Response Time (ms)")

    plt.savefig("performance_plot.png")
    plt.show()

    
def test_model_perf(client):
    timestamps = {}
    for test_case, input_news in test_cases:
        timestamps[test_case]=[]
        for i in range(100):
            start_time = time.time()
            prediction = client.post(
                EC2_URL,
                data=json.dumps(input_news),
                content_type='application/json'
            )
            end_time = time.time()
            response_time = (end_time - start_time)*1000
            timestamps[test_case].append((start_time, end_time, response_time))
            predict_data = json.loads(prediction.data)
            assert predict_data["prediction"] in ('REAL','FAKE')

    write_to_csv(test_case_performance, timestamps)
    plot_performance(test_case_performance)


   