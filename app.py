import os

from flask import Flask, request, abort

from model import My_Classifier_Model

app = Flask(__name__)

DATA_PATH = 'data'


@app.route('/llm_predict', methods=['POST'])
def llm_predict():
    try:
        data = request.get_json()
        dataset_path = data['dataset_path']
        model.inference_llm(dataset_path)
        return "Model predicted successfully!"
    except Exception as e:
        return abort(500, str(e))


@app.route('/tfidf_predict', methods=['POST'])
def tfidf_predict():
    try:
        data = request.get_json()
        dataset_path = data['dataset_path']
        model.inference_tfidf(dataset_path)
        return "Model predicted successfully!"
    except Exception as e:
        return abort(500, str(e))


if __name__ == '__main__':
    os.makedirs(DATA_PATH, exist_ok=True)

    model = My_Classifier_Model()

    app.run(host='0.0.0.0')
