# Maslov Victor 972201

# "How To"

## Without Docker
1. Download repository
2. Download archive https://drive.google.com/drive/folders/11abMJXk18IPzLK9jrphRdUgVBTmXoagP and unpack into the repository
3. Install spaceshiptitanic-0.1.0-py3-none-any.whl by pip or conda
4. Run command 'python app.py' from root repository folder

### Predict using CLI with LLM pipeline
Run command 'python model.py predict --dataset=/path/to/dataset --pipeline=llm' from root repository folder

### Predict using CLI with TF-IDF pipeline
Run command 'python model.py predict --dataset=/path/to/dataset --pipeline=tfidf' from root repository folder

### Predict using API with LLM pipeline
Send request POST to http://127.0.0.1:5000/predict_llm with body
{
    "dataset_path": "/path/to/dataset"
}

### Predict using API with TF-IDF pipeline
Send request POST to http://127.0.0.1:5000/predict_tfidf with body
{
    "dataset_path": "/path/to/dataset"
}

## With Docker
1. Download repository
2. Download archive https://drive.google.com/drive/folders/11abMJXk18IPzLK9jrphRdUgVBTmXoagP and unpack into the repository
3. Run command 'docker compose up -d' from root repo folder

### Predict using CLI with LLM pipeline
Run command '../usr/local/bin/python model.py predict --dataset=/path/to/dataset --pipeline=llm' from 'app' folder in container

### Predict using CLI with TF-IDF pipeline
Run command '../usr/local/bin/python model.py predict --dataset=/path/to/dataset --pipeline=tfidf' from 'app' folder in container

### Predict using API with LLM pipeline
Send request POST to http://127.0.0.1:5000/predict_llm in container with body
{
    "dataset_path": "/path/to/dataset/from/app/folder"
}

### Predict using API with TF-IDF pipeline
Send request POST to http://127.0.0.1:5000/predict_tfidf in container with body
{
    "dataset_path": "/path/to/dataset/from/app/folder"
}

## Utilizided resources
https://education.yandex.ru/handbook/ml https://arxiv.org/pdf/1806.06407 https://scikit-learn.org/stable/ https://docs.docker.com/reference/cli/docker/image/build/ https://flask.palletsprojects.com/en/3.0.x/ https://arxiv.org/pdf/2006.03654 https://onnx.ai/onnx/index.html