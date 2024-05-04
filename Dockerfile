FROM python:3.10-slim

RUN apt-get update && apt-get install -y curl

WORKDIR app

COPY . .

RUN pip install detectaitext-0.1.0-py3-none-any.whl

EXPOSE 5000

CMD [ "python", "app.py" ]