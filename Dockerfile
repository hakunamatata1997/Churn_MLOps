FROM jupyter/scipy-notebook

ENV http_proxy http://172.30.10.43:3128
ENV https_proxy http://172.30.10.43:3128

RUN pip install joblib


USER root

RUN apt-get update && apt-get install -y jq

RUN mkdir /home/coea

RUN cd /home/coea

RUN mkdir model raw_data processed_data results 


ENV RAW_DATA_DIR=/home/coea/raw_data
ENV PROCESSED_DATA_DIR=/home/coea/processed_data
ENV MODEL_DIR=/home/coea/model
ENV RESULTS_DIR=/home/coea/results
ENV RAW_DATA_FILE=Churn_Prediction.csv


COPY Churn_Prediction.csv ./raw_data/Churn_Prediction.csv
COPY preprocessing.py ./preprocessing.py
COPY train.py ./train.py
COPY test.py ./test.py
COPY Deployment .


