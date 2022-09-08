FROM python:3.9.13-slim

RUN pip install flask
RUN pip install mlflow
RUN pip install pandas
RUN pip install gunicorn

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

