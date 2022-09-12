FROM python:3.9.13-slim

RUN pip install flask>=2.2.2
RUN pip install mlflow>=1.28.0
RUN pip install pandas>=1.4.3
RUN pip install gunicorn>=20.1.0

COPY [ "predict.py", "./" ]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]

