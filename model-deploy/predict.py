from flask import Flask
import mlflow
import pandas as pd
from mlflow.entities import ViewType

MLFLOW_SERVER_URI = "http://43.205.178.191:5000"
RUN_ID='110b8ed7fbc7462faae213e7ab705047'
app = Flask('predict')


def get_experiment_name_deployment():
    return("Experiment In The Cloud")
def flask_initialize_mlflow ():
    mlflow.autolog(disable=True)
    mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
    mlflow.set_experiment(get_experiment_name_deployment())
    mlflow_client = mlflow.MlflowClient()
    return mlflow_client

def get_model(model_uri):
    logged_model = f'{model_uri}'
    model = mlflow.pyfunc.load_model(logged_model)
    return (model)


@app.route('/predict/<slen>/<swid>/<plen>/<pwid>', methods=['GET'])
def predict(slen, swid, plen, pwid):
    # args = sys.argv[1:]
    # slen = args[0]
    # swid = args[1]
    # plen = args[2]
    # pwid = args[3]

    run = mlflow.get_run(RUN_ID)

    model = get_model('runs:/' + RUN_ID + '/model')

    x_data = pd.DataFrame(columns=['slen', 'swid', 'plen', 'pwid'])

    x_data.loc[0] = [float(slen), float(swid), float(plen), float(pwid)]

    fe = run.data.metrics['FE']

    x_data["relations_length"] = x_data["plen"] ** fe / x_data["slen"]
    x_data["relations_width"] = x_data["pwid"] ** fe / x_data["swid"]

    y_pred = model.predict(x_data)

    return (str(y_pred))


if __name__ == "__main__":

    flask_initialize_mlflow()

    app.run(debug=True, host='0.0.0.0', port=9696)
