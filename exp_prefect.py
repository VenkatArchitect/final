import numpy as np
import pandas as pd
import boto3
import mlflow
from mlflow.entities import ViewType
from io import StringIO

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from prefect import task, flow
from prefect.task_runners import SequentialTaskRunner

BUCKET_NAME = 'tf-dataset-bucket'
DATA_FILE_NAME = 'iris-data.csv'
TRACKING_URI = "http://127.0.0.1:5000"

def get_experiment_id():
    return("Experiment In The Cloud")

def initialize_mlflow ():
    mlflow.autolog(disable=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(get_experiment_id())
    mlflow_client = mlflow.MlflowClient()
    return mlflow_client

def new_prediction_scores(param_grids, clf, x_train, y_train, cvs):
    grid = GridSearchCV(clf, param_grids, scoring="accuracy", cv=cvs)
    grid.fit(x_train, y_train)
    clf = grid.best_estimator_
    return(clf, cross_val_score(clf, x_train, y_train, cv=cvs, scoring="accuracy"))

def get_dataset_from_s3(bucket, filename):
    client = boto3.client('s3')
    csv_obj = client.get_object(Bucket=bucket, Key=filename)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_string), header=None)
    return (df)

def prepare_data():
    data = get_dataset_from_s3(BUCKET_NAME, DATA_FILE_NAME)
    data = data.sample(frac=1, random_state=99)
    data.columns = ["slen","swid","plen","pwid","class"]
    data['Id'] = data.index
    data.set_index("Id", inplace=True)

    features = ["slen","swid","plen","pwid"]

    x = data.loc[:, features]
    y = data.loc[:, ['class']]

    train = data[0:75]
    test = data[75:150]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size = .50)

    x_train = train.drop("class", axis=1)
    y_train = train["class"].copy()

    x_test = test.drop("class", axis=1)
    y_test = test["class"].copy()

    proc_data = (x_train, y_train, x_test, y_test, train, test)

    return(proc_data)


def run_experiment (proc_data, algorithm, cvs, fe, ht):

    x_train, y_train, x_test, y_test, train, test = proc_data

    scores = cross_val_score(algorithm, x_train, y_train, cv=cvs, scoring="accuracy")

    if (fe != 0): # feature engineering check

        train["relations_length"] = train["plen"]**fe / train["slen"]
        train["relations_width"] = train["pwid"]**fe / train["swid"]

        test["relations_length"] = test["plen"]**fe / test["slen"]
        test["relations_width"] = test["pwid"]**fe / test["swid"]

        x_train_new = train.drop("class", axis=1)
        x_test_new = test.drop("class", axis=1)
    else:
        x_train_new = x_train
        x_test_new = x_test

    scores_new = ''
    alg_specific_params = ''

    if (ht == True):
        if (type(algorithm) == type(RandomForestClassifier())):
            rf_param_grids = {
            "n_estimators": [10, 12, 15, 20],
            "max_depth": [15, 20, 30, 40],
            "max_leaf_nodes": [5, 8, 10, 20, 30]
            }
            alg_specific_params, scores_new = new_prediction_scores(rf_param_grids, algorithm, x_train_new, y_train, cvs)
        elif (type(algorithm) == type(KNeighborsClassifier())):

            knn_param_grids = {
            "n_neighbors": np.arange(1,11),
            "weights": ["uniform", "distance"],
            "leaf_size": [20, 30, 40]
            }

            alg_specific_params, scores_new = new_prediction_scores(knn_param_grids, algorithm, x_train_new, y_train, cvs)
        elif (type(algorithm) == type(SVC())):
            svm_param_grids = {
            "C": [1, 10, 30, 100],
            "kernel": ["rbf", "linear", "sigmoid"],
            "gamma": ["auto", "scale"],
            "decision_function_shape": ["ovo", "ovr"]
            }
            alg_specific_params, scores_new = new_prediction_scores(svm_param_grids, algorithm, x_train_new, y_train, cvs)
        elif (type(algorithm) == type(LogisticRegression())):
            logreg_param_grids = {
            "C": [1.0, 3.0, 10.0, 50.0],
            "max_iter": [50, 100, 200, 1000],
            "multi_class": ["auto", "ovr", "multinomial"],
            }
            alg_specific_params, scores_new = new_prediction_scores(logreg_param_grids, algorithm, x_train_new, y_train, cvs)
        else:
            print("Wrong Algorithm! Aborting!")
            exit(1)
    else:
        alg_specific_params = "Hyperparameter tuning not applied"
        scores_new = scores

    # Now let's look at the prediction accuracy of the algorithms after the feature engineering and
    # hyperparameter tuning on the test data

    algorithm.fit(x_train_new, y_train)
    preds_new = algorithm.predict(x_test_new)

    with mlflow.start_run():
        metrics = {"Wrongpredictions": (list(preds_new == y_test.values).count(False)),
                   "Score": float(sum(scores_new) / cvs),
                   "FE": fe, "CVs": cvs, "HT": ht}
        params = {"Alg.": algorithm, "Alg. Params": alg_specific_params}
        mlflow.log_metrics(metrics)
        mlflow.log_params(params)
        mlflow.sklearn.log_model(algorithm, artifact_path="model")

@task
def run_experiments ():

    x_train, y_train, x_test, y_test, train, test = prepare_data()

    proc_data = (x_train, y_train, x_test, y_test, train, test)

    #with 4 cross-validations
    run_experiment (proc_data, KNeighborsClassifier(), 4, 0, False)
    run_experiment (proc_data, RandomForestClassifier(), 4, 0, False)
    run_experiment (proc_data, SVC(), 4, 0, False)
    run_experiment (proc_data, LogisticRegression(), 4, 0, False)

    run_experiment (proc_data, KNeighborsClassifier(), 4, 2, False)
    run_experiment (proc_data, RandomForestClassifier(), 4, 2, False)
    run_experiment (proc_data, SVC(), 4, 2, False)
    run_experiment (proc_data, LogisticRegression(), 4, 2, False)

    run_experiment (proc_data, KNeighborsClassifier(), 4, 2, True)
    run_experiment (proc_data, RandomForestClassifier(), 4, 2, True)
    run_experiment (proc_data, SVC(), 4, 2, True)
    run_experiment (proc_data, LogisticRegression(), 4, 2, True)

    run_experiment (proc_data, KNeighborsClassifier(), 4, 3, False)
    run_experiment (proc_data, RandomForestClassifier(), 4, 3, False)
    run_experiment (proc_data, SVC(), 4, 3, False)
    run_experiment (proc_data, LogisticRegression(), 4, 3, False)

    run_experiment (proc_data, KNeighborsClassifier(), 4, 3, True)
    run_experiment (proc_data, RandomForestClassifier(), 4, 3, True)
    run_experiment (proc_data, SVC(), 4, 3, True)
    run_experiment (proc_data, LogisticRegression(), 4, 3, True)

    # with 10 cross-validations
    run_experiment (proc_data, KNeighborsClassifier(), 10, 0, False)
    run_experiment (proc_data, RandomForestClassifier(), 10, 0, False)
    run_experiment (proc_data, SVC(), 10, 0, False)
    run_experiment (proc_data, LogisticRegression(), 10, 0, False)

    run_experiment (proc_data, KNeighborsClassifier(), 10, 2, False)
    run_experiment (proc_data, RandomForestClassifier(), 10, 2, False)
    run_experiment (proc_data, SVC(), 10, 2, False)
    run_experiment (proc_data, LogisticRegression(), 10, 2, False)

    run_experiment (proc_data, KNeighborsClassifier(), 10, 2, True)
    run_experiment (proc_data, RandomForestClassifier(), 10, 2, True)
    run_experiment (proc_data, SVC(), 10, 2, True)
    run_experiment (proc_data, LogisticRegression(), 10, 2, True)

    run_experiment (proc_data, KNeighborsClassifier(), 10, 3, False)
    run_experiment (proc_data, RandomForestClassifier(), 10, 3, False)
    run_experiment (proc_data, SVC(), 10, 3, False)
    run_experiment (proc_data, LogisticRegression(), 10, 3, False)

    run_experiment (proc_data, KNeighborsClassifier(), 10, 3, True)
    run_experiment (proc_data, RandomForestClassifier(), 10, 3, True)
    run_experiment (proc_data, SVC(), 10, 3, True)
    run_experiment (proc_data, LogisticRegression(), 10, 3, True)

    # with 9 cross-validations
    run_experiment (proc_data, KNeighborsClassifier(), 9, 0, False)
    run_experiment (proc_data, RandomForestClassifier(), 9, 0, False)
    run_experiment (proc_data, SVC(), 9, 0, False)
    run_experiment (proc_data, LogisticRegression(), 9, 0, False)

    run_experiment (proc_data, KNeighborsClassifier(), 9, 2, False)
    run_experiment (proc_data, RandomForestClassifier(), 9, 2, False)
    run_experiment (proc_data, SVC(), 9, 2, False)
    run_experiment (proc_data, LogisticRegression(), 9, 2, False)

    run_experiment (proc_data, KNeighborsClassifier(), 9, 2, True)
    run_experiment (proc_data, RandomForestClassifier(), 9, 2, True)
    run_experiment (proc_data, SVC(), 9, 2, True)
    run_experiment (proc_data, LogisticRegression(), 9, 2, True)

    run_experiment (proc_data, KNeighborsClassifier(), 9, 3, False)
    run_experiment (proc_data, RandomForestClassifier(), 9, 3, False)
    run_experiment (proc_data, SVC(), 9, 3, False)
    run_experiment (proc_data, LogisticRegression(), 9, 3, False)

    run_experiment (proc_data, KNeighborsClassifier(), 9, 3, True)
    run_experiment (proc_data, RandomForestClassifier(), 9, 3, True)
    run_experiment (proc_data, SVC(), 9, 3, True)
    run_experiment (proc_data, LogisticRegression(), 9, 3, True)


def get_best_model_version(exp_id, fclient):

    experiment = fclient.get_experiment_by_name(exp_id)

    experiment_instance = experiment.experiment_id

    run_list = fclient.search_runs(experiment_instance,
                                   filter_string="metric.Wrongpredictions < 2", run_view_type=ViewType.ALL,
                                   max_results=1,
                                   order_by=["metric.Score DESC"])

    if len(run_list) == 0:
        run_list = fclient.search_runs(experiment_instance,
                                       filter_string="", run_view_type=ViewType.ALL,
                                       max_results=1,
                                       order_by=["metric.Score DESC"])

    run = run_list[0]

    version = int(mlflow.register_model(model_uri='runs:/' + run.info.run_id + '/model', name="The model with the best performance").version)

    return (version)

@task
def deploy_best_model(model_version, fclient):

    # Now that we have selected the best model, we will move it to staging and production
    fclient.transition_model_version_stage(name="The model with the best performance", version=str(model_version),
                                          stage="Staging")
    fclient.transition_model_version_stage(name="The model with the best performance", version=str(model_version),
                                          stage="Production")

@flow(task_runner=SequentialTaskRunner())
def workflow_orchestration():
    exp_id = get_experiment_id()
    fclient = initialize_mlflow()
    run_experiments()
    deploy_best_model(get_best_model_version(exp_id, fclient), fclient)

if __name__ == '__main__':
    workflow_orchestration()
