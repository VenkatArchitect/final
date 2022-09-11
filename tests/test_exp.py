import mlflow
from exp import get_experiment_name, get_dataset_from_s3, prepare_data, get_best_model_version


#
# test_initialize_flow is a UNIT test. It asserts for
# experiment's name, tags, lifecycle stage, and the
# mlflow client being available. It is an important
# function before mlflow experiments are started.
#
def test_initialize_mlflow():
    mlflow.set_experiment(get_experiment_name())

    experiment = mlflow.get_experiment_by_name(get_experiment_name())
    assert (experiment.name == get_experiment_name())
    assert (experiment.tags == {})
    assert (experiment.lifecycle_stage == 'active')
    assert (mlflow.MlflowClient() != None)


BUCKET_NAME = 'tf-dataset-bucket'
DATA_FILE_NAME = 'iris-data.csv'


#
# test_get_dataset_from_s3 is a UNIT test. It asserts
# that the size of the dataset read from s3 is non-zero.
#
def test_get_dataset_from_s3():
    assert (len(get_dataset_from_s3(BUCKET_NAME, DATA_FILE_NAME)) != 0)


#
# test_prepare_data is a UNIT test. It asserts
# the shapes and numbers of various data structures
# used for preparing the data for the experiment runs.
#

def test_prepare_data():
    (x_train, y_train, x_test, y_test, train, test) = prepare_data()

    xtrain_columns_list = x_train.columns.values.tolist()
    assert ("class" not in xtrain_columns_list)

    ytrain_columns_list = y_train.to_frame().columns.values.tolist()
    assert ("class" in ytrain_columns_list)
    assert (len(ytrain_columns_list) == 1)

#
# test_get_best_model_version is an INTEGRATION test. It asserts
# that the version of the model to be moved to production is not zero.
# The version will be zero only if no model could be obtained from
# the experiment runs as a candidate to be moved to production.
# But from the experiment runs, at least one model should be available
# to be moved to production, so this assertion should never fail.
# If it fails, it means that there is something wrong in the experiment
# runs (prefect failure, mlflow failure, s3/server/rds failures, etc.
# This test checks the integration of all the functionality available
# in this project.
#


def test_get_best_model_version():
    version = get_best_model_version (get_experiment_name(), mlflow.MlflowClient())

    assert(version != 0)
