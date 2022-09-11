import mlflow
from exp import get_experiment_id, get_dataset_from_s3, prepare_data


#
# test_initialize_flow is a UNIT test. It asserts for
# experiment's name, tags, lifecycle stage, and the
# mlflow client being available. It is an important
# function before mlflow experiments are started.
#
def test_initialize_mlflow():
    mlflow.set_experiment(get_experiment_id())

    experiment = mlflow.get_experiment_by_name(get_experiment_id())
    assert (experiment.name == get_experiment_id())
    assert (experiment.tags == {})
    assert (experiment.lifecycle_stage == 'active')
    assert (mlflow.MlflowClient() != None)


BUCKET_NAME = 'tf-dataset-bucket'
DATA_FILE_NAME = 'iris-data.csv'


#
# test_get_dataset_from_s3 is a UNIT test. It asserts
# that the size of the dataset read from s3 is non-zero.
def test_get_dataset_from_s3():
    assert (len(get_dataset_from_s3(BUCKET_NAME, DATA_FILE_NAME)) != 0)


def test_prepare_data():
    (x_train, y_train, x_test, y_test, train, test) = prepare_data()

    xtrain_columns_list = x_train.columns.values.tolist()
    assert ("class" not in xtrain_columns_list)

    ytrain_columns_list = y_train.to_frame().columns.values.tolist()
    assert ("class" in ytrain_columns_list)
    assert (len(ytrain_columns_list) == 1)