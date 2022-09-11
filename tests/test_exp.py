import mlflow
from exp import get_experiment_id

#
# test_initialize_flow is a UNIT test. It asserts for
# experiment's name, tags, lifecycle stage, and the
# mlflow client being available. It is an important
# function before mlflow experiments are started.
#
def test_initialize_mlflow():

    mlflow.set_experiment(get_experiment_id())

    experiment = mlflow.get_experiment_by_name(get_experiment_id())
    assert(experiment.name == get_experiment_id())
    assert(experiment.tags == {})
    assert(experiment.lifecycle_stage == 'active')
    assert(mlflow.MlflowClient() != None)
