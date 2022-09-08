from prefect.deployments import Deployment
from exp_prefect import workflow_orchestration

deployment = Deployment.build_from_flow(flow=workflow_orchestration,\
                                        name="Iris Workflow Orchestration", version="1", tags=["Hourly Deployment"],\
                                        schedule={'interval': 600, 'timezone': "Asia/Kolkata"},
                                        work_queue_name="iris-hour-work-queue")

if __name__=='__main__':
    deployment.apply()


