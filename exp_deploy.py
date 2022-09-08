from prefect.deployments import Deployment
from exp import workflow_orchestration

deployment = Deployment.build_from_flow(flow=workflow_orchestration,\
                                        name="Iris Workflow Orchestration", version="1", tags=["Monthly Deployment"],\
                                        schedule={'interval': 600, 'timezone': "Asia/Kolkata"},
                                        work_queue_name="iris-monthly-work-queue")

if __name__=='__main__':
    deployment.apply()


