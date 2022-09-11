from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule

from exp import workflow_orchestration

deployment = Deployment.build_from_flow(flow=workflow_orchestration,\
                                        name="Iris Workflow Orchestration", version="2", tags=["Monthly Deployment"],\
                                        schedule=(CronSchedule(cron="0 0 1 * *", timezone="America/Chicago")),
                                        work_queue_name="iris-monthly-work-queue")


if __name__=='__main__':
    deployment.apply()