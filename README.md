    PROBLEM STATEMENT

    Iris is a flowering plant that has around 300 species.  Attempts
    have been made to identify the plant species and their flowers
    using the properties of their flowers.  In 1936, Ronald Fisher
    came up with Iris flower data set which contained the classifications
    for three Iris varieties.  A Wikipedia article about this data set
    can be found at https://en.wikipedia.org/wiki/Iris_flower_data_set

    Three flower types are included in the above-mentioned dataset, which are 

        * Iris Setosa
        * Iris Virginica
        * Iris Versicolor

    The flower types are characterized by their four features:

        * Sepal length
        * Sepal width
        * Setal length
        * Petal width

    150 samples are provided in the data set, 50 from each flower type.

    The problem that we are trying to solve is to uniquely identify the
    flower type based on its features. The dataset is provided in a csv
    file.


    SOLUTION DETAILS

    In the dataset, the flower characteristics as well as the target variable
    identified. We will split the dataset into training and testing data,
    remove the target variable from the test data, do the predictions, and
    then do the comparison of the predicted values with the actuals identified
    in the dataset.

    While trying to classify the flower types, one challenge that we face
    is that two of the properties, sepal length and sepal width, of flower
    types Iris-virginca and Irisversicolor are not clearly distinguishable
    as their values overlap. But fortunately, their petal length and petal
    width clearly distinguish between each other, and hence we will be able
    to do feature engineering to isolate those two flower types by increasing
    the importance of petal length and petal width and decreasing the
    importance of sepal length and sepal width. We will incorporate this
    feature engineering technique in the solution.

    We will be using four classification algorithms. They are:

        * K-nearest-neighbours
        * Random Forest
        * Support Vector Machine
        * Logistic Regression

    There are several combinations of factors that can influence the accuracy of
    the predictions. We will consider the following factors while running the
    experiments to see if they have an impact on the end results:

        * The algorithm used
        * Using the best algorithm hyperparameters
        * Applied Feature Engineering
        * No. of cross validation cycles used while training the model

    We will run several experiments with combinations of above factors to pick a
    model that gives the best possible accuracy.  The accuracy of
    the results will be gauged primarily by two factors:

        * No. of correct predictions
        * Highest cross validation score

    Since the dataset is small, the model that does the most number of 
    correct predictions will be given the top priority. If there is a tie,
    the algorithm that has the highest cross-validation score will be promoted
    to production.

    Since the Iris dataset is rarely updated, the experiments will be run
    once in a month on the first day of every month, and the best model will
    be selected and moved to production in MLFlow.


    SOLUTION OVERVIEW

    The solution architecture is illustrated at:
    https://github.com/VenkatArchitect/gProject/blob/master/solution-arch.jpg

	* MLFlow is used for experiment tracking and Model Registry
	* Prefect is used for workflow orchestration
      * Model is deployed as a web service
	* gunicorn is used for serving requests for prediction
      * Resources (infrastructure) are allocated on the cloud as
        Infrastructure-as-code (IaC) using terraform


    The best model that has been promoted to production in MLFlow
    is fetched using Docker-containerized code, and this code is used
    to load the model and do the prediction for a given set of data.
    

    INSTRUCTIONS FOR REPRODUCIBILITY

    The repository is available at: https://github.com/VenkatArchitect/gProject

    Get a local copy of the repository to run the project.


    RUNNING THE EXPERIMENT (exp.py)
    

       To run the experiment using various algorithms, their hyperparameters,
       and feature engineering, and select the best model for registering into
       production for the first time, and to make prefect run the experiment
       run every month through deployment, do the following:

       Deploy the necessary infrastructure through IaC (Infrastructure-As-Code)
       through Terraform.

       * The terraform file that defines the infrastructure is available in the
       project directory as "tf-iac.tf". This file uses AWS as the service
       provider. Please refer to terraform documentation for how to use it.

       Note: Have the "tf-iac.tf" in a different directory than the project
       directory, so that the project directory is not cluttered with huge-sized
       terraform files (usually created under .terraform directory under the
       current path).

       * Plan, validate, and apply the infrastructure as defined in "tf-iac.tf",
       so that the infrastructure is available for the project to get executed.

       

       Initializing MLFlow

       * For MLFlow related infrastructure created above, follow the instructions
         in the link below to make sure that the settings are done correctly.
         Note: Don't create the infrastructure; they have been created already.
         Just make sure that the settings are right:
         https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md

         Note: Make sure that the security group you select for the RDS has permissions to the ports referred
         to in the link above.

         The MLFlow Postgresql database will be hereafterwards referred
         in this document as POSTGRESQLNAME

         The MLFlow S3 bucket will be hereafterwards referred in this document as MLFLOW_AWS_S3_BUCKET.

         The database username and password will be known hereafterwards as DB_USER and DB_PASSWORD.

         The endpoint name of the RDS database will be known hereafterwards as DB_ENDPOINT.

         The public IP address of the EC2 instance hereafterwards will be referred in this document as MLFLOW_ADDRESS

       * In the local copy of exp.py of the project:
            A) Modify the IP address in the TRACKING_URI definition somewhere around line 16
               to MLFLOW_ADDRESS noted above.
            B) Specify the bucket name for BUCKET_NAME definition somewhere around line 14.
            C) Copy iris-data.csv file from your local copy to the S3 bucket.

       * The name of the experiment is initialized to "Experiment In The Cloud" in this project. You will
         find an experiment created in MLFlow with this name.

       * Launch the MLFlow AWS EC2 instance (hereafterwards known as MLFLOW_AWS_EC2) in AWS
        
       * Copy mlflow-server-requirements.txt from project local repository to MLFLOW_AWS_EC2 home directory (usually /home/ec2) using SCP or SFTP or simply
         opening a file in MLFLOW_AWS_EC2 like /home/ec2/mlflow-server-requirements.txt and copying the contents of the file to that file.

       * Run 'pip3 install -r mlflow-server-requirements.txt'

         Note: In the latest versions of Python, pip3 is just pip. Check what works in your AWS EC2 instance.

       * Make sure that you have your AWS credentials specified in this server by running 'aws configure'.
     
       * Start the MLFlow server in MLFLOW_AWS_EC2 instance with the following command after making sure that the security groups associated with the 
         MLFLOW_AWS_EC2 and POSTGRESQLNAME are configured to talk to each other through port 5432 (PostGre):
         
         mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/POSTGRESQLNAME --default-artifact-root s3://MLFLOW_AWS_S3_BUCKET_NAME

       * Locate a machine that can run as your experiment server. We will call this machine as EXP_SERVER hereafterwards.

         IMPORTANT NOTE: THE MACHINE SHOULD HAVE THE FOLLOWING VERSION DEPENDENCIES SATISFIED.

         1) Python should be >= 3.9.6, better be 3.9.13.

         2) sqlite3 should be as latest as possible, better be 3.33.0.

         After making sure that the above requirements are satisfied, please go to step 3) below.

         3) The versions of the dependencies are specified in the exp-requirements.txt file itself,
         but repeating it once more here for clarity:

         Dependency Versions:

         pandas>=1.4.3
         numpy>=1.23.2
         mlflow>=1.28.0
         scikit-learn>=1.1.2
         prefect>=2.3.1
         boto3>=1.24.61
         

       * Copy exp-requirements.txt, exp.py, and exp-deploy.py from your local copy to a directory/environment
         in EXP_SERVER where you will run the experiments.

       * Do 'sudo yum install' in Linux environments.

       * run "pip3 install -r exp-requirements.txt" to install the dependencies. 
         Note: In the latest versions of Python, pip3 is just pip. Check what works in your EXP_SERVER.

       * Run 'aws configure' and configure your aws credentials.

       * Start the Prefect Cloud UI using command in the browser - "https://app.prefect.cloud". Create a workspace
         where you will be monitoring all the flow runs.

       * Link your local environment to the workspace created in the previous step using:
         "prefect cloud workspace set" and then selecting the workspace from the options provided.

       * Start Prefect agent in a terminal of EXP_SERVER using command 'prefect agent start -q <QUEUE_NAME>
         Note: The QUEUE_NAME is specified under the deployment definition in exp_deploy.py

       * Run 'python3 exp_deploy.py' or 'python exp_deploy.py' depending on what Python Version 3 is available in your
         machine. It is recommended that you run Python 3.9.13 for the latest and stable version in
         the 3.9 branch.

       * The python program exp_deploy.py runs and does the following:

         - Executes the logic from exp.py:

             - Runs the experiment with several combinations of algorithms, feature engineering, hyperparameters
               and number of cross validation cycles to be performed.

             - Selects the best model and registers it in MLFlow

             - Promotes the best model as Version 1 in production in MLFlow (when run for the first time)

         - Deploys the prefect deployment specification in exp_deploy.py and schedules
           prefect flow runs that would run once in every month.

       * Check the Prefect flow runs created, scheduled, and completed in the launched Prefect Cloud
         under the workspace created.

       The flow runs should be deployed and completed at the interval configured in the deployment spec.
       defined in exp_deploy.py.

       This completes the instructions for running, scheduling and tracking experiment in exp_deploy.py


     MODEL DEPLOYMENT AND PREDICTION (model_deploy/predict.py)

       * In a linux machine, you can run the model deployment and prediction.
         Note that since this module uses 'gunicorn', the machine should be Linux,
         because gunicorn does not run in Windows.
       
       To predict, first the latest model that gives the best performance needs to be taken from MLFlow.

       * Launch MLFlow UI in the browser at 'http://<MLFLOW_ADDRESS>:5000'

       * Navigate to the experiment "Experiment In The Cloud" in the left pane, click on "Models" tab on the right panel,
         and select the version that you would like to use. Then, click on the run ID displayed in the model version page
         to get to the run.  Copy the URL of this page. This link RUN_LINK will be used in deployment
         and prediction.

       * Open model_deploy/predict.py and make the following modifications:

             1) Update the MLFLOW_SERVER_URI with the MLFLOW_ADDRESS IP address

             2) Update the RUN variable with the RUN_LINK

       * Create a docker image in the current directory (model_deploy directory) using the following command.
         The Dockerfile for building the docker image is available in the same directory.

             'docker build -t predict:v1 .'

       * After the image is built, find the image from the 'docker images' list, and then run the docker image:

             'docker run -it --rm -p 9696:9696 predict:v1'

       * Now for the prediction part. Predict the IRIS flower type given the flower's sepal length,
         sepal width, petal length, and petal width.

         As a sample, select one data point which has:

         Sepal Length: 5.1
         Sepal Width: 3.5
         Petal Length: 1.4
         Petal Width: 0.2

         The correct IRIS flower type for these values is 'Iris Setosa'

         Launch the browser with URL: http://127.0.0.1:9696/predict/5.1/3.5/1.4/0.2

         and verify that 'Iris Setosa' is displayed.

         You can similarly provide various data values from the Iris-data.csv file included
         in this project repository to check if the flower types are predicted correctly. Most of them
         should be, but there could be very minimal wrong predictions here and there. Accuraries I got:

             - Best accuracy: 1 wrong prediction out of 150 entries, which is 0.66%. The worst prediction accuracy
             - Worst accuracy: 5 wrong predictions out of 150 entries, which is 3.33%.

         This completes the instructions for model deployment and prediction.



    





