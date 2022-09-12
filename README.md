    PROBLEM STATEMENT

    Iris is a flowering plant that has around 300 species.  Attempts
    have been made to identify the plant species and their flowers
    using the properties of their flowers.  In 1936, Ronald Fisher
    came up with Iris flower data set which contained the classifications
    for three Iris varieties.  The Wikipedia article about this data set
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

    The data set contains 150 samples, 50 from each flower type.

    The problem that we are trying to solve is to uniquely identify the
    flower type based on its features. The dataset is provided in a csv
    file.


    SOLUTION DETAILS

    In the dataset, the flower characteristics as well as the target variable
    are identified. We will split the dataset into training and testing data,
    remove the target variable from the test data, do the predictions, and
    then do the comparison of the predicted values with the actuals identified
    in the dataset.

    While trying to classify the flower types, one challenge that we face
    is that two of the properties, sepal length and sepal width of flower
    types Iris-virginca and Irisversicolor are not clearly distinguishable
    as their values overlap. But their petal length and petal width clearly
    distinguish between each other, and hence we will be able
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
    the predictions. We will consider the following factors while doing the
    experiment runs to see if they have an impact on the end results:

        * The algorithm used
        * The algorithm's hyperparameters
        * Application of Feature Engineering
        * No. of cross validation cycles used while training the model

    We will do several runs of the experiment with combinations of above factors
    to pick a model that gives the best possible accuracy.  The accuracy of
    the results will be gauged by two factors:

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

    The repository is available at: https://github.com/VenkatArchitect/final

    Get a local copy of the repository to run the project.


    RUNNING THE EXPERIMENT (exp.py)

       1. Deploy the necessary infrastructure through IaC (Infrastructure-As-Code)
       through Terraform. The terraform file that defines the infrastructure is
       available in the project directory as "tf-iac.tf", using AWS as the service
       provider. Plan, validate, and apply the infrastructure to execute the project.


       2. Setup the mlflow server created above using the instructions in
       https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md

       Make sure that the security group you select for the RDS has permissions to the ports referred
       to in the link above. 


       We will use the following naming conventions hereafterwards in this document:

       POSTGRESQLNAME : MLFlow RDS database
       MLFLOW_AWS_S3_BUCKET: MLFlow artifacts bucket ('tf-mlflow-artifact-bucket' in tf-iac.tf)
       DB_USER: RDS username
       DB_PASSWORD: RDS password
       DB_ENDPOINT: RDS endpoint
       MLFLOW_ADDRESS: MLFlow public IP address
       DATASET_S3: Dataset S3 bucket ('tf-dataset-bucket' in tf-iac.tf)
       MLFLOW_AWS_EC2: MLFlow server ('tf-mlflow-server' in tf-iac.tf)
       EXP_SERVER: Server to do the experiment runs ('tf-exp-server in tf-iac.tf)


       3. Do the following steps from the local directory

       a) Modify the IP address in the TRACKING_URI definition in local exp.py somewhere around line 19
          to MLFLOW_ADDRESS noted above.
       b) Specify the bucket name for BUCKET_NAME definition somewhere around line 17 in local exp.py.
       c) Upload iris-data.csv from local directory to DATASET_S3 bucket.


       4. Start MLFLOW_AWS_EC2 and connect to it. Do the following steps in the MLFLOW_AWS_EC2:

       a) Do 'sudo yum update'
       b) Install Python version 3.9.13 and pip version 22.2.0 (Use superuser mode if applicable)
       c) Copy mlflow-server-requirements.txt from local directory to MLFLOW_AWS_EC2.
       d) Run 'pip install -r mlflow-server-requirements.txt'
       e) Configure AWS credentials using 'aws configure'.

       
       5. Start mlflow in MLFLOW_AWS_EC2 using the following command:

       mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/POSTGRESQLNAME --default-artifact-root s3://MLFLOW_AWS_S3_BUCKET_NAME


       6. Start EXP_SERVER and do the following setup:
       a) Do 'sudo yum update'
       b) Install Python version 3.9.13, pip version 22.2.0, and SQLite3 version 3.33.0 (Use superuser mode if applicable)
       c) Copy exp-requirements.txt, exp.py, and exp_deploy.py from your local directory to EXP_SERVER. The exp-requirements.txt
          has the version dependencies for the various software specified.
       d) Run "pip install -r exp-requirements.txt
       e) Run 'aws configure' and configure your aws credentials.
       f) Start Prefect Cloud UI using command in the browser - "https://app.prefect.cloud". Create a workspace
         where you will be monitoring all the flow runs.
       g) Link the workspace created to EXP_SERVER environment using
         "prefect cloud workspace set" and then selecting the workspace from the options provided.
       h) Start Prefect agent in a new terminal in EXP_SERVER using command 'prefect agent start -q QUEUE_NAME
         Note: The QUEUE_NAME is specified under the deployment definition in exp_deploy.py
       i) Run 'python exp_deploy.py'. This does the following:
 
             - Runs the experiment with several combinations of algorithms, feature engineering, hyperparameters
               and number of cross validation cycles to be performed.

             - Selects the best model and registers it in MLFlow

             - Promotes the best model as Version 1 in production in MLFlow (when run for the first time)

             - Deploys the prefect deployment specification in exp_deploy.py and schedules
               prefect flow runs that would run once in every month.

       j) Check the Prefect flow runs created, scheduled, and completed in the launched Prefect Cloud
         under the workspace created. The experiment name will be "Experiment In The Cloud". The flow runs
         should be deployed and completed at the interval configured in the deployment spec. defined in exp_deploy.py.


    This completes the instructions for running, scheduling and tracking experiment runs.


    MODEL DEPLOYMENT AND PREDICTION (model_deploy/predict.py)

    1. Model deployment should be run on a Linux machine as gunicorn is not supported in Windows. Do the following
       steps in the machine:

        Let's note the machine IP address as PREDICT_ADDRESS

        a) Do 'sudo yum update'
        b) Install Python version 3.9.13, pip version 22.2.0, and SQLite3 version 3.33.0 (Use superuser mode if applicable)
        c) Create a directory called model_deploy. Copy predict.py and Dockerfile to this directory. The Dockerfile
           has the version dependencies for the various software used.
        d) Launch mlflow to find the run at 'http://<MLFLOW_ADDRESS>:5000'. Navigate to the experiment "Experiment In The Cloud" 
           in the left pane, click on "Models" tab on the right panel,
           and select the version that you would like to use. Note down the run id. displayed in this page as RUN_ID.
        e) Open predict.py and make the following modifications:
            - Update the MLFLOW_SERVER_URI with the MLFLOW_ADDRESS IP address
            - Update the RUN_ID variable with the RUN_ID noted above.
        f) Run 'aws configure' and configure your aws credentials.
        g) Launch MLFlow UI in the browser at 'http://<MLFLOW_ADDRESS>:5000'
        h) Make sure that the security group of the server allows 9696 port in inbound rules.
        i) Install docker in the machine by following the instructions at:
            https://www.cyberciti.biz/faq/how-to-install-docker-on-amazon-linux-2/
        j) Create a docker image in model_deploy directory using the following command.

             'docker build -t predict:v1 .'

    2. After the image is built, find the image from the 'docker images' list, and then run the docker image:

             'docker run -it --rm -p 9696:9696 predict:v1'

    3. Predict the IRIS flower type given the flower's sepal length, sepal width, petal length, and petal width.

         Select one data point which has:

         Sepal Length: 5.1, Sepal Width: 3.5, Petal Length: 1.4, Petal Width: 0.2
         The correct IRIS flower type for these values is 'Iris Setosa'

         Launch the browser with URL: http://PREDICT_ADDRESS:9696/predict/5.1/3.5/1.4/0.2
         and verify that 'Iris Setosa' is displayed.

         You can similarly provide various data values from the Iris-data.csv file included
         in this project repository to check if the flower types are predicted correctly. Most of them
         should be, but there could be very minimal wrong predictions here and there. Accuraries I got:

             - Best accuracy: 1 wrong prediction out of 150 entries, which is 0.66%. The worst prediction accuracy
             - Worst accuracy: 5 wrong predictions out of 150 entries, which is 3.33%.

     This completes the instructions for model deployment and prediction.


     UNIT AND INTEGRATION TESTS

     Unit and integration tests are available in tests/test_exp.py.
