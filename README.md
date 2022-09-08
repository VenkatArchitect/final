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

       Initializing MLFlow

       * Create an AWS EC2 instance for MLFlow along with an AWS S3 bucket for
         artifact(s) and AWS RDS for MLFlow database as described at:
         https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md

         Configure EC2 instance's security group to accept TCP port 5000 as inbound
         in the inbound rules to let MLFlow inbound connections in.

         The MLFlow Postgresql database, which will be hereafterwards referred
         in this document as POSTGRESQLNAME

         The MLFlow artifact, which is the AWS S3 bucket will be hereafterwards referred
         in this document as MLFLOW_AWS_S3_BUCKET.

         The username and password noted down during the database creation will be known
         hereafterwards as DB_USER and DB_PASSWORD.

         Note down the endpoint name of the RDS database from its dashboard. It will be
         known hereafterwards as DB_ENDPOINT.

       * Note down the public IP address of the above-created EC2 instance, hereafterwards
         referred in this document as MLFLOW_ADDRESS

       * In the local copy of the project:
            A) Modify the IP address in the TRACKING_URI definition in exp.py somewhere around line 16
               to MLFLOW_ADDRESS noted above.
            B) Create an AWS S3 bucket and specify the bucket name for BUCKET_NAME definition somewhere
               around line 14 of exp.py.
            C) Copy iris-data.csv file from your local copy to the S3 bucket.

       * The name of the experiment is initialized to "Experiment In The Cloud" in this project. You will
         find an experiment created in MLFlow with this name.

       * Launch the MLFlow AWS instance and install mlflow with 'pip3 install mlflow'.
         Note: If there is an error about psycopg2, then do 'pip3 install psycopg2-binary' to install the dependency.

       * Install 'boto3' for AWS S3 access using 'pip3 install boto3'.

       * Make sure that you have your AWS credentials specified in this server by running 'aws configure'.
     
       * Start the MLFlow server in the MLFlow AWS EC2 instance with the following command:
         
         mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://DB_USER:DB_PASSWORD@DB_ENDPOINT:5432/POSTGRESQLNAME --default-artifact-root s3://MLFLOW_AWS_S3_BUCKET_NAME

       * Create an AWS EC2 instance for running the experiments, which will be
         hereafterwards referred to as EXP_AWS_EC2

       * Copy requirements.txt and exp.py from your local copy to the AWS EC2 instance created
         (usually under "/home/ec2-user")

       * run "pip3 install -r requirements.txt" to install the dependencies. 
         Note: In the latest versions of Python, pip3 is just pip. Check what works in your AWS EC2 instance.

         The versions of the dependencies are specified in the requirements.txt file itself,
         but repeating it once more here for clarity:

         Dependency Versions:

         pandas>=1.4.3
         numpy>=1.23.2
         mlflow>=1.28.0
         scikit-learn>=1.1.2
         prefect>=2.3.1
         boto3>=1.24.61

       * Start the Prefect Orion UI using command 'prefect orion start' in a terminal of the EXP_AWS_EC2         

       * Start Prefect agent in one more terminal of EXP_AWS_EC2 using command 'prefect agent start -q <QUEUE_NAME>
         Note: The QUEUE_NAME is specified under the deployment definition in exp.py

       * Run 'python3 exp.py' or 'python exp.py' depending on what Python Version 3 is available in your
         AWS EC2 instance. It is recommended that you run Python 3.9.13 for the latest and stable version in
         the 3.9 branch.

       * The python program exp.py runs and does the following:

         - Runs the experiment with several combinations of algorithms, feature engineering, hyperparameters
           and number of cross validation cycles to be performed.

         - Selects the best model and registers it in MLFlow

         - Promotes the best model as Version 1 in production in MLFlow (when run for the first time)

         - Reads the Prefect deployment specification specified somewhere around line 278 of exp.py and schedules
           prefect flow runs that would run once in every month.

       * To check the Prefect flow runs created, scheduled, and completed, launch the prefect orion UI in the browser:

         http://127.0.0.1/4200

       This completes the instructions for running, scheduling and tracking experiment in exp.py


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
         in this repository to check if the flower types are predicted correctly. Most of them
         should be, but there can be wrong predictions here and there. The best prediction
         accuracy I got was 1 wrong prediction out of 150 entries, which is 0.66%. The worst prediction accuracy
         I got was 5 wrong predictions out of 150 entries, which is 3.33%.

         This completes the instructions for model deployment and prediction.





       






    




