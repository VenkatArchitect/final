terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 4.16"
    }
  }

  required_version = ">= 1.2.0"
}

provider "aws" {
  region  = "ap-south-1"
}

resource "aws_instance" "tf-exp-server" {
  ami           = "ami-06489866022e12a14"
  instance_type = "t2.micro"
  key_name = "ec2-keypair"
  security_groups = ["launch-wizard-2"]

  tags = {
    Name = "tf-exp-server"
  }
}

resource "aws_instance" "tf-mlflow-server" {
  ami           = "ami-06489866022e12a14"
  instance_type = "t2.micro"
  key_name = "ec2-keypair"
  security_groups = ["launch-wizard-2"]

  tags = {
      Name = "tf-mlflow-server"
  }
}

resource "aws_s3_bucket" "tf-dataset-bucket" {
  bucket = "tf-dataset-bucket"

  tags = {
    Name        = "tf-dataset-bucket"
    Environment = "Dev"
  }
}

resource "aws_s3_bucket" "tf-mlflow-artifact-bucket" {
  bucket = "tf-mlflow-artifact-bucket"
}

resource "aws_db_instance" "tf-mlflow-rds" {
  allocated_storage    = 20
  engine               = "postgres"
  instance_class       = "db.t3.micro"
  db_name              = "tfMlflowBackendDb"
  username             = "mlflowuser"
  password             = "mlflowpass"
  security_group_names = ["launch-wizard-2"]
  skip_final_snapshot  = true
}

