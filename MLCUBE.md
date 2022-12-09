# MLCUBE

MLCube brings the concept of interchangeable parts to the world of machine learning models. It is the shipping container that enables researchers and developers to easily share the software that powers machine learning.

MLCube is a set of common conventions for creating ML software that can just "plug-and-play" on many systems. MLCube makes it easier for researchers to share innovative ML models, for a developer to experiment with many models, and for software companies to create infrastructure for models. It creates opportunities by putting ML in the hands of more people.

MLCube isn’t a new framework or service; MLCube is a consistent interface to machine learning models in containers like Docker. Models published with the MLCube interface can be run on local machines, on a variety of major clouds, or in Kubernetes clusters - all using the same code. MLCommons provides open source “runners” for each of these environments that make training a model in an MLCube™ a single command.

To learn more about the MLCube core concepts please refer to the [documentation](https://mlcommons.github.io/mlcube/getting-started/concepts/).

## Configuring MLCube

Cubes need to be configured before they can run. MLCube runners do that automatically, and users do not need to run the configure step manually. If for some reason this needs to be done, for instance, to pre-build or pull docker images (if these processes take too much time), MLCube runtime implements configure command. The Hello World cube is a Docker-based cube, and users can configure the MLCube by running the following command: mlcube configure --mlcube=. --platform=docker The Docker runner will build or will pull the docker image for the Hello World cube. As it is mentioned above, this step is optional and is only required when MLCubes need to be rebuilt. This can happen when users change implementation files and want to re-package their ML project into MLCube. In other situations, MLCube runners can auto-detect if configure command needs to be run before running MLCube tasks.

## MLCube tasks

The machine learning (ML) community has seen an explosive growth and innovation in the last decade. New models emerge on a daily basis, but sharing those models remains an ad-hoc process. Often, when a researcher wants to use a model produced elsewhere, they must waste hours or days on a frustrating attempt to get the model to work. Similarly, a ML engineer may struggle to port and tune models between development and production environments which can be significantly different from each other. This challenge is magnified when working with a set of models, such as reproducing related work, employing a performance benchmark suite like MLPerf, or developing model management infrastructures. Reproducibility, transparency and consistent performance measurement are cornerstones of good science and engineering.

The field needs to make sharing models simple for model creators, model users, developers and operators for both experimental and production purpose while following responsible practices. Prior works in the MLOps space have provided a variety of tools and processes that simplify user journey of deploying and managing ML in various environments, which include management of models, datasets, and dependencies, tracking of metadata and experiments, deployment and management of ML lifecycles, automation of performance evaluations and analysis, etc.

We propose an MLCube, a contract for packaging ML tasks and models that enables easy sharing and consistent reproduction of models, experiments and benchmarks amidst these existing MLOps processes. MLCube differs from an operation tool by acting as a contract and specification as opposed to a product or implementation.

This repository contains a number of MLCube examples that can run in different environments using MLCube runners.

    MNIST MLCube downloads data and trains a simple neural network. This MLCube can run with Docker or Singularity locally and on remote hosts. The README file provides instructions on how to run it. MLCube documentation provides additional details.
    Hello World MLCube is a simple exampled described in this tutorial.
    EMDenoise MLCube downloads data and trains a deep convolutional neural network for Electron Microscopy Benchmark. This MLCube can only run the Docker container. The README file provides instructions on how to run it.
    Matmul Matmul performs a matrix multiply.


## Installation

### Install MLCube Docker runner¶

pip install mlcube-docker # Install. mlcube config --get runners # Check it was installed. mlcube config --list # Show system settings for local MLCube runners. Depending on how your local system is configured, it may be required to change the following settings: - platforms.docker.docker (string): A docker executable. Examples are docker, nvidia-docker, sudo docker, podman etc. - platforms.docker.env_args (dictionary) and platforms.docker.build_args (dictionary). Environmental variables for docker run and build phases. Http and https proxy settings can be configured here. A custom configuration could look like: yaml platforms: docker: docker: sudo docker env_args: http_proxy: http://proxy.company.com:8088 https_proxy: https://proxy.company.com.net:8088 build_args: http_proxy: http://proxy.company.com:8088 https_proxy: https://proxy.company.com:8088

Download and install docker using this link. Make sure that you have docker runing before following the next steps.

Instal virtualenv

To install virtualenv run the following command on your terminal

pip install virtualenv
Create and activate virtual enviroment

In order to create and activate a virtual environment running the following commands:

virtualenv -p python3 ./env
source ./env/bin/activate
Install mlcube runner

Once you activate your environment, you have to install mlcube running the following command:

pip install mlcube-docker
putting your model into place

When the previous step is completed, you will find a directory called dataperf-vision-selection. Access that folder and go to the route selection.py where you are going to find a class call "Predictor" with a method named "selection". This metod is the one that you must update.

### Running MLCubes

Docker runner runs the following command:
${docker.docker} run {run_args} ${docker.env_args} {volumes} ${docker.image} {task_args}
where:

    ${docker.docker} is the docker executable.
    {run_args} are either ${docker.cpu_args} or ${docker.gpu_args} depending on ${platform.num_accelerators} value.
    ${docker.env_args} are the docker environmental variables.
    {volumes} are the mount points that the runner automatically constructs based upon the task input/output specifications.
    ${docker.image} is the docker image name.
    {task_args} is the task command line arguments, constructed automatically by the runner.
