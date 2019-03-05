# GIPHY Celebrity Detector

GIPHY's Open Source Celebrity Detection Deep Learning Model and Code

## About

GIPHY is proud to release our custom machine learning model that is able to discern over 2,300 celebrity faces with 98% accuracy. The model was trained to identify the most popular celebs on GIPHY, and can identify and make predictions for multiple faces across a sequence of images, like GIFs and videos.

This project was developed by the GIPHY R&D team with the goal to build a deep learning model that could annotate our most popular content as well as, or hopefully better than, similar models offered by major tech companies. We’re extremely proud of our results, and have released this model and training code to the public in hopes that others might build off our work, integrate the model into their own projects, or perhaps learn from our approach.

Read more about the project on the [GIPHY engineering blog](https://engineering.giphy.com/giphys-ai-can-identify-lil-yachty-can-yours).

You can play with the model on the [demo page](https://celebrity-detection.giphy.com/), and we’ve provided a [3D projection](https://celebrity-detection-projector.giphy.com/) of all our celebrity class embeddings along with a [list of all celebrities](https://github.com/Giphy/celeb-detection-oss/blob/master/examples/resources/face_recognition/labels.csv) available with the model.

Thank you!

The GIPHY R&D Team

Nick Hasty [@jnhasty](https://github.com/jnhasty), Ihor Kroosh [@tilast](https://github.com/tilast), Dmitry Voitekh [@dvoitekh](https://github.com/dvoitekh), Dmytro Korduban [@dkorduban](https://github.com/dkorduban)


## Try it out!

Follow the instructions in the [examples directory](./examples) to download the model and test it on your own GIFs and videos.

## Prerequisites

1. Python 3.6 or higher

2. For Linux: libsm, libxext, libxrender

## Training & Transfer Learning Experimentation Pipeline

Preliminary steps:

1. Create a work directory to store results of experiments (it's not mandatory to locate this directory within the project). Example is provided [here](./workdir/).

2. Inside a work directory create an experiment directory. It's name must match the name of the related experiment python file (e.g. [example_experiment](./workdir/example_experiment/) directory for [example_experiment.py](./experiments/example_experiment.py) file).

2. Create a directory `face_recognition` inside work directory, which must contain weights for MTCNN model (3 files with names `det1.npy`, `det2.npy`, and `det3.npy` that can be copied from [Giphy pretrained resources archive](https://s3.amazonaws.com/giphy-public/models/celeb-detection/resources.tar.gz)).

3. Create a file `labels.csv` inside the experiment directory. It must be of the following structure (see example [here](examples/resources/face_recognition/labels.csv)):

    ```
    Label,Index
    Person1,0
    Person2,1
    Person3,3
    ...
    ```

4. Create directory `raw_dataset` inside the experiment directory. It's a dataset of uncropped images. It must be of the following structure:

    ```
    - raw_dataset
      - Person1
        image1.jpg
        image2.jpg
        image3.jpg
        ...
      + Person2
      + Person3
      ...
    ```

So the overall structure of the work directory is as follows:

```
- workdir
  - example_experiment
    - raw_dataset
      - Person1
        image1.jpg
        image2.jpg
        image3.jpg
        ...
      + Person2
      + Person3
      ...
    labels.csv
  - face_detection
    det1.npy
    det2.npy
    det3.npy
```

After that, you need to choose where you going to run training: on CPU or on GPU. According to this decision you need to change `requirements_cpu.txt` to `requirements_gpu.txt` in `setup.py`, or leave it as is. Also, please, mind changing `tensorflow` version if necessary.

### Using Python 3.6 Package

1. Create a virtual environment to localize dependencies (optional):

    https://virtualenv.pypa.io/en/latest/

    ```
    pip install --upgrade virtualenv
    virtualenv -p python3 venv
    activate
    source ./venv/bin/activate
    ```

2. Install the package:

    ```
    pip install -e .
    ```

3. Create your experiment (see `experiments` directory for example).

4. `cp .env.example .env` and fill the missing values in the `.env` file if needed.

5. Run it from the top-level directory like:

    ```
    python experiments/example_experiment.py
    ```

### Using Docker

1. Install [nvidia-docker2](https://github.com/NVIDIA/nvidia-docker)

2. Modify your `/etc/docker/daemon.json` file:

    ```
    {
        "default-runtime": "nvidia",
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
            }
        }
    }
    ```

3. Create your experiment (see `experiments` directory for examples). After that, specify name of this file in `Dockerfile` in `CMD` section.

4. `cp .env.example .env` and fill the missing values, also ensure that you changed data directory and tensorboard port in `docker-compose.yml` file.

5. Before running any commands you need to explicitly activate the following env variables in your shell, since they are required during container launch:

    1) `TENSORBOARD_PORT`

    2) `WORKDIR`

    3) `LOCAL_WORKDIR`

6. Run via docker-compose:

    ```
    docker-compose up --build
    ```

    Or via plain Docker commands, for example:

    ```
    docker build -t celebrity-detection-model-train .
    docker run --rm --volume $LOCAL_WORKDIR:$WORKDIR --env-file .env --runtime=nvidia --shm-size 8G -p $TENSORBOARD_PORT:$TENSORBOARD_PORT celebrity-detection-model-train
    ```

