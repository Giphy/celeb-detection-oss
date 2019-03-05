FROM ufoym/deepo:pytorch-py36

LABEL description="Celebrity Detection Model Training"

RUN apt update && \
    apt install -y libsm6 libxext6 libxrender-dev

ADD . .
RUN pip install -e .

WORKDIR /experiments

EXPOSE $TENSORBOARD_PORT

CMD python $EXPERIMENT_FILENAME
