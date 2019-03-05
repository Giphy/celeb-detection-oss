# Examples of using celebrity detection model

1. Before running these examples you need to ensure that you downloaded the latest archive with Giphy pretrained models (resources).
    It can be done by [this script](download_model.py):

    ```
    python download_model.py
    ```

    It will download the latest archive with all needed components and extract it into [resources directory](resources/).

2. Create `.env` file:

    ```
    cp .env.example .env
    ```

    and ensure that `APP_DATA_DIR` variable points to the [resources directory](resources/).

3. Create a virtual environment to localize dependencies (optional):

    https://virtualenv.pypa.io/en/latest/

    ```
    pip install --upgrade virtualenv
    virtualenv -p python3 ../venv
    activate
    source ../venv/bin/activate
    ```

4. Install `model_train` package by running the following command from the top-level directory:

    ```
    pip install -e .
    ```

5. Install requirements from [examples directory](./):

    ```
    pip install -r requirements.txt
    ```


##

Once you've completed the steps above, there are 2 ways to interact with our model: an inference script and flask application.

### Script for model inference

With url or local path to `jpg` image:

```
python inference.py --image_path media/image.jpg
```

or with url or local path to `gif` (or `mp4`) video:

```
python inference.py --gif_path media/video.gif
```

### Flask application

To start the app just run:

```
python app.py
```

Open provided url in your browser and you should see a UI where you can paste links to images and gifs to test the model.
