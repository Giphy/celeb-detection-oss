import os
import hashlib
import logging
import sys
import moviepy.editor as mov_editor
import traceback
import numpy as np

from flask import Flask, render_template, request, send_from_directory
from dotenv import load_dotenv
from skimage import io

from model_training.helpers.labels import Labels
from model_training.helpers.face_recognizer import FaceRecognizer
from model_training.utils import ensure_dir, evenly_spaced_sampling, ACCEPTABLE_ERRORS, preprocess_image
from model_training.preprocessors.face_detection.face_detector import FaceDetector

load_dotenv('.env')
app = Flask(__name__)


def exception_handler(_):
    app.logger.debug(traceback.format_exc())
    return render_template('500.html')


def render_predictions(face_images, faces_predictions, filename):
    if len(faces_predictions) == 0:
        return render_template('empty_result.html', image_name=filename)

    crops_names = []
    for i, image in enumerate(face_images):
        crop_name = f'{i}_{filename}'
        io.imsave(os.path.join(TMP_DIR, crop_name), image)
        crops_names.append(crop_name)
    return render_template('result.html', image_name=filename,
                           crops=crops_names, faces_predictions=faces_predictions)


def load_image(path, url):
    try:
        image = io.imread(url)
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        image_hash = hashlib.md5(image.tostring()).hexdigest()
        base_name = f'{image_hash}.jpg'
        filename = os.path.join(path, base_name)
        io.imsave(filename, image)
        app.logger.info(f'Saved {url} to {filename}')
        return base_name, image
    except ACCEPTABLE_ERRORS as ex:
        app.logger.warn(f'Cannot download {url} error: {ex}')


def load_gif(path, url):
    try:
        gif = mov_editor.VideoFileClip(url)
        gif_hash = hashlib.md5(gif.filename.encode('utf-8')).hexdigest()
        base_name = f'{gif_hash}.gif'
        filename = os.path.join(path, base_name)
        gif.write_gif(filename)
        app.logger.info(f'Saved {url} to {filename}')
        return base_name, gif
    except ACCEPTABLE_ERRORS as ex:
        app.logger.warn(f'Cannot download {url} error: {ex}')


def process_image(image_url):
    filename, image = load_image(TMP_DIR, image_url)
    face_images = face_detector.perform_single(image)
    face_images = [preprocess_image(image, image_size) for image, _ in face_images]
    faces_predictions = face_recognizer.perform(face_images)

    return render_predictions(face_images, faces_predictions, filename)


def process_gif(gif_url):
    filename, gif = load_gif(TMP_DIR, gif_url)
    selected_frames = evenly_spaced_sampling(list(gif.iter_frames()), gif_frames)
    face_images_by_frames = face_detector.perform_bulk(selected_frames, range(len(selected_frames)))
    face_images = []
    for frame_faces in face_images_by_frames.values():
        face_images.extend([preprocess_image(image, image_size) for image, _ in frame_faces])
    faces_predictions = face_recognizer.perform(face_images)

    return render_predictions(face_images, faces_predictions, filename)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', available_labels=model_labels.labels_list)


@app.route('/img/<path>')
def img(path):
    return send_from_directory(TMP_DIR, path)


@app.route('/process', methods=['POST'])
def process():
    target_url = request.form['image_url']
    if any(target_url.endswith(e) for e in ('.gif', '.mp4')):
        return process_gif(target_url)
    return process_image(target_url)


if __name__ == '__main__':
    model_labels = Labels(resources_path=os.getenv('APP_DATA_DIR'))
    face_detector = FaceDetector(
        os.getenv('APP_DATA_DIR'),
        margin=float(os.getenv('APP_FACE_MARGIN', 0.2)),
        use_cuda=os.getenv('APP_USE_CUDA') == "true"
    )
    face_recognizer = FaceRecognizer(
        labels=model_labels,
        resources_path=os.getenv('APP_DATA_DIR'),
        use_cuda=os.getenv('USE_CUDA') == "true"
    )
    TMP_DIR = os.getenv('APP_TMP_DIR', './tmp')
    ensure_dir(TMP_DIR)

    image_size = int(os.getenv('APP_FACE_SIZE', 224))
    gif_frames = int(os.getenv('APP_GIF_FRAMES', 20))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.DEBUG)
    app.register_error_handler(Exception, exception_handler)
    app.run(port=int(os.getenv('APP_PORT', 5000)))
