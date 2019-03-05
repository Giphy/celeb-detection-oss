import os
import argparse
import moviepy.editor as mov_editor

from dotenv import load_dotenv
from skimage import io
from pprint import pprint

from model_training.utils import preprocess_image
from model_training.helpers.labels import Labels
from model_training.helpers.face_recognizer import FaceRecognizer
from model_training.utils import evenly_spaced_sampling
from model_training.preprocessors.face_detection.face_detector import FaceDetector


def process_gif(path):
    gif = mov_editor.VideoFileClip(path)
    selected_frames = evenly_spaced_sampling(list(gif.iter_frames()), gif_frames)
    face_images_by_frames = face_detector.perform_bulk(selected_frames, range(len(selected_frames)))
    face_images = []
    for frame_faces in face_images_by_frames.values():
        face_images.extend([preprocess_image(image, image_size) for image, _ in frame_faces])
    return face_recognizer.perform(face_images)


def process_image(path):
    image = io.imread(path)
    face_images = face_detector.perform_single(image)
    face_images = [preprocess_image(image, image_size) for image, _ in face_images]
    return face_recognizer.perform(face_images)


if __name__ == '__main__':
    load_dotenv('.env')
    parser = argparse.ArgumentParser(description='Inference script for Giphy Celebrity Classifier model')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_path', type=str, help='path or link to the image', default=None)
    group.add_argument('--gif_path', type=str, help='path or link to the gif', default=None)
    args = parser.parse_args()

    image_size = int(os.getenv('APP_FACE_SIZE', 224))
    gif_frames = int(os.getenv('APP_GIF_FRAMES', 20))

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

    if args.image_path:
        predictions = process_image(args.image_path)
    else:
        predictions = process_gif(args.gif_path)

    pprint(predictions)
