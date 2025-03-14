import os
import cv2
import tarfile
import pydload
import logging
import numpy as np
import onnxruntime
from .video_utils import get_interest_frames_from_video
from .image_utils import load_images, load_unsave_images
from PIL import Image as pil_image


class Classifier:
    """
    Class for loading model and running predictions.
    For example on how to use take a look the if __name__ == '__main__' part.
    """
    nsfw_model = None

    def __init__(self, model_path):
        """
        model = Classifier()
        """
        # url = "https://github.com/notAI-tech/NudeNet/releases/download/v0/classifier_model.onnx"
        # home = os.path.expanduser("~")
        # model_folder = os.path.join(home, ".NudeNet/")
        # if not os.path.exists(model_folder):
        #     os.mkdir(model_folder)

        # model_path = os.path.join(model_folder, os.path.basename(url))

        # if not os.path.exists(model_path):
        #     print("Downloading the checkpoint to", model_path)
        #     pydload.dload(url, save_to_path=model_path, max_time=None)

        self.nsfw_model = onnxruntime.InferenceSession(model_path)

    def classify(
        self,
        images=[],
        image_names=[],
        batch_size=4,
        image_size=(256, 256),
        categories=["unsafe", "safe"],
    ):
        """
        inputs:
            image_paths: list of image paths or can be a string too (for single image)
            batch_size: batch_size for running predictions
            image_size: size to which the image needs to be resized
            categories: since the model predicts numbers, categories is the list of actual names of categories
        """
        if not isinstance(images, list):
            images = [images]

        loaded_images = load_unsave_images(
            images, image_size
        )
        loaded_image_paths = image_names

        if not loaded_image_paths:
            return {}

        preds = []
        model_preds = []
        while len(loaded_images):
            _model_preds = self.nsfw_model.run(
                [self.nsfw_model.get_outputs()[0].name],
                {self.nsfw_model.get_inputs()[0].name: loaded_images[:batch_size]},
            )[0]
            model_preds.append(_model_preds)
            preds += np.argsort(_model_preds, axis=1).tolist()
            loaded_images = loaded_images[batch_size:]

        probs = []
        for i, single_preds in enumerate(preds):
            single_probs = []
            for j, pred in enumerate(single_preds):
                single_probs.append(
                    model_preds[int(i / batch_size)][int(i % batch_size)][pred]
                )
                preds[i][j] = categories[pred]

            probs.append(single_probs)

        images_preds = {}

        for i, loaded_image_path in enumerate(loaded_image_paths):
            if not isinstance(loaded_image_path, str):
                loaded_image_path = i

            images_preds[loaded_image_path] = {}
            for _ in range(len(preds[i])):
                images_preds[loaded_image_path][preds[i][_]] = float(probs[i][_])

        return images_preds


# if __name__ == "__main__":
#     m = Classifier()

#     while 1:
#         print(
#             "\n Enter single image path or multiple images seperated by || (2 pipes) \n"
#         )
#         images = input().split("||")
#         images = [image.strip() for image in images]
#         print(m.predict(images), "\n")
