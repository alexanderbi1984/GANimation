import os
import argparse
import glob
import cv2
from utils import face_utils
from utils import cv_utils
import face_recognition
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle
import numpy as np
from models.models import ModelsFactory
from options.test_options import TestOptions

class MorphFacesInTheWild:
    def __init__(self, opt):
        self._opt = opt
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])

    def morph_file(self, img_path, expresion):
        img = cv_utils.read_cv2_img(img_path)

        if img is None:
            raise ValueError(f"Image at {img_path} could not be loaded.")

        print("Loaded image shape:", img.shape)

        morphed_img = self._img_morph(img, expresion)
        output_name = '%s_out.png' % os.path.basename(img_path)
        self._save_img(morphed_img, output_name)

    # def _img_morph(self, img, expresion):
    #     # Ensure the image is in the right format
    #     print("Initial image shape:", img.shape)
    #     print("Initial image data type:", img.dtype)
    #
    #     if img.dtype != np.uint8:
    #         img = img.astype(np.uint8)  # Convert to uint8 if it's not already
    #
    #     # Convert BGR to RGB if necessary
    #     if img.shape[2] == 3:  # Check if the image has 3 channels
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     elif img.shape[2] == 4:  # Check if the image has 4 channels
    #         img = img[:, :, :3]  # Discard the alpha channel
    #
    #     # Check the image min/max values
    #     print("Processed image data type:", img.dtype)
    #     print("Processed image min/max values:", img.min(), img.max())
    #
    #     # Now proceed to detect faces
    #     bbs = face_recognition.face_locations(img)
    #     print("Bounding boxes found:", bbs)
    #
    #     if len(bbs) > 0:
    #         y, right, bottom, x = bbs[0]
    #         bb = x, y, (right - x), (bottom - y)
    #         face = face_utils.crop_face_with_bb(img, bb)
    #         face = face_utils.resize_face(face)
    #     else:
    #         print("No faces found, using the entire image.")
    #         face = face_utils.resize_face(img)
    #
    #     # Perform face morphing
    #     morphed_face = self._morph_face(face, expresion)
    #
    #     return morphed_face

    import cv2
    import numpy as np
    from PIL import Image

    def _img_morph(self, img, expresion):
        print("Initial image shape:", img.shape)
        print("Initial image data type:", img.dtype)

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)  # Ensure it's uint8

        # Convert the image from BGR (if loaded with OpenCV) to RGB
        if img.shape[2] == 3:  # Check if the image has 3 channels
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to PIL and back to ensure proper format
        img_pil = Image.fromarray(img)
        img = np.array(img_pil.convert('RGB'))

        # Ensure the image is contiguous
        img = np.ascontiguousarray(img)

        print("Processed image shape:", img.shape)
        print("Processed image data type:", img.dtype)

        # Load Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detect faces
        bbs = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        print("Bounding boxes found:", bbs)

        if len(bbs) > 0:
            # Take the first detected face
            x, y, width, height = bbs[0]
            bb = x, y, width, height
            face = face_utils.crop_face_with_bb(img, bb)  # Assuming this function exists
            face = face_utils.resize_face(face)  # Resize the detected face
        else:
            print("No faces found, using the entire image.")
            face = face_utils.resize_face(img)

        # Perform face morphing
        morphed_face = self._morph_face(face, expresion)

        return morphed_face

    def _morph_face(self, face, expresion):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        expresion = torch.unsqueeze(torch.from_numpy(expresion/5.0), 0)
        test_batch = {'real_img': face, 'real_cond': expresion, 'desired_cond': expresion, 'sample_id': torch.FloatTensor(), 'real_img_path': []}
        self._model.set_input(test_batch)
        imgs, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return imgs['concat']

    def _save_img(self, img, filename):
        filepath = os.path.join(self._opt.output_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)


def main():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    print("Current working directory:", os.getcwd())

    morph = MorphFacesInTheWild(opt)
    image_path = opt.input_path
    # Check if the input image path is valid
    if not os.path.isfile(image_path):
        raise ValueError(f"The specified image path does not exist or is not a file: {image_path}")


    expression = np.random.uniform(0, 1, opt.cond_nc)
    morph.morph_file(image_path, expression)



if __name__ == '__main__':
    main()
