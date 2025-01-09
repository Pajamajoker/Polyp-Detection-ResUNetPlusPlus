import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence

def parse_image(img_path, image_size):
    image_rgb = cv2.imread(img_path, 1)
    h, w, _ = image_rgb.shape
    if (h == image_size) and (w == image_size):
        pass
    else:
        image_rgb = cv2.resize(image_rgb, (image_size, image_size))
    image_rgb = image_rgb/255.0
    return image_rgb

def parse_mask(mask_path, image_size):
    mask = cv2.imread(mask_path, -1)
    h, w = mask.shape
    if (h == image_size) and (w == image_size):
        pass
    else:
        mask = cv2.resize(mask, (image_size, image_size))
    mask = np.expand_dims(mask, -1)
    mask = mask/255.0

    return mask

class DataGen(Sequence):
    def __init__(self, image_size, images_path, masks_path, batch_size=8):
        self.image_size = image_size
        self.images_path = images_path
        self.masks_path = masks_path
        self.batch_size = batch_size
        self.on_epoch_end()
    def on_epoch_end(self):
            pass

    def __len__(self):
        return int(np.ceil(len(self.images_path)/float(self.batch_size)))

    def __getitem__(self, index):
        # Adjust batch size if it's the last batch
        if (index + 1) * self.batch_size > len(self.images_path):
            self.batch_size = len(self.images_path) - index * self.batch_size

        images_path = self.images_path[index * self.batch_size : (index + 1) * self.batch_size]
        masks_path = self.masks_path[index * self.batch_size : (index + 1) * self.batch_size]

        images_batch = []
        masks_batch = []

        for i in range(len(images_path)):
            # Read image and mask
            image = parse_image(images_path[i], self.image_size)
            mask = parse_mask(masks_path[i], self.image_size)

            images_batch.append(image)
            masks_batch.append(mask)

        # Convert to NumPy arrays for returning
        images_array = np.array(images_batch)
        masks_array = np.array(masks_batch)

        print(f"Batch {index}: Images shape: {images_array.shape}, Masks shape: {masks_array.shape}")
        return images_array, masks_array

