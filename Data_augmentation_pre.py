import Augmentor
import pandas as pd
import os
import numpy as np
from PIL import Image


def data_augmentation():

    p1 = Augmentor.Pipeline("augmentation_image/train/0")
    p1.rotate(1, max_left_rotation=5, max_right_rotation=5)
    p1.flip_top_bottom(0.5)
    p1.flip_left_right(probability=0.5)
    p1.zoom(probability=0.3, min_factor=0.8, max_factor=1.2)
    p1.random_distortion(probability=0.3, grid_width=4, grid_height=4, magnitude=8)
    p1.sample(1500)

    p2 = Augmentor.Pipeline("augmentation_image/train/1")
    p2.rotate(1, max_left_rotation=5, max_right_rotation=5)
    p2.flip_top_bottom(0.5)
    p2.flip_left_right(probability=0.5)
    p2.zoom(probability=0.3, min_factor=0.8, max_factor=1.2)
    p2.random_distortion(probability=0.3, grid_width=4, grid_height=4, magnitude=8)
    p2.sample(1500)

    p3 = Augmentor.Pipeline("augmentation_image/validation/0")
    p3.rotate(1, max_left_rotation=5, max_right_rotation=5)
    p3.flip_top_bottom(0.5)
    p3.flip_left_right(probability=0.5)
    p3.zoom(probability=0.3, min_factor=0.8, max_factor=1.2)
    p3.random_distortion(probability=0.3, grid_width=4, grid_height=4, magnitude=8)
    p3.sample(500)

    p4 = Augmentor.Pipeline("augmentation_image/validation/1")
    p4.rotate(1, max_left_rotation=5, max_right_rotation=5)
    p4.flip_top_bottom(0.5)
    p4.flip_left_right(probability=0.5)
    p4.zoom(probability=0.3, min_factor=0.8, max_factor=1.2)
    p4.random_distortion(probability=0.3, grid_width=4, grid_height=4, magnitude=8)
    p4.sample(500)

data_augmentation()

