import Augmentor
import pandas as pd
import os
import numpy as np
from PIL import Image


table = pd.read_csv('models/division/patient_division397modify.csv')


def save_image():
    count = 0
    for m in range(table.shape[0]):
        id = table.iat[m,0]
        cohort = table.iat[m,2]
        k = table.iat[m,1]
        for root, dirs, files in os.walk("/Users/liuqiang/PycharmProjects/Medical/final_pngs_2"):
            for dir in dirs:
                if str(dir).find(str(id), 0, len(str(dir))) > -1:
                    in_path = os.path.join(root, dir)
                    list = os.listdir(in_path)
                    if int(1 * len(list) / 2) <= 5:
                        for i in range(int(1 * len(list) / 4), int(3 * len(list) / 4)):
                            if str(list[i]).find('.DS_Store', 0, len(str(list[i]))) > -1:
                                continue
                            path = os.path.join(in_path, list[i])
                            if os.path.isfile(path):
                                im = Image.open(path)
                                if cohort == 'train':
                                    im.save("augmentation_image/train/"+str(k)+"/image_"+str(count)+".png")
                                    count += 1
                                elif cohort == 'validation':
                                    im.save("augmentation_image/validation/" + str(k) + "/image_" + str(count) + ".png")
                                    count += 1
                    else:
                        for i in range(int(1 * len(list) / 2) - 3, int(1 * len(list) / 2) + 2):
                            if str(list[i]).find('.DS_Store', 0, len(str(list[i]))) > -1:
                                continue
                            path = os.path.join(in_path, list[i])
                            if os.path.isfile(path):
                                im = Image.open(path)
                                if cohort == 'train':
                                    im.save("augmentation_image/train/"+str(k)+"/image_"+str(count)+".png")
                                    count += 1
                                elif cohort == 'validation':
                                    im.save("augmentation_image/validation/" + str(k) + "/image_" + str(count) + ".png")
                                    count += 1

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

