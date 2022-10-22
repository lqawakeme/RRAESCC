from __future__ import print_function
import logging
import SimpleITK as sitk
import numpy as np
import radiomics
from radiomics import featureextractor
import cv2
import xlrd
from PIL import Image
import os
import random
import pandas as pd

# Get some test data
# Download the test case to temporary files and return it's location. If already downloaded, it is not downloaded again,
# but it's location is still returned.
data1 = xlrd.open_workbook('../CR-PR/CR-PR_2.xlsx')
table = data1.sheets()[0]
params = 'para_original.yaml'
X = []
X_mask = []
path_image_origin = list()
path_image_mask = list()
indexs = list()
for m in range(table.nrows - 1):
    if_find = False
    k = table.cell(m, 5).value
    patient_id = table.cell(m, 0).value
    for root, dirs, files in os.walk(
            "/Users/liuqiang/PycharmProjects/Medical/origin_pngs_2"):
        for dir in dirs:
            if str(dir).find(str(patient_id), 0, len(str(dir))) > -1:
                in_path = os.path.join(root, dir)
                list = os.listdir(in_path)
                path = os.path.join(in_path, list[0])
                image_set = []
                for i in range(int(1 * len(list) / 2)-2, int(1 * len(list) / 2) + 2):
                    path = os.path.join(in_path, list[i])
                    if os.path.isfile(path):
                        im = cv2.imread(path)
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        im = im / 255.
                        images = sitk.GetImageFromArray(np.array(im.astype(np.int32)))
                        X.append(images)
                        indexs.append(patient_id)
                        path_image_origin.append(path)
    for root, dirs, files in os.walk(
            "/Users/liuqiang/PycharmProjects/Medical/masked_pngs_2"):
        for dir in dirs:
            if str(dir).find(str(patient_id), 0, len(str(dir))) > -1:
                in_path = os.path.join(root, dir)
                list = os.listdir(in_path)
                path = os.path.join(in_path, list[0])
                image_set = []
                for i in range(int(1 * len(list) / 2)-2, int(1 * len(list) / 2) + 2):
                    path = os.path.join(in_path, list[i])
                    if os.path.isfile(path):
                        im = cv2.imread(path)
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                        # blank_image = np.zeros((512, 512, 3), np.uint8)
                        # blank_image.fill(255)
                        # im = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
                        # cv2.imshow("ll",im)
                        # cv2.waitKey(0)
                        im = im / 255.
                        images_mask = sitk.GetImageFromArray(np.array(im.astype(np.int32)))
                        X_mask.append(images_mask)
                        path_image_mask.append(path)

# Regulate verbosity with radiomics.verbosity (default verbosity level = WARNING)
# radiomics.setVerbosity(logging.INFO)

# Get the PyRadiomics logger (default log-level = INFO)
logger = radiomics.logger
logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

# Set up the handler to write out all log entries to a file
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Define settings for signature calculation
# These are currently set equal to the respective default values
extractor = featureextractor.RadiomicsFeatureExtractor(params)

# By default, only original is enabled. Optionally enable some image types:
# extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})
#extractor.disableAllFeatures()
# Disable all classes except firstorder
extractor.enableAllFeatures()

# Enable all features in firstorder
#extractor.enableFeatureClassByName('glcm')

# Only enable mean and skewness in firstorder
#extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])
# featureVector = extractor.execute(images_1, images_2)
# print(featureVector)
print("Calculating features")
result = []
for m in range(len(X)):
    featureVector = extractor.execute(X[m], X_mask[m])
    result.append(featureVector)
    print(str(m))
df = pd.DataFrame(result)
df['path_origin_image']=path_image_origin
df['path_mask_image']=path_image_mask
#df.drop(df.columns[0:21],axis=1,inplace=True)
df.index = indexs
#df.to_csv('feature_original_2.csv')
