import xlrd
import os
import  pandas as pd
import shutil
from PIL import Image
import numpy as np
from PIL import Image
import random
from sklearn.metrics import roc_curve, auc
from PIL import ImageDraw
import cv2
from math import sin, cos, pi
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, LeakyReLU, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input,GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Add, Activation, Lambda
from tensorflow.keras import backend as K
import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.activations import sigmoid
import pickle
from xlrd import xldate_as_tuple
import datetime

X = []
Y = []
X_test = []
Y_test = []
for root, dirs, files in os.walk("/Users/liuqiang/PycharmProjects/Medical/CR-PR/augmentation_image/train/0/output"):
    for file in files:
        if str(file).find("png")>0:
            path = os.path.join(root,file)
            if os.path.isfile(path):
                im = Image.open(path)
                im = im.resize((64, 64))
              #  im = im.convert('RGB')
                np_im = np.array(im) / 255.
                X.append(np_im)
                Y.append(0)
for root, dirs, files in os.walk("/Users/liuqiang/PycharmProjects/Medical/CR-PR/augmentation_image/train/1/output"):
    for file in files:
        if str(file).find("png")>0:
            path = os.path.join(root,file)
            if os.path.isfile(path):
                im = Image.open(path)
                im = im.resize((64, 64))
               # im = im.convert('RGB')
                np_im = np.array(im) / 255.
                X.append(np_im)
                Y.append(1)
for root, dirs, files in os.walk("/Users/liuqiang/PycharmProjects/Medical/CR-PR/augmentation_image/validation/0/output"):
    for file in files:
        if str(file).find("png")>0:
            path = os.path.join(root,file)
            if os.path.isfile(path):
                im = Image.open(path)
                im = im.resize((64, 64))
              #  im = im.convert('RGB')
                np_im = np.array(im) / 255.
                X_test.append(np_im)
                Y_test.append(0)
for root, dirs, files in os.walk("/Users/liuqiang/PycharmProjects/Medical/CR-PR/augmentation_image/validation/1/output"):
    for file in files:
        if str(file).find("png")>0:
            path = os.path.join(root,file)
            if os.path.isfile(path):
                im = Image.open(path)
                im = im.resize((64, 64))
             #   im = im.convert('RGB')
                np_im = np.array(im) / 255.
                X_test.append(np_im)
                Y_test.append(1)

X = np.array(X)
Y = np.array(Y)
X_test = np.array(X_test)
Y_test = np.array(Y_test)
print(X.shape,Y.shape,len(X_test),len(Y_test))

channel_axis = 1 if K.image_data_format() == "channels_first" else 3
def channel_attention(input_xs, reduction_ratio=0.125):
    # get channel
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    maxpool_channel = KL.Reshape((1, 1, channel))(maxpool_channel)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    avgpool_channel = KL.Reshape((1, 1, channel))(avgpool_channel)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel), activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    # max path
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # avg path
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = KL.Activation('sigmoid')(channel_attention_feature)
    return KL.Multiply()([channel_attention_feature, input_xs])

# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    return KL.Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(max_avg_pool_spatial)


def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    return KL.Add()([refined_feature, input_xs])
def auc_score(pro_y, pro_scores):
    fpr, tpr, threshold = roc_curve(pro_y, pro_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return roc_auc
def auc_curve(pro_y, pro_scores, figname):
    fpr, tpr, threshold = roc_curve(pro_y, pro_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Training cohort')
    plt.legend(loc="lower right")
    plt.savefig(figname)
    plt.close()

def train_model():
    K.set_image_data_format('channels_last')
    pretrained_model = ResNet50(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
    pretrained_model.trainable = True
    inputs = Input(shape=(64, 64, 3))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Conv2D(3, (1, 1), padding='same', input_shape=(64, 64, 3))(inputs)
    x = LeakyReLU(alpha=0.1, name = 'Relu1')(x)
    x = pretrained_model(x)
    x = channel_attention(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1)(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.summary()

    earlyStopping = EarlyStopping(monitor='loss', patience=30, mode='min',
                                  baseline=None)

    rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.7, patience=5, min_lr=1e-6, mode='min', verbose=1)
    prop = optimizers.RMSprop(lr=0.01)
    adam = optimizers.Adam(lr=0.01)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X, Y, validation_data=(X_test,Y_test),epochs=50, batch_size=32,
                        callbacks=[earlyStopping, rlp])
    scores = model.predict(X)
    scores2 = model.predict(X_test)
    print(auc_score(Y,scores),auc_score(Y_test,scores2))
    model.save("augmentation_image/pretrained_model1.h5")

train_model()






