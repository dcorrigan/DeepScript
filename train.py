from __future__ import print_function

import random
import shutil
import os
import pickle

import sys

sys.setrecursionlimit(10000)

SEED = 1066987
import numpy as np

np.random.seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from scipy.misc import imsave

import keras
from keras.utils import np_utils

from DeepScript import utils
from DeepScript import resnet50, vgg16


NB_ROWS, NB_COLS = 150, 150
BATCH_SIZE = 30
NB_EPOCHS = 100
NB_TRAIN_PATCHES = 100
NB_TEST_PATCHES = 30
MODEL_NAME = "final"
MODEL_TYPE = "vgg16"  # resnet50 or vgg16

from keras.callbacks import ModelCheckpoint


def train():
    train_images, train_categories = utils.load_dir("data/splits/train")
    dev_images, dev_categories = utils.load_dir("data/splits/dev")

    try:
        os.mkdir("models")
    except:
        pass

    try:
        shutil.rmtree("models/" + MODEL_NAME)
    except:
        pass

    os.mkdir("models/" + MODEL_NAME)

    label_encoder = LabelEncoder().fit(train_categories)
    train_y_int = label_encoder.transform(train_categories)
    dev_y_int = label_encoder.transform(dev_categories)
    train_Y = np_utils.to_categorical(
        train_y_int, num_classes=len(label_encoder.classes_)
    )

    print(
        "-> Working on", len(label_encoder.classes_), "classes:", label_encoder.classes_
    )

    pickle.dump(label_encoder, open("models/" + MODEL_NAME + "/label_encoder.p", "wb"))

    if MODEL_TYPE == "resnet50":
        model = resnet50.ResNet50(
            weights=None,
            nb_classes=len(label_encoder.classes_),
            nb_rows=NB_ROWS,
            nb_cols=NB_COLS,
        )
    elif MODEL_TYPE == "vgg16":
        model = keras.applications.vgg16.VGG16(
            weights=None, include_top=True, input_shape=(1,NB_ROWS,NB_COLS), classes=12
        )
    else:
        raise ValueError("Unsupported model type: " + MODEL_TYPE)

    model.summary()
    print(model.summary())
    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.sgd())

    with open("models/" + MODEL_NAME + "/architecture.json", "w") as F:
        F.write(model.to_json())

    best_dev_acc = 0.0

    # build dev inputs once:
    print("-> building dev inputs once:")
    dev_inputs = []
    for idx, img in enumerate(dev_images):
        i = utils.augment_test_image(
            image=img, nb_rows=NB_ROWS, nb_cols=NB_COLS, nb_patches=NB_TEST_PATCHES
        )
        dev_inputs.append(i)

    tmp_train_X, tmp_train_Y = utils.augment_train_images(
        images=train_images,
        categories=train_Y,
        nb_rows=NB_ROWS,
        nb_cols=NB_COLS,
        nb_patches=NB_TRAIN_PATCHES,
    )

    model.fit(
        x=tmp_train_X, y=tmp_train_Y, batch_size=BATCH_SIZE, epochs=NB_EPOCHS, shuffle=True
    )

    dev_preds = []
    for inp in dev_inputs:
        pred = model.predict(inp, batch_size=BATCH_SIZE)
        pred = pred.mean(axis=0)
        dev_preds.append(np.argmax(pred, axis=0))

    # calculate accuracy:
    curr_acc = accuracy_score(dev_preds, dev_y_int)
    print("  curr val acc:", curr_acc)

    # save weights, if appropriate:
    if curr_acc > best_dev_acc:
        print("    -> saving model")
        model.save_weights("models/" + MODEL_NAME + "/weights.hdf5", overwrite=True)
        best_dev_acc = curr_acc


if __name__ == "__main__":
    train()
