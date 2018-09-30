from __future__ import print_function

import random
import shutil
import os
import pickle
import glob

import sys

from sklearn.metrics import accuracy_score

sys.setrecursionlimit(10000)

SEED = 1066987
import numpy as np

np.random.seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

from scipy.misc import imsave

import keras
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


NB_ROWS, NB_COLS = 150, 150
BATCH_SIZE = 30
NB_EPOCHS = 100
NB_TRAIN_PATCHES = 100
NB_TEST_PATCHES = 30
MODEL_NAME = "final"
MODEL_TYPE = "vgg16"  # resnet50 or vgg16

from keras.callbacks import ModelCheckpoint

CATEGORIES = [
    "caroline",
    "cursiva",
    "half_uncial",
    "humanistic",
    "humanistic_cursive",
    "hybrida",
    "praegothica",
    "semihybrida",
    "semitextualis",
    "southern_textualis",
    "textualis",
    "uncial",
]


def train(parent_dir, output_dir, epochs=20):
    # need to sort some into the validation dir
    csv_file = glob.glob(os.path.join(parent_dir, "*.csv"))[0]
    for line in open(csv_file):
        if "FILENAME" in line:
            continue

        filename, category_index, _period = line.split(";")
        category = CATEGORIES[int(category_index) - 1]
        directory = os.path.join(output_dir, category)

        os.makedirs(directory, exist_ok=True)
        src_path = os.path.join(parent_dir, filename)
        dest_path = os.path.join(directory, filename)
        try:
            os.symlink(src_path, dest_path)
        except FileExistsError:
            pass

    datagen = ImageDataGenerator()
    # need to make a training dataset
    train_generator = datagen.flow_from_directory(
        output_dir, color_mode="grayscale", follow_links=True
    )

    model = keras.applications.vgg16.VGG16(
        weights=None,
        include_top=True,
        input_shape=train_generator.image_shape,
        classes=12,
    )

    print(model.summary())
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"],
    )

    model.fit_generator(train_generator, epochs=epochs)

    model.save("model_output.keras")


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
