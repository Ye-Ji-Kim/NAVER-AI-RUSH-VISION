import os
from pathlib import Path
import shutil
from tempfile import mkdtemp
from typing import Tuple
from warnings import warn
from PIL import Image

from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.resnet_v2 import preprocess_input
import pandas as pd
import numpy as np
from nsml.constants import DATASET_PATH
from random import *


class EmptyContentError(Exception):
    pass


UNLABELED = -1


class Dataset:
    """
    Basic dataset that can be used in combination with Keras fit_generator.
    Reorders the data to have one folder per class.
    """

    def __init__(self, classes, input_size):
        self.classes = classes
        self.img_size = input_size
        self.base_dir = Path(mkdtemp())
        self._len = None
        self.validation_fraction = 0.2

    def __del__(self):
        """
        Deletes the temporary folder that we created for the dataset.
        """
        shutil.rmtree(self.base_dir)

    def train_val_gen(self, batch_size: int):
        """
        Splits the train_data folder into train/val generators. Applies some image augmentation for the train dataset.

        Args:
            batch_size: int

        Returns:
            train_generator: Keras data generator.
            val_generator: Keras data generator.
        """
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            horizontal_flip=True,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            validation_split=self.validation_fraction
        )

        train_generator = train_datagen.flow_from_directory(
            directory=self.base_dir / 'train',
            shuffle=True,
            batch_size=batch_size,
            target_size=self.img_size[:-1],
            classes=self.classes,
            subset='training')

        val_generator = train_datagen.flow_from_directory(
            directory=self.base_dir / 'train',
            batch_size=batch_size,
            target_size=self.img_size[:-1],
            classes=self.classes,
            shuffle=True,
            subset='validation')
        assert self.classes == list(iter(train_generator.class_indices))

        return train_generator, val_generator

    def test_gen(self, test_dir: str, batch_size: int):
        """
        Note that the test dataset is not rearranged.

        Args:
            test_dir: Path to the test dataseet.
            batch_size: Number of examples per batch. Reduce if encountering memory issues.

        Returns:
            gen: Keras generator for the test dataset.
            files: [str]
                A list of files. These are the same order as the images returned from the generator.

        """
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        files = [str(p.name) for p in (Path(test_dir) / 'test_data').glob('*.*') if p.suffix not in ['.gif', '.GIF']]
        metadata = pd.DataFrame({'filename': files})
        gen = datagen.flow_from_dataframe(metadata, directory=f'{test_dir}/test_data', x_col='filename',
                                          class_mode=None, shuffle=False, batch_size=batch_size)
        return gen, files

    def len(self, dataset):
        """
        Utility function to compute the number of datapoints in a given dataset.
        """
        if self._len is None:
            self._len = {
                dataset: sum([len(files) for r, d, files in os.walk(self.base_dir / dataset)]) for dataset in
                ['train']}
            self._len['train'] = int(self._len['train'] * (1 - self.validation_fraction))
            self._len['val'] = int(self._len['train'] * self.validation_fraction)
        return self._len[dataset]

    def prepare(self):
        """
        The resulting folder structure is compatible with the Keras function that generates a dataset from folders.
        """
        dataset = 'train'
        self._initialize_directory(dataset)
        self._rearrange(dataset)

    def _initialize_directory(self, dataset: str) -> None:
        """
        Initialized directory structure for a given dataset, in a way so that it's compatible with the Keras dataloader.
        """
        dataset_path = self.base_dir / dataset
        dataset_path.mkdir()
        for c in self.classes:
            (dataset_path / c).mkdir()

    def _rearrange(self, dataset: str) -> None:
        """
        Then rearranges the files based on the attached metadata. The resulting format is
        --
         |-train
             |-normal
                 |-img0
                 |-img1
                 ...
             |-montone
                 ...
             |-screenshot
                 ...
             |_unknown
                 ...
        """
        output_dir = self.base_dir / dataset
        src_dir = Path(DATASET_PATH) / dataset
        metadata = pd.read_csv(src_dir / f'{dataset}_label')
        for _, row in metadata.iterrows():
            if row['annotation'] == UNLABELED:
                continue
            #if row['annotation'] == 0:
            #    if random() < 0.4:
            #        continue
            src = src_dir / 'train_data' / row['filename']
            if not src.exists():
                raise FileNotFoundError
            dst = output_dir / self.classes[row['annotation']] / row['filename']
            if dst.exists():
                warn(f'File {src} already exists, this should not happen. Please notify 서동필 or 방지환.')
            else:
                shutil.copy(src=src, dst=dst)
    def get_unlabeled_data(self, dataset):
        src_dir = Path(DATASET_PATH) / dataset
        metadata = pd.read_csv(src_dir / f'{dataset}_label')
        src_dir_list = []
        img_list = []
        for _, row in metadata.iterrows():
            if row['annotation'] == UNLABELED:
                src = src_dir / 'train_data' / row['filename']
                if not src.exists():
                    raise FileNotFoundError
                src_dir_list.append(src)
        for f_dir in src_dir_list:
            img = image.load_img(f_dir, target_size = self.img_size[:-1])
            img = image.img_to_array(img)
            img = preprocess_input(img)
            img_list.append(img)
        return img_list, src_dir_list

    def save_unlabeled_data(self, dataset, src, class_idx):
        output_dir = self.base_dir / dataset
        dst = output_dir / self.classes[class_idx] / str(src).split('/')[-1]
        shutil.copy(src=src, dst=dst)
    def show_data_size(self, dataset):
        dataset_path = self.base_dir / dataset
        for i in range(4):
            class_name = self.classes[i]
            class_dir = dataset_path / class_name
            count = len(next(os.walk(class_dir))[2])
            print(f"{class_name} have {count} data")

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        input_img = preprocess_input(input_img)
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser