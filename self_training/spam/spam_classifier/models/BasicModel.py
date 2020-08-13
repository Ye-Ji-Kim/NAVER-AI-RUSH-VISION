from typing import Callable, List

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
import nsml
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

from spam.spam_classifier.datasets.dataset import Dataset
from spam.spam_classifier.models.utils import Metrics, NSMLReportCallback, evaluate
from spam.spam_classifier.networks.resnet50 import frozen_resnet

global model_dir

class BasicModel:
    """
    A basic model that first finetunes the last layer of a pre-trained network, and then unfreezes all layers and
    train them.
    """

    def __init__(self, network_fn: Callable, dataset_cls: Dataset, dataset_kwargs, network_kwargs):
        self.data: Dataset = dataset_cls(**kwargs_or_empty_dict(dataset_kwargs))
        self.network: keras.Model = network_fn(**kwargs_or_empty_dict(network_kwargs))
        self.debug = False

    def fit(self, epochs_finetune, epochs_full, batch_size, debug=False):
        self.debug = debug
        #self.data.prepare()
        #self.network = multi_gpu_model(self.network, gpus=4)
        self.network.compile(
            loss=self.loss(),
            optimizer=self.optimizer('finetune'),
            metrics=self.fit_metrics()
        )

        steps_per_epoch_train = int(self.data.len('train') / batch_size) if not self.debug else 2
        model_path_finetune = 'model_finetuned.h5'
        train_gen, val_gen = self.data.train_val_gen(batch_size)
        nsml.save(checkpoint='best')
        #class_weights = class_weight.compute_class_weight(
        #    'balanced',
        #    np.unique(train_gen.classes),
        #    train_gen.classes)
        #class_weights = [1, 43.6, 25.3, 5.46]
        self.network.fit_generator(generator=train_gen,
                                   steps_per_epoch=steps_per_epoch_train,
                                   epochs=epochs_finetune,
                                   callbacks=self.callbacks(
                                       model_path=model_path_finetune,
                                       model_prefix='last_layer_tuning',
                                       patience=5,
                                       val_gen=val_gen,
                                       classes=self.data.classes),
                                   validation_data=val_gen,
                                   use_multiprocessing=True,
                                   workers=20,
                                   #class_weight=class_weights
                                   )  # TODO change to be dependent on n_cpus

        self.network.load_weights(model_path_finetune)
        self.unfreeze()

        self.network.compile(
            loss=self.loss(),
            optimizer=self.optimizer('full'),
            metrics=self.fit_metrics()
        )

        model_path_full = 'model_full.h5'
        self.network.fit_generator(generator=train_gen,
                                   steps_per_epoch=steps_per_epoch_train,
                                   epochs=epochs_full,
                                   callbacks=self.callbacks(
                                       model_path=model_path_full,
                                       model_prefix='full_tuning',
                                       val_gen=val_gen,
                                       patience=10,
                                       classes=self.data.classes),
                                   validation_data=val_gen,
                                   use_multiprocessing=True,
                                   workers=20,
                                   )

        self.network.load_weights(model_path_full)
        nsml.save(checkpoint='best')
        print('Done')
        self.metrics(gen=val_gen)

    def unfreeze(self) -> None:
        for layer in self.network.layers[19:]:
            layer.trainable = True

    def loss(self) -> str:
        loss = keras.losses.CategoricalCrossentropy()
        return loss

    def optimizer(self, stage: str) -> keras.optimizers.Optimizer:
        return {
            'finetune': Adam(lr=1e-4),
            'full': SGD(lr=1e-5, momentum=0.9, nesterov=True)
        }[stage]

    def fit_metrics(self) -> List[str]:
        return ['accuracy']

    def callbacks(self, model_path, model_prefix, patience, classes, val_gen):
        callbacks = [
            # TODO Change to the score we're using for ModelCheckpoint
            ReduceLROnPlateau(patience=3),  # TODO Change to cyclic LR
            NSMLReportCallback(prefix=model_prefix),
            Metrics(name=model_prefix,
                    classes=classes,
                    val_data=val_gen,
                    n_val_samples=self.data.len('val') if not self.debug else 256),
            ModelCheckpoint(model_path, monitor=f'val/{model_prefix}/macro avg/f1-score', verbose=1,
                            save_best_only=True, mode='max'),
            # TODO Change to the score we're using for ModelCheckpoint
            EarlyStopping(patience=patience)  # EarlyStopping needs to be placed last, due to a bug fixed in tf2.2
        ]
        return callbacks

    def evaluate(self, test_dir: str) -> pd.DataFrame:
        """

        Args:
            test_dir: Path to the test dataset.

        Returns:
            ret: A dataframe with the columns filename and y_pred. One row is the prediction (y_pred)
                for that file (filename). It is important that this format is used for NSML to be able to evaluate
                the model for the leaderboard.

        """
        gen, filenames = self.data.test_gen(test_dir=test_dir, batch_size=64)
        y_pred = self.network.predict_generator(gen)
        ret = pd.DataFrame({'filename': filenames, 'y_pred': np.argmax(y_pred, axis=1)})
        return ret

    def metrics(self, gen) -> None:
        """
        Generate and print metrics.

        Args:
            gen: Keras generator for which to get metrics
            n_batches: How many batches that can be fetched from the data generator.
        """
        y_true, y_pred = evaluate(data_gen=gen, model=self.network)
        y_true, y_pred = [np.argmax(y, axis=1) for y in [y_true, y_pred]]

        cls_report = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            output_dict=True,
            target_names=self.data.classes,
            labels=np.arange(len(self.data.classes))
        )
        print(
            f'Classification report for validation dataset:\n-----------------------------\n{cls_report}\n=============\n')

    def self_training(self):
        self.data.prepare()
        dataset = 'train'
        img_list, src_dir_list = self.data.get_unlabeled_data(dataset)
        nsml.load(checkpoint='best', session='Ye-Ji-Kim/spam-3/87')

        for idx, img in enumerate(img_list):
            img = np.array([img])
            predict =self.network.predict(img)
            predict = predict[0]
            if np.max(predict) > 0.90:
                if np.argmax(predict) != 0:
                    self.data.save_unlabeled_data(dataset, src_dir_list[idx], np.argmax(predict))
                    print(f'save in {np.argmax(predict)}')
        self.data.show_data_size(dataset)

    def initialize_model(self, network_fn: Callable, dataset_cls: Dataset, dataset_kwargs, network_kwargs):
        self.network: keras.Model = network_fn(**kwargs_or_empty_dict(network_kwargs))


def bind_model(model: BasicModel):
    """
    Utility function to make the model work with leaderboard submission.
    """

    def load(dirname, **kwargs):
        model.network.load_weights(f'{dirname}/model')
        return model

    def save(dirname, **kwargs):
        filename = f'{dirname}/model'
        print(f'Trying to save to {filename}')
        global model_dir
        model_dir = filename
        model.network.save_weights(filename)

    def infer(test_dir, **kwargs):
        return model.evaluate(test_dir)

    nsml.bind(load=load, save=save, infer=infer)


def kwargs_or_empty_dict(kwargs):
    if kwargs is None:
        kwargs = {}
    return kwargs
