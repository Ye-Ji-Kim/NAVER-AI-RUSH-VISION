from spam.spam_classifier.datasets.dataset import Dataset
from spam.spam_classifier.models.BasicModel import BasicModel
from spam.spam_classifier.networks.resnet50 import frozen_resnet

input_size = (256, 256, 3)
classes = ['normal', 'monotone', 'screenshot', 'unknown']
config = {
    'model': BasicModel,
    'fit_kwargs': {
        'batch_size': 64,
        'epochs_finetune': 5,
        'epochs_full': 5,
        'debug': False
    },
    'model_kwargs': {
        'network_fn': frozen_resnet,
        'network_kwargs': {
            'input_size': input_size,
            'n_classes': len(classes)
        },
        'dataset_cls': Dataset,
        'dataset_kwargs': {
            'classes': classes,
            'input_size': input_size
        },
    },
}
