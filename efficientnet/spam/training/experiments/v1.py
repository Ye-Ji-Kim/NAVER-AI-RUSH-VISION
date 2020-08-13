from spam.spam_classifier.datasets.dataset import Dataset
from spam.spam_classifier.models.BasicModel import BasicModel
from spam.spam_classifier.networks.EfficientNet import frozen_efficientnet

input_size = (256, 256, 3)
classes = ['normal', 'monotone', 'screenshot', 'unknown']
config = {
    'model': BasicModel,
    'fit_kwargs': {
        'batch_size': 64,
        'epochs_finetune': 50,
        'epochs_full': 50,
        'debug': False
    },
    'model_kwargs': {
        'network_fn': frozen_efficientnet,
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
