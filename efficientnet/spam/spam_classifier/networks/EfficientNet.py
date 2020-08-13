from keras import Input, Model
import efficientnet.keras as efn
from keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation, GlobalMaxPooling2D
from keras.applications import Xception

class SwishActivation(Activation):
    def __init__(self, activation, **kwargs):
        super(SwishActivation, self).__init__(activation, **kwargs)
        self.__name__ = 'swish_act'

def swish_act(x, beta=1):
    return (x*sigmoid(beta*x))

def frozen_efficientnet(input_size, n_classes):
    get_custom_objects().update({'swish_act': SwishActivation(swish_act)})

    model_ = efn.EfficientNetB3(
        include_top=False,
        input_tensor=Input(shape=input_size),
        weights='imagenet')


    for layer in model_.layers:
        layer.trainable = False

    x = model_.output
    x = GlobalAveragePooling2D()(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.2)(x)

    #x = Dense(512)(x)
    #x = BatchNormalization()(x)
    #x = Activation(swish_act)(x)
    #x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x)
    #x = BatchNormalization()(x)
    #x = Activation(swish_act)(x)
    x = Dropout(0.5)(x)
    # = Flatten(input_shape=model_.output_shape[1:])(model_.layers[-1].output)
    x = Dense(n_classes, activation='softmax')(x)

    frozen_model = Model(model_.input, x)

    return frozen_model
