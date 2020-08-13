from keras import Input, Model
from keras.applications.resnet_v2 import ResNet50V2
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from os import path
from keras.regularizers import l2


def frozen_resnet(input_size, n_classes, local_weights="/resnets/resnet50v2_notop.h5"):
    if local_weights and path.exists(local_weights):
        print(f'Using {local_weights} as local weights.')
        model_ = ResNet50V2(
            include_top=False,
            input_tensor=Input(shape=input_size),
            weights=local_weights)
    else:
        print(
            f'Could not find local weights {local_weights} for ResNet. Using remote weights.')
        model_ = ResNet50V2(
            include_top=False,
            input_tensor=Input(shape=input_size))

    #x = Flatten(input_shape=model_.output_shape[1:])(model_.layers[-1].output)
    x = model_.output
    x = GlobalAveragePooling2D()(x)
    #x = Dense(256, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    #x = Dense(256, activation='relu')(x)
    #x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)
    frozen_model = Model(model_.input, x)

    return frozen_model
