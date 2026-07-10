import numpy as np
import tensorflow as tf 

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2


def plain_block(x, filters, stride=1):
    x = Conv2D(
            filters=filters,
            kernel_size=(3,3),
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4)
    )(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(
            filters=filters,
            kernel_size=(3,3),
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4)
    )(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x
   

n = 9
inputs = Input(shape=(32, 32, 3))

# the model
x = Conv2D(
        filters=16, 
        kernel_size=(3,3), 
        padding="same", 
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-4)
    )(inputs)

x = BatchNormalization()(x)
x = ReLU()(x)

# 2n layers with 16 filters 
for i in range (n):
    x = plain_block(x, filters=16)

# 2n layers with 32 filters
x = plain_block(x, filters=32, stride=2)
for i in range (n-1):
    x = plain_block(x, filters=32)

# 2n layers with 64 filters
x = plain_block(x, filters=64, stride=2)
for i in range (n-1):
    x = plain_block(x, filters=64)

x = GlobalAveragePooling2D()(x)
x = Dense(10, activation="softmax")(x)

model = Model(inputs=inputs, outputs=x)


