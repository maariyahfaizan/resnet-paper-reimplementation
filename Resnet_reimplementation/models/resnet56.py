import numpy as np
import tensorflow as tf 

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Lambda

def residual_block(x, filters, stride=1):
    shortcut = x

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

    if stride != 1:

        # Step 1: Downsample the shortcut.
        # The shortcut must have the same height and width before
        # the tensors can be added together.
        #
        # The paper's Option A performs this by simply taking every
        # second pixel instead of using another convolution.
        #
        # Tensor slicing:
        #
        # t[:, ::2, ::2, :]

        shortcut = Lambda(
            lambda t: t[:, ::2, ::2, :]
        )(shortcut)
        # Step 2: Pad the channel dimension with zeros.
        #
        # After downsampling, the shortcut still has the old number
        # of channels.
        #
        # Example:
        #
        # Shortcut : 16 × 16 × 16
        # Main path: 16 × 16 × 32
        #
        # The tensors cannot be added because their channel counts
        # are different.
        #
        # Instead of learning a 1×1 convolution (Option B), the paper
        # simply pads the shortcut with zeros (Option A).
        shortcut = Lambda(
            lambda t: tf.pad(
                t,
                [
                    [0,0],
                    [0,0],
                    [0,0],
                    [filters // 4, filters // 4]
                ]
            )
        )(shortcut)

    x = Add()([x, shortcut])
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
    x = residual_block(x, filters=16)

# 2n layers with 32 filters
x = residual_block(x, filters=32, stride=2)
for i in range (n-1):
    x = residual_block(x, filters=32)

# 2n layers with 64 filters
x = residual_block(x, filters=64, stride=2)
for i in range (n-1):
    x = residual_block(x, filters=64)

x = GlobalAveragePooling2D()(x)
x = Dense(10, activation="softmax")(x)

model = Model(inputs=inputs, outputs=x)

