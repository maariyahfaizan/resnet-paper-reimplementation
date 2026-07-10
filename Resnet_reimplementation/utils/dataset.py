import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import cifar10


def load_dataset():
    """
    Loads the CIFAR-10 dataset and computes the
    mean image from the training set.
    """

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # 45k / 5k split
    x_val = x_train[45000:]
    y_val = y_train[45000:]

    x_train = x_train[:45000]
    y_train = y_train[:45000]

    mean_image = np.mean(x_train, axis=0)

    return x_train, y_train, x_val, y_val, x_test, y_test, mean_image


def preprocess_train(image, label, mean_image):
    """
    Training preprocessing from the ResNet paper.
    """

    image = tf.cast(image, tf.float32)

    # Per-pixel mean subtraction
    image = image - mean_image

    # Pad 4 pixels on each side
    image = tf.pad(
        image,
        paddings=[[4, 4], [4, 4], [0, 0]] # isnt the imput image now 40 x 40 x 3? --> okay no it isnt cause we crop 32 x 32
    )

    # Random crop
    image = tf.image.random_crop(
        image,
        size=[32, 32, 3]
    )

    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)

    return image, label


def preprocess_test(image, label, mean_image):
    """
    Test preprocessing.
    """

    image = tf.cast(image, tf.float32)

    image = image - mean_image

    return image, label


def create_datasets(batch_size=128):

    x_train, y_train, x_val, y_val, x_test, y_test, mean_image = load_dataset()

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)
    )

    validation_dataset = tf.data.Dataset.from_tensor_slices(
    (x_val, y_val)
    )

    train_dataset = train_dataset.map(
        lambda image, label: preprocess_train(
            image,
            label,
            mean_image
        )
    )

    validation_dataset = validation_dataset.map(
        lambda image, label: preprocess_test(
            image,
            label,
            mean_image
        )
    )

    test_dataset = test_dataset.map(
        lambda image, label: preprocess_test(
            image,
            label,
            mean_image
        )
    )

    train_dataset = train_dataset.shuffle(50000)

    train_dataset = train_dataset.batch(batch_size)

    validation_dataset = validation_dataset.batch(batch_size)

    test_dataset = test_dataset.batch(batch_size)

    """ 

    with prefetch the next batch is ready 
    immeadiately after the current batch is trained 

    """
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE) 
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset
