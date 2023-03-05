import PIL
import tensorflow as tf
from keras import Sequential, layers
from keras.utils.image_dataset import 

IMG_SIZE = (1920, 1080)

#take image dataset, desired size and output augmented image dataset (tf.data.Dataset)
def perform_image_augmentation(input_img_dataset: tf.data.Dataset, img_size: tuple(int, int)):
    #image augmentation pipeline
    augmentation_pipeline = Sequential([
        layers.Resizing(IMG_SIZE[0], IMG_SIZE[1])
    ])

    aug_img_dataset = input_img_dataset.map(lambda x, y: augmentation_pipeline(x), y)

    return aug_img_dataset