import PIL
import tensorflow as tf
from keras import Sequential, layers

IMG_SIZE = (1920, 1080)

def resize_and_rescale(input_img_dataset, img_size):
    resize_and_rescale = Sequential([
        layers.Resizing(img_size[0], img_size[1]),
        layers.Rescaling(1./255)
    ])

    #resize and rescale dataset
    ds = input_img_dataset.map(lambda x, y: (resize_and_rescale(x), y), 
                num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds

#take image dataset, desired size and output augmented image dataset (tf.data.Dataset)
def perform_image_augmentation(input_img_dataset: tf.data.Dataset, img_size):
    #image augmentation pipeline
    augmentation_pipeline = Sequential([
        
    ])

    aug_img_dataset = input_img_dataset.map(lambda x, y: (augmentation_pipeline(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    return aug_img_dataset