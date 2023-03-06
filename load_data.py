import os 
from PIL import ImageOps
import tensorflow as tf
from keras.utils import load_img
import numpy as np

INPUT_DIR = "fire-detection-data/GRAFFITI-dataset/Level_I"
LABEL_DIR = "fire-detection-data/NIR-labelled-data/NIRlabeling"

def get_paths(input_dir, label_dir):
    #load input and label image paths
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith("RGNIR.png")
        ]
    )

    label_img_paths = sorted(
        [
            os.path.join(label_dir, fname)
            for fname in os.listdir(label_dir)
            if fname.endswith("fl.png")
        ]
    )

    #display all image paths
    all_image_paths = label_img_paths + input_img_paths
    # for path in all_image_paths: 
        # print("Loaded image path: {}".format(path))
    
    return input_img_paths, label_img_paths

#parses filenames and returns images
def _parse_function(input_fn, label_fn):
    image_string = tf.io.read_file(input_fn)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    image_decoded.set_shape([None, None, 3])
    image_decoded = tf.image.resize(images=image_decoded, size=[1088, 1920])
    image = tf.cast(image_decoded, tf.float32)

    label_string = tf.io.read_file(label_fn)
    label_decoded = tf.image.decode_png(label_string, channels=1)
    label_decoded.set_shape([None, None, 1])
    label_decoded = tf.image.resize(images=label_decoded, size=[1088, 1920])
    label = tf.cast(label_decoded, tf.uint8)

    return image, label 

#returns raw tf.data.Dataset from path lists
def load_dataset_from_paths(input_paths, label_paths): 
    input_filenames = tf.constant(input_paths)
    label_filenames = tf.constant(label_paths)

    temp_ds = tf.data.Dataset.from_tensor_slices((input_filenames, label_filenames))

    ds = temp_ds.map(_parse_function)

    return ds 

#split raw tf.data.Dataset into train, val, test sets
def get_dataset_split(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds