from load_data import load_dataset_from_paths, get_dataset_split, get_paths, INPUT_DIR, LABEL_DIR
from img_aug import perform_image_augmentation, resize_and_rescale
import tensorflow as tf 
import keras
from keras import layers
from keras.utils import load_img
from model import get_model
import matplotlib.pyplot as plt
from PIL import ImageOps, Image
import numpy as np


BATCH_SIZE = 1

def encode_label(label_img):
    img = tf.squeeze(label_img, axis=-1)
    one_hot = tf.one_hot(img, 2, dtype=tf.uint8)
    print(one_hot)
    return one_hot

def prepare(ds, shuffle=False, augment=False):
  #resize and rescale all datasets
  ds = resize_and_rescale(ds, img_size=(1088, 1920))

  if shuffle:
    ds = ds.shuffle(1000)

  # Batch all datasets.
  ds = ds.batch(BATCH_SIZE)

  # Use data augmentation only on the training set.
  if augment:
    ds = perform_image_augmentation(ds, img_size=(1088, 1920))

  #one hot encode labels
  ds = ds.map(lambda x, y: (x, encode_label(y)))

  # Use buffered prefetching on all datasets.
  return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

def train():
    keras.backend.clear_session()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #load dataset
    input_paths, label_paths = get_paths(INPUT_DIR, LABEL_DIR)
    dataset = load_dataset_from_paths(input_paths, label_paths)
    print("Paths loaded!")

    #train-test split
    train_dataset, val_dataset, test_dataset = get_dataset_split(dataset, dataset.cardinality().numpy(), train_split=0.6, val_split=0.2, test_split=0.2)
    print("Train-test split loaded!")
    
    #perform dataset preparation: shuffling, batching, augmentation
    train_dataset = prepare(train_dataset, shuffle=True, augment=True)
    val_dataset = prepare(val_dataset)
    test_dataset = prepare(test_dataset)

    model = get_model(img_size=(1088, 1920), num_classes=2)
    # model.summary()

    optimizer = tf.optimizers.SGD(momentum=0.99)
    loss = keras.losses.BinaryFocalCrossentropy()

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    epochs = 20
    history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, steps_per_epoch=1)

    plt.plot(history.history["loss"])
    plt.title("Training Loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(history.history["accuracy"])
    plt.title("Training Accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(history.history["val_loss"])
    plt.title("Validation Loss")
    plt.ylabel("val_loss")
    plt.xlabel("epoch")
    plt.show()

    plt.plot(history.history["val_accuracy"])
    plt.title("Validation Accuracy")
    plt.ylabel("val_accuracy")
    plt.xlabel("epoch")
    plt.show()

    test_preds = model.predict(test_dataset)
      
        
train()