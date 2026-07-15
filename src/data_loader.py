import tensorflow as tf
import os

def load_data(data_path, img_size, batch_size):

    # Data preprocessing(Crete configurations for the dataset)

    # Training Dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_path, "train"),
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical"
    )

    # Validation dataset
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_path, "val"),
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical"
    )

    #Testing set
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(data_path, "test"),
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="categorical"
    )

    return train_ds, val_ds,test_ds
