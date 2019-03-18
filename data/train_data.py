import tensorflow as tf
import pathlib
import random
from . import data_utils

# mysterious autotune parameter
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_root = pathlib.Path('../data/Training')
train_root = train_root.resolve()

print(f'Training data root is {train_root.absolute()}')


def create_zipped_set():
    # path of all images
    all_image_paths = list(train_root.glob('*/*'))
    # shuffle images
    random.shuffle(all_image_paths)
    # get all the training labels in order
    train_labels = [str(path.parent.name) for path in all_image_paths]
    # stringify all paths
    all_image_paths = [str(path) for path in all_image_paths]
    # count images
    image_count = len(all_image_paths)

    # get all labels in training set
    all_labels = set(train_labels)
    label_to_index = dict((name, index) for index, name in enumerate(all_labels))
    all_labels = [label_to_index[pathlib.Path(path).parent.name]
                  for path in all_image_paths]

    # build tensor from image paths
    path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    # build tensor from images
    image_ds = path_ds.map(data_utils.load_image, num_parallel_calls=AUTOTUNE)
    # build tensor from labels
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_labels, tf.int64))
    # zip the two together
    zipped = tf.data.Dataset.zip((image_ds, label_ds))
    # print some basic info
    print(f'There are {image_count} training images.')
    print(zipped)
    return zipped, image_count


def create_train_set(BATCH_SIZE):
    (data, count) = create_zipped_set()
    ds = data.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=count))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def get_train_count():
    return len(list(train_root.glob('*/*')))