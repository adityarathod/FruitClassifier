import tensorflow as tf


def load_image(path):
    print(path)
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [100, 100])
    return augment(image)


# augment(image) and create_image_tensor(image) are copies of the code from the original paper authors
# The symbols have been changed.
# See git.io/fjvgW.
def augment(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_hue(image, 0.02)
    image = tf.image.random_hue(image, 0.02)
    image = tf.image.random_saturation(image, 0.9, 1.2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return create_image_tensor(image)


def create_image_tensor(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    grayscale = tf.image.rgb_to_grayscale(image)
    hsv = tf.image.rgb_to_hsv(image)
    res = tf.concat([hsv, grayscale], 2)
    return res
