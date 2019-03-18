# Data Input Pipeline

This model uses the `tf.data` API as it is defined
in the TF 2.0 alpha docs to create an image dataset.

The paper modifies the data a bit before
feeding into the model:

    # Python pseudocode
    read_images(images)
    apply_random_hue_saturation_changes(images)
    apply_random_vertical_horizontal_flips(images)
    convert_to_hsv(images)
    add_grayscale_layer(images)
    
    
 Therefore, each image is a tensor of shape `(100,100,4)`,
 with 4 channels (H, S, V, grayscale).
 
 The `augment(image)` function in data_utils.py performs the
 data augmentations specified in the last 4 lines of the
 pseudocode. 