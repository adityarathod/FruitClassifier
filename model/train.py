import tensorflow as tf
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from data import train_data
import model

batch_size = 300

dataset = train_data.create_train_set(batch_size)
count = train_data.get_train_count()

epoch_steps = tf.math.ceil(count / batch_size).numpy()


model = model.create_model()

model.fit(dataset, epochs=200, steps_per_epoch=epoch_steps)