import matplotlib.pyplot as plt
import math
import tensorflow as tf
import tensorflow_datasets as tfds


# https://www.tensorflow.org/datasets/catalog/fashion_mnist by Zalando Research
target_dataset = 'fashion_mnist'

data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

print(data)

training_data = data['train']
testing_data = data['test']

# Normalize data ~ Accelerates the proccess
# Pass image pixel classification from 0-255 to 0-1


def normalize(image, tags):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, tags


training_data = training_data.map(normalize)
testing_data = testing_data.map(normalize)

# Send to cache

training_data = training_data.cache()
testing_data = testing_data.cache()


# Model

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),  # For classification
])


# Compile

model.compile(
    optimizer='Adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Reduce batch size

training_data_examples = metadata.splits['train'].num_examples
testing_data_examples = metadata.splits['test'].num_examples


BATCH_SIZE = 32

training_data = training_data.repeat().shuffle(
    training_data_examples).batch(BATCH_SIZE)
testing_data = testing_data.repeat().shuffle(
    testing_data_examples).batch(BATCH_SIZE)


historical = model.fit(training_data, epochs=10, steps_per_epoch=math.ceil(
    training_data_examples/BATCH_SIZE))


plt.xlabel('Epoch number')
plt.ylabel("Loss magnitude")
plt.plot(historical.history['loss'])
plt.show()


# Save model

model.save("clotches_model.h5")