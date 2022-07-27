import matplotlib.pyplot as plt
import tensorflow
import numpy

celsius = numpy.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
expected_fahrenheit = numpy.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

neural_layer = tensorflow.keras.layers.Dense(units=1, input_shape=[1])

# Units = quantity of layers
# input_shape = quantity of neurons in layer

model = tensorflow.keras.Sequential([neural_layer])

model.compile(
    optimizer=tensorflow.keras.optimizers.Adam(
        0.1),  # learning rate too low = slow learning but more accurate too fast = inaccurate
    loss='mean_squared_error'
)

print("\nTraining model...")

historial = model.fit(celsius, expected_fahrenheit, epochs=1000, verbose=False)

print("Model is trained")


plt.xlabel('Epoch number')
plt.ylabel("Loss magnitude")
plt.plot(historial.history['loss'])
plt.show()

print("First prediction: ")
print("model.predict([-20]) = ", model.predict([-20]))

# Internal Varibales
print(neural_layer.get_weights())
