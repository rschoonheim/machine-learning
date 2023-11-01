import tensorflow as tf
import numpy as np

# Configure tensorflow
#
tf.get_logger().setLevel('ERROR')


# Create the training data set.
#
celsius = np.array(range(-200, 200), dtype=float)
fahrenheit = np.array([], dtype=float)
for i in range(-200, 200):
    fahrenheit = np.append(fahrenheit, (i * 9 / 5) + 32)

layer_1 = tf.keras.layers.Dense(
    units=1,
    input_shape=[1],
    weights=[
        np.array([[1.8]], dtype=float),
        np.array([32.0], dtype=float)
    ]
)


# Build the model
#
model = tf.keras.Sequential([
    layer_1,
])


# Compile the model
#
model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(0.1)
)

# Train the model
#
history = model.fit(
    celsius,
    fahrenheit,
    epochs=500,
    verbose=False
)

# Save the model
#
model.save('models/celsius_to_fahrenheit')
