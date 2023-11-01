import tensorflow as tf
import numpy as np

# Create the training data
# -----------------------------------------------------------
#
#
work_logs = []
result = []

for hours in range(-200, 200):
    for wage in range(1, 25):
        work_logs.append([hours, wage])
        result.append(hours * wage)
#%%


# Build layer that will be executing the formula
# hours * wage = salary
# -----------------------------------------------------------
#
#
layer = tf.keras.layers.Dense(
    units=2,
    input_shape=[2]
)

# Build the model
# -----------------------------------------------------------
#
#
model = tf.keras.Sequential([layer])

# Compile the model
# -----------------------------------------------------------
#
#
model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(0.1)
)

# Train the model
# -----------------------------------------------------------
#
#
history = model.fit(
    work_logs,
    result,
    epochs=500,
)

# Use the model
# -----------------------------------------------------------
#
#
print(model.predict([[1, 10]]))





#%%
