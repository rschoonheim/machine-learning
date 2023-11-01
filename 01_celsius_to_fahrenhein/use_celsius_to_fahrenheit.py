import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Load the model
#
model = tf.keras.models.load_model('models/celsius_to_fahrenheit')

# Convert 10 degrees celsius to fahrenheit
#
print(model.predict([10.0]))

