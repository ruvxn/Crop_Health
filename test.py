import tensorflow as tf

# Create a simple tensor and check device placement
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])
c = a * b

print("TensorFlow is running on:", c.device)
