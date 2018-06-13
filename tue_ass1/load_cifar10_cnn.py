import tensorflow as tf
from tensorflow.python.keras.models import load_model

# returns a compiled model
# Windows path, for Linux change \\ to /
model = load_model('./model.h5')

# print layers
for layer in model.layers:
  print(layer)

_, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype('float32')
x_test /= 256
num_classes = 10
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Score trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
