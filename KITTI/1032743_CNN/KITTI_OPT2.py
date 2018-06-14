from __future__ import print_function
import tensorflow as tf
from tensorflow.python import keras
#import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
import os
import h5py
from sklearn.metrics import classification_report
import numpy as np

batch_size = 32
num_classes = 8
epochs = 10
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'KITTI_trained_model_'
APPEND = 'OPT2'
model_name += APPEND +'.h5'

# load data

os.chdir('/')
os.chdir('content/drive/CNN/data/')
train_data_name = 'train.h5'
validation_data_name = 'validation.h5'


h5f = h5py.File(train_data_name,'r')
x_train = h5f['dataset_1'][:]
y_train = h5f['dataset_2'][:]
h5f.close()
h5f = h5py.File(validation_data_name,'r')
x_test = h5f['dataset_1'][:]
y_test = h5f['dataset_2'][:]
h5f.close()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# transform from [0, 255] to [0, 1)
x_train = x_train.astype('float32') / 256
x_test = x_test.astype('float32') / 256

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(73,102,3))) # specify the input_shape mannually
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes)) # logits, number of neurons equal to number of class
model.add(Activation('softmax'))

#TBoardCallback = keras.callbacks.TensorBoard(log_dir='~/Documents/KITTI_process/logs',histogram_freq=0,batch_size=32,write_graph=True,write_grads=False,write_images=False)
#TBoardCallback = TensorBoard(logdir='~/Documents/KITTI_process/,')

opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train, y_train,

                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
#model.save(model_path, overwrite=False)
#print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print(history.history.keys())

train_loss = history.history['loss']
train_acc  = history.history['acc']
val_loss   = history.history['val_loss']
val_acc    = history.history['val_acc']
xc         = range(epochs)

# acc, loss per epoch
print('loss=',train_loss)
print('acc=',train_acc)
print('val_loss=',val_loss)
print('val_acc=',val_acc)


Y_test = np.argmax(y_test, axis=1) # Convert one-hot to index
y_pred = model.predict_classes(x_test,batch_size=32, verbose = 1)
print(classification_report(Y_test, y_pred))