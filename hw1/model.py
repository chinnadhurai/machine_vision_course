'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import numpy
import random
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping


batch_size = 256
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


X2_train = X_train.copy()
y2_train = y_train.copy()

count = 0
for i in range(X2_train.shape[0]):
    for j in range(X2_train[i].shape[0]):
        X2_train[i][j] = numpy.fliplr(X2_train[i][j])
        count+=1

print(count)
print("done")

X_train = numpy.concatenate((X_train, X2_train))
y_train = numpy.concatenate((y_train, y2_train))


ind = numpy.arange(X_train.shape[0])
random.shuffle(ind)
X_train = X_train[ind]
y_train = y_train[ind]

print(X_train.shape)
print(y_train.shape)


X_valid = X_train[0:5000,:,:]
y_valid = y_train[0:5000,:]

X_train = X_train[5000:,:,:]
y_train = y_train[5000:,:]


print('X_train shape:', X_train.shape)
print('Y_train shape:', y_train.shape)
print('X_valid shape:', X_valid.shape)
print('Y_valid shape:', y_valid.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', y_test.shape)

print(X_train.shape[0], 'train samples')
print(X_valid.shape[0], 'valid samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(1024, 4, 4, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(1024, 1, 1, border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(nb_classes, 1, 1, border_mode='valid'))
#model.add(BatchNormalization())
model.add(Flatten())
#model.add(Dense(nb_classes))
model.add(Activation('softmax'))



# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
#X_train /= 255
#X_valid /= 255
#X_test /= 255

#print('X_train shape:', X_train.shape)



checkpointer = ModelCheckpoint(filepath="/data/lisatmp4/chinna/data/ift6268/temp/2/weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
c2 = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='max')


if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train, batch_size=batch_size,
              nb_epoch=nb_epoch, show_accuracy=True,
              validation_data=(X_test, Y_test), shuffle=True)
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    vdatagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    vdatagen.fit(X_valid)


    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch, show_accuracy=True,
                        validation_data=vdatagen.flow(X_valid, Y_valid, batch_size=X_valid.shape[0]),
                        nb_val_samples=X_valid.shape[0],
                        nb_worker=1,
                        callbacks=[checkpointer,c2])
