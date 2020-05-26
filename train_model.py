import numpy as np
np.random.seed(123)  #for reproducibility

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
import glob as glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ptrain = glob.glob('training_data.npz')
ptest = glob.glob('test_data.npz')
for tr_file in ptrain:
    with np.load(tr_file) as train_data:
        X_train = train_data['train']
        Y_train = train_data['train_labels']
        
for tt_file in ptest:
    with np.load(tt_file) as test_data:
        X_test = test_data['train']
        Y_test = test_data['train_labels']

#preprocess the input data
X_train = np.array(X_train).reshape(X_train.shape[0], 32, 32, 3)
X_test = np.array(X_test).reshape(X_test.shape[0], 32, 32, 3)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /=255.0
X_test /=255.0

#Preporocess class labels
#convert 1-dimensional class arrays to 7-dimensional class matrices
Y_train = np_utils.to_categorical(Y_train, 43)
Y_test = np_utils.to_categorical(Y_test, 43)


#Declare a sequential model
model = Sequential()
model.add(Convolution2D(32, kernel_size=(3, 3),activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1)))
model.add(BatchNormalization())
model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Convolution2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))
#compile model
model.compile(loss = 'categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])
#Fit keras model on training data
history = model.fit(X_train, Y_train, batch_size = 32, validation_split = 0.2,  epochs = 10)
model.save("trafficsigns.h5")
#Evaluate keras model on test data
val = model.evaluate(X_test, Y_test, verbose = 0)

print ("Evaluation of test data: %s, %s" %(model.metrics_names[0],model.metrics_names[1]))
#print val

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
