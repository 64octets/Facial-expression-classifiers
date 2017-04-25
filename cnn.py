import pandas as pd
import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils
from keras import backend as K
from keras.models import model_from_json

K.set_image_dim_ordering('th')
# Read competition data files:
train = pd.read_csv("/Users/shalachen/Dropbox/SYDE675/finalProject/train_all.csv").values
test  = pd.read_csv("/Users/shalachen/Dropbox/SYDE675/finalProject/test_all.csv").values

nb_epoch = 20 # Change to 100
batch_size = 256
img_rows, img_cols = 48, 48
nb_filters_1 = 32 # 64
nb_filters_2 = 64 # 128
nb_filters_3 = 128 # 256
kernel_size = 5

trainX = train[:, 1:].reshape(train.shape[0], 1, img_rows, img_cols)
trainX = trainX.astype(float)
trainX /= 255.0 # preprocess the data
trainY = kutils.to_categorical(train[:, 0])
nb_classes = trainY.shape[1]

cnn = models.Sequential()

cnn.add(conv.ZeroPadding2D((1,1), input_shape=(1, img_rows, img_cols)))
cnn.add(conv.Convolution2D(nb_filters_2, kernel_size, kernel_size, activation="relu"))
cnn.add(conv.Convolution2D(nb_filters_2, kernel_size, kernel_size, activation="relu"))
cnn.add(conv.MaxPooling2D(strides=(2,2)))
#cnn.add(conv.ZeroPadding2D((1, 1)))
cnn.add(conv.Convolution2D(nb_filters_2, kernel_size, kernel_size, activation="relu"))
cnn.add(conv.Convolution2D(nb_filters_2, kernel_size, kernel_size, activation="relu"))
cnn.add(conv.MaxPooling2D(strides=(2,2)))

cnn.add(core.Flatten())
cnn.add(core.Dropout(0.2))
cnn.add(core.Dense(nb_filters_3, activation="relu")) # 4096
cnn.add(core.Dense(nb_classes, activation="softmax"))

cnn.summary()
cnn.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

cnn.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
##
testX = test[:, 1:].reshape(test.shape[0], 1, 48, 48)
testX = testX.astype(float)
testX /= 255.0
testY = kutils.to_categorical(test[:, 0])


yPred = cnn.predict_classes(testX)

np.savetxt('cnn_python.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

scores = cnn.evaluate(trainX, trainY, verbose=1)
print("%s: %.2f%%" % (cnn.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = cnn.to_json()
with open("model3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
cnn.save_weights("model.h5")
print("Saved model to disk")
 
# later...
# load json and create model
# json_file = open('model3.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
 
# evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# cnn.add(conv.ZeroPadding2D((1, 1)))
# cnn.add(conv.Convolution2D(nb_filters_3, kernel_size, kernel_size, activation="relu"))
# cnn.add(conv.ZeroPadding2D((1, 1)))
# cnn.add(conv.Convolution2D(nb_filters_3, kernel_size, kernel_size, activation="relu"))
# cnn.add(conv.ZeroPadding2D((1, 1)))
# cnn.add(conv.Convolution2D(nb_filters_3, kernel_size, kernel_size, activation="relu"))
# cnn.add(conv.ZeroPadding2D((1, 1)))
# cnn.add(conv.Convolution2D(nb_filters_3, kernel_size, kernel_size, activation="relu"))
# cnn.add(conv.MaxPooling2D(strides=(2,2)))
