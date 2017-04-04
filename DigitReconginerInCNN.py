import pandas as pd
import numpy as np
import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils
from keras import backend as K
K.set_image_dim_ordering('th')
# Read competition data files:
train = pd.read_csv("train_csv.csv").values
test  = pd.read_csv("test_csv.csv").values

nb_epoch = 10 # Change to 100

batch_size = 16
img_rows, img_cols = 48, 48

nb_filters_1 = 32 # 64
nb_filters_2 = 64 # 128
nb_filters_3 = 128 # 256
nb_conv = 3

trainX = train[0:20, 1:].reshape(train[0:20,1:].shape[0], 1, img_rows, img_cols)
trainX = trainX.astype(float)
trainX /= 255.0 # preprocess the data

trainY = kutils.to_categorical(train[0:20, 0])
nb_classes = trainY.shape[1]

cnn = models.Sequential()

cnn.add(conv.ZeroPadding2D((1,1), input_shape=(1, 48, 48),))
cnn.add(conv.Convolution2D(32, 3,3,  activation="relu"))
cnn.add(conv.ZeroPadding2D((1, 1)))
cnn.add(conv.Convolution2D(32, 3, 3, activation="relu"))
cnn.add(conv.MaxPooling2D(strides=(2,2)))

cnn.add(conv.ZeroPadding2D((1, 1)))
cnn.add(conv.Convolution2D(64, 3, 3, activation="relu"))
cnn.add(conv.ZeroPadding2D((1, 1)))
cnn.add(conv.Convolution2D(64, 3, 3, activation="relu"))
cnn.add(conv.MaxPooling2D(strides=(2,2)))

# cnn.add(conv.ZeroPadding2D((1, 1)))
# cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu"))
# cnn.add(conv.ZeroPadding2D((1, 1)))
# cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu"))
# cnn.add(conv.ZeroPadding2D((1, 1)))
# cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu"))
# cnn.add(conv.ZeroPadding2D((1, 1)))
# cnn.add(conv.Convolution2D(nb_filters_3, nb_conv, nb_conv, activation="relu"))
# cnn.add(conv.MaxPooling2D(strides=(2,2)))

cnn.add(core.Flatten())
cnn.add(core.Dropout(0.2))
cnn.add(core.Dense(128, activation="relu")) # 4096
cnn.add(core.Dense(nb_classes, activation="softmax"))

cnn.summary()
cnn.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

cnn.fit(trainX, trainY, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

testX = test[:, 1:].reshape(test.shape[0], 1, img_rows, img_cols)
testX = testX.astype(float)
testX /= 255.0

yPred = cnn.predict_classes(testX)

np.savetxt('cnn_python.csv', np.c_[range(1,len(yPred)+1),yPred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')

scores = cnn.evaluate(trainX, trainY, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




# serialize model to JSON

model_json = cnn.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))




