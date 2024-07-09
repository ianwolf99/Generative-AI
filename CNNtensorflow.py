import tensorflow as tf 
from tensorflow.keras import datasets, layers ,models ,optimizers

#cifar_10
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS 32

#constant
BATCH_SIZE = 128
EPOCHS = 20
CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.20
OPTIM = tf.keras.optmizers.RMSprop()

#def the convnet
def build(input_shape, classes):
	model = models.Sequential()
	model.add(layers.Convolution2D(32, (3,3), activation='relu', input_shape=input_shape))
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))
	model.add(layers.Dropout(0.25))	
	model.add(layers.Flatten())
	model.add(layers.Dense(512, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(classes, activation='softmax'))
	return models

	

