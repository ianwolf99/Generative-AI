import tensorflow as tensorflow
from tensorflow.keras import datasets, layers, models, regularizers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as numpy

EPOCHS=50
NUM_CLASSES = 10

def build_model():
 	model = models.Sequential()

 	# 1st block
 	model.add(layers.Conv2D(32, (3,3), padding='same',
 	input_shape=x_train.shape[1:], activation='relu'))
 	model.add(layers.BatchNormalization())
 	model.add(layers.Conv2D(32, (3,3), padding='same', activation='relu'))
 	model.add(layers.BatchNormalization())
 	model.add(layers.MaxPooling2D(pool_size=(2,2)))
 	model.add(layers.Dropout(0.2))
 	# 2nd block
 	model.add(layers.Conv2D(64, (3,3), padding='same',activation='relu'))
 	model.add(layers.BatchNormalization())
 	model.add(layers.Conv2D(64, (3,3), padding='same',activation='relu'))
 	model.add(layers.BatchNormalization())
 	model.add(layers.MaxPooling2D(pool_size=(2,2)))
 	model.add(layers.Dropout(0.3))
 	# 3d block
 	model.add(layers.Conv2D(128, (3,3), padding='same',activation='relu'))
 	model.add(layers.BatchNormalization())
 	model.add(layers.Conv2D(128, (3,3), padding='same',activation='relu'))
 	model.add(layers.BatchNormalization())
 	model.add(layers.MaxPooling2D(pool_size=(2,2)))
 	model.add(layers.Dropout(0.4))
 	# dense
 	model.add(layers.Flatten())
 	model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
 	return model
 	model.summary()

def load_data():
	(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')

	#normalize
	mean = np.mean(x_train, axis=(0,1,2,3))
	std = np.std(x_train, axis=(0,1,2,3))
	x_train = (x_train-mean)/(std+1e-7)
	x_test = (x_test-mean)/(std+1e-7)

	y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
	y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

	return x_train, y_train, x_test, y_test

#we need part to train the network
(x_train, y_train, x_test, y_test) = load_data()
model = build_model()
model.compile(loss='catergorical_crossentropy', optimizers='RMSprop', metrics=['accuracy']
#train
'''
batch_size = 64
model.fit(x_train, y_train, batch_size=batch_size, epochs=EPOCHS, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest Score:", score[0])
print("Test acuracy:", score[1])
'''
#image augmentation
datagen = ImageDataGenerator(
	rotation_range=30,
 	width_shift_range=0.2,
 	height_shift_range=0.2,
 	horizontal_flip=True,
 	)
datagen.fit(x_train)
# train
batch_size = 64
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), epochs=EPOCHS, verbose=1, validation_data=(x_test,y_test))
# save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
	json_file.write(model_json)
model.save_weights('model.h5')
# test
scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (scores[1]*100,scores[0])) 