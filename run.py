import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def normalize(array):
	return array / array.max()

def preprocess(x_train, x_test, y_train, y_test):
	#Changes the arrays to include all the example images, and the channel
	#The images are 28x28, with one color-channel
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) 
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
	
	# Float 32 is more accurate than the standard numpy ndarray
	# With very small gradients, the lack of accuracy will give us problems
	# Convert the data to float32 
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	
	# Normalizes the input data for the network
	# The range of any value in the image-matrix is now 0-1
	x_train = normalize(x_train)
	x_test = normalize(x_test)

	# Converts a class vector (integers) to binary class matrix
	# This is used to be able to use the loss-function "categorial_crossentropy"
	# Output data is divided into 10 classes, which represents the numbers 0-9
	num_classes = 10
	y_train = keras.utils.to_categorical(y_train, num_classes) 
	y_test = keras.utils.to_categorical(y_test, num_classes)

	return x_train, x_test, y_train, y_test

batch_size = 128 #Size of the batch to train, splits the data into chunks and trains on one chunk at a time
epochs = 5 #How many iterations should the neural net go over the whole dataset?

# input image dimensions
img_width, img_height = 28, 28

# Data handling
input_shape = (img_width, img_height, 1) #An image is 3-D: Width, height and color-channel. 

# Split the dataset into train and test-sets. Normally one uses train, test and validation to avoid selection-bias
(x_train, y_train), (x_test, y_test) = mnist.load_data() #Loads the mnist dataset
x_train, x_test, y_train, y_test = preprocess(x_train, x_test, y_train, y_test) #Preprocess



model = Sequential()
model.add(Conv2D(32, #Using a 'filter' size of 32, the convolution will produce an output space of the same dimensionality
				kernel_size=(3, 3), #Using a kernal size of 3x3
				activation='relu', 
				input_shape=input_shape)) #Convolution layer for 2D images
model.add(Conv2D(64, (3, 3), activation='relu')) #Convolution layer
model.add(MaxPooling2D(pool_size=(2, 2))) #Maxpooling, is analogus to how anti-aliasing works
model.add(Dropout(0.25)) #Hinton's dropout, kills some(25%) neurons output at random to avoid neurons co-adopting. This helps with overfitting
model.add(Flatten()) #Flattens the 3D input to a 2D vector output
model.add(Dense(128, activation='relu')) #Hidden fully-connected layer, uses rectified linnear unit for speed of computation
model.add(Dropout(0.5)) #Hinton's dropout 50% of neurons killed
model.add(Dense(10, activation='softmax')) #Output layer fully-connected, uses softmax for a probabilistic output

#Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(), #Adadelta, a better version of gradient descent than the vanilla version
              metrics=['accuracy']) #Meassure accuracy

#Run the training scheme provided, and validate the results on the test data
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) # 5 epochs gave a test accuracy of around 98 %