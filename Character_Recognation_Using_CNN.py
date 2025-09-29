from tensorflow.keras import layers,models
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1,28,28,1)).astype('float32')/255
x_test = x_test.reshape((-1,28,28,1)).astype('float32')/255

# One-hot encode the labels
lb=LabelBinarizer()
y_train=lb.fit_transform(y_train)
y_test=lb.transform(y_test)

# Model creation
model=models.sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(200,activation='relu'),
    layers.dense(10,activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))