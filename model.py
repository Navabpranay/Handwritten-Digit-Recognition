from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

def build_fc_model():
    """
    Build a simple fully connected neural network model.
    """
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))  # Flatten 2D images to 1D vector
    model.add(Dense(128, activation='relu'))     # Hidden layer with 128 neurons
    model.add(Dense(10, activation='softmax'))   # Output layer for 10 digit classes
    return model

def build_cnn_model():
    """
    Build a convolutional neural network (CNN) model.
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())                          # Flatten for Dense layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))   # Output layer for 10 digit classes
    return model

#model = build_cnn_model()
#model.summary()
