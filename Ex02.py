import numpy as np
from keras.models import Sequential
from keras import layers
from keras import activations
from keras import optimizers
from keras import losses
from keras import metrics
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# New model
model = Sequential()

# Input Layer
input_layer = layers.Dense(
    input_dim=1,
    units=1,
    activation=activations.linear
)
model.add(input_layer)

# Compile
model.compile(
    optimizer=optimizers.Adam(),
    loss=losses.mean_squared_error,
    metrics=[metrics.mean_squared_error]
)

model.summary()

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

earlyStopping = EarlyStopping(monitor='loss', min_delta=1e-10, patience=5, verbose=1, mode='auto')
history = model.fit(
    X, y,
    epochs=1000,
    verbose=1,
    batch_size=128,
    shuffle=True,
    callbacks=[earlyStopping]
)
print("Layer 1 weight thetas: ", input_layer.get_weights()[0])
print("Layer 1 weight biases: ", input_layer.get_weights()[1])
print("y = %f * x + %f" % (input_layer.get_weights()[0][0][0], input_layer.get_weights()[1][0]))

print("Predicts: ", model.predict(np.array([[1000]])))

plt.plot(history.history['loss'])
plt.plot(history.history['mean_squared_error'])
plt.title('Model training history')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
