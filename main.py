import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

model = keras.models.Sequential([
    keras.layers.Dense(32, input_dim = 784),
    keras.layers.Activation('relu'),
    keras.layers.Dense(10),
    keras.layers.Activation('softmax'),
])

rmsprop = keras.optimizers.RMSprop(learning_rate = 0.0001, rho = 0.9, epsilon = 1e-08, decay = 0.0)

model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs = 2, batch_size = 32)

loss, accuracy = model.evaluate(X_test, y_test)

# 預測(prediction)
input_x = X_test[0:10,:]
output_arr = model.predict(input_x)
predictions = np.argmax(output_arr, axis = -1)

print(predictions)

# 顯示 第一筆訓練資料的圖形，確認是否正確
plt.imshow(X_test[0].reshape(28,28))
print(X_test[3].reshape(28,28))
plt.savefig('output.png')
