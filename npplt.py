import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

np.random.seed(1235)
x = np.linspace(-1, 1, 200)
np.random.shuffle(x)
y = np.random.normal(0.025, 0.01, 200)+0.2*x+1
X_train = x[:160]
Y_train = y[:160]
xtest = x[160:]
ytest = y[160:]

model = Sequential()
model.add(Dense(1, input_dim=1))
model.compile(loss='mse', optimizer='sgd')

for index in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if index % 100 == 0:
        print('train cost: '+str(cost))
pre = model.predict(xtest)
plt.scatter(X_train,Y_train,color='blue')
plt.scatter(xtest,ytest,color='orange')
plt.twinx()
plt.plot(xtest,pre,label='predict')
plt.show()
