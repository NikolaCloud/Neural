from matplotlib.pyplot import axis
#from prepatation_data import *
from neural_network import model
from prepatation_data import x_train, Y_train, x_test, x_my_test
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

Y_train.to_csv('Save_model/Y_train.csv')
history = model.fit(x_train, Y_train, epochs=100, batch_size=128)

pred = model.predict(x_test)
y_pred = np.argmax(pred, axis=1)

np.save('Save_model/y_pred', y_pred)

y_p = ['Malignant' if i==1 else 'Benign' if i==0 else i for i in y_pred]

output = pd.concat([pd.DataFrame(y_p), pd.DataFrame(y_pred)], axis=1)
output.to_csv('Save_model/pred.csv')

model.save('Save_model/Neural_network')