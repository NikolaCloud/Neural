import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error)
from prepatation_data import Y_test, y_21_test, x_21_test
import keras

model_loaded = keras.models.load_model('Save_model/Neural_network')
y_pred = np.load('Save_model/y_pred.npy')

confusion_matrix(Y_test, y_pred)
accuracy = accuracy_score(Y_test, y_pred)*100
print(accuracy)
print(y_pred)
print(y_21_test)

pred = model_loaded.predict(x_21_test)
y_pred = np.argmax(pred, axis=1)
confusion_matrix(y_21_test, y_pred)
print(y_pred)

acc_21 = accuracy_score(y_21_test, y_pred)*100
print(acc_21)

recall = recall_score(y_21_test, y_pred, average="binary")
precision = precision_score(y_21_test, y_pred, average="binary")
f1 = f1_score(y_21_test, y_pred, average="binary")
print("F-Score: ", f1*100)
print("Precision: ", precision*100)
print("Recall: ", recall*100)
print("Accuracy: ", acc_21)