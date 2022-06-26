import os
from pyexpat import model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import numpy as np
import pandas as pd
from keras.utils.vis_utils import plot_model
from prepatation_data import featureV, x_my_test, y_my_test, my_test
from sklearn.metrics import confusion_matrix

model_loaded = keras.models.load_model('Save_model/Neural_network')

"""показ нейронной сети в виде текста и изображения
model_loaded.get_weights()
model_loaded.summary()
plot_model(model_loaded, 'Save_model/Neural_network.png', show_shapes=True)"""

pred = model_loaded.predict(x_my_test)
y_pred = np.argmax(pred, axis=1)
acc = [pred[0][1] if pred[0][1]>pred[0][0] else pred[0][0]]
y_p = ['Угроза' if i==1 else 'Норма' if i==0 else i for i in y_pred]
confusion_matrix(y_my_test, y_pred)
print(y_p[0], 'с вероятностью - ',round(acc[0]*100,2),'%' )