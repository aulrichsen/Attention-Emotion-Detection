import keras
import numpy as np

from numpy import loadtxt
from keras.models import load_model

modelN = load_model('C:/Users/Lewis/Documents/Uni_Work/FifthYear/ImageProcessing/Project/emotionNet_small_model')

X_fname = 'C:/Users/Lewis/Documents/Uni_Work/FifthYear/ImageProcessing/Project/X_test_privatetest6_100pct.npy'
y_fname = 'C:/Users/Lewis/Documents/Uni_Work/FifthYear/ImageProcessing/Project/y_test_privatetest6_100pct.npy'
X = np.load(X_fname)
y = np.load(y_fname)

score = modelN.evaluate(X, y, verbose=0)
print("model %s: %.2f%%" % (modelN.metrics_names[1], score[1]*100))