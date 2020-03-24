import numpy as np
from keras.models import load_model
import cv2


class EmotionNetClass:
    def __init__(self, img):
        #load the model which is local save of google colab model
        self.modelN = load_model('C:/Users/Lewis/Documents/Uni_Work/FifthYear/ImageProcessing/Project/emotionNet_small_model')
        #store input image
        self.img = img

    def accFunc(self):
        #print accuracy of trained model
        #uses test data set of 3589 images stored in X
        #class labels stored in y
        self.X = np.load('C:/Users/Lewis/Documents/Uni_Work/FifthYear/ImageProcessing/Project/X_test_privatetest6_100pct.npy')
        self.y = np.load('C:/Users/Lewis/Documents/Uni_Work/FifthYear/ImageProcessing/Project/y_test_privatetest6_100pct.npy')
        self.score = self.modelN.evaluate(self.X, self.y, verbose=0)
        print("model %s: %.2f%%" % (self.modelN.metrics_names[1], self.score[1]*100))

    def imagePredict(self):
        #predict the class of the input image
        self.categories = ["Happy", "Sad", "Neutral"]
        prediction = self.modelN.predict([self.new_img])
        print(prediction)
        prediction = list(prediction[0])
        print(prediction)
        print(self.categories[prediction.index(max(prediction))])

    def prepare(self):
        #shape any image into one which matches the trained model
        IMG_SIZE = 48
        img_array = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        self.new_img = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        


img = "C:/Users/Lewis/Pictures/Crying-girl.jpg"     #input any image
p1 = EmotionNetClass(img)
p1.accFunc()
p1.prepare()
p1.imagePredict()









    
