import numpy as np
from keras.models import load_model
import cv2
import os

class EmotionNetClass:
    def __init__(self):
        #load the model which is local save of google colab model
        self.modelN = load_model('ExpressionClassification/emotionNet_small_model')
        #store input image
        #self.img = ""

    def accFunc(self):
        #print accuracy of trained model
        #uses test data set of 3589 images stored in X
        #class labels stored in y
        self.X = np.load('X_test_privatetest6_100pct.npy')
        self.y = np.load('y_test_privatetest6_100pct.npy')
        self.score = self.modelN.evaluate(self.X, self.y, verbose=0)
        print("model %s: %.2f%%" % (self.modelN.metrics_names[1], self.score[1]*100))

    def imagePredict(self):
        #predict the class of the input image
        self.categories = ["Happy", "Sad", "Neutral"]
        prediction = self.modelN.predict([self.new_img])
        #print(prediction)
        prediction = list(prediction[0])
        #print(prediction)
        return(self.categories[prediction.index(max(prediction))])

    def prepare(self, img):
        self.img = img
        
        #shape any image into one which matches the trained model
        IMG_SIZE = 48
        #img_array = cv2.imread(self.img, cv2.IMREAD_GRAYSCALE)
        img_array = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) 
        #cv2.imshow("Gray", img_array)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        #cv2.imshow("Rezise", new_array)
        #print(new_array)
        self.new_img = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        


#img = "R5.png"     #input any image
#p1 = EmotionNetClass()
#print("----Init-----")
#p1.accFunc()
#print("----acc-----")
#img = cv2.imread("R5.png", cv2.IMREAD_COLOR)
#p1.prepare(img)
#print("----prep-----")
#print(p1.imagePredict())
#print("----Img1-----")
#path = os.path.join("M8.jpg")
#im = cv2.imread(path)
#cv2.imshow("Test", im)
#p1.prepare(img)
#print("----prep-----")
#print(p1.imagePredict())
#print("----Img2-----")
 
def test():  
    gd = EmotionNetClass()        
    cap = cv2.VideoCapture(0)
    
    ret,frame=cap.read()
    #print("Type", frame.type())
    cv2.imshow('Cam Feed', frame)
    
    while True:
        ret,frame=cap.read()
    
        if ret is False:
            break
        
    
        if cv2.waitKey(33) ==ord('g'):
            cv2.imshow('Cam Feed', frame)
            gd.prepare(frame)
            print(gd.imagePredict())
    
        #esc to break
        key=cv2.waitKey(30)
        if key==27:
           break
    
    cap.release()
    cv2.destroyAllWindows()