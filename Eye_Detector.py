import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from ExpressionClassification.emotionNet import EmotionNetClass


#Function to sort detected eye values according to their position in the face frame
def sortEyes(eyes):
    xPos = np.array([eyes[0,0], eyes[1,0]])
    xPosSort = np.sort(xPos, axis=0)

    eyesOut = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    eyesOut[0,:] = eyes[0,:]
    eyesOut[1,:] = eyes[1,:]
        
    if (xPos[0] != xPosSort[0]):
        eyesOut[0,:] = eyes[1,:]
        eyesOut[1,:] = eyes[0,:]

    return eyesOut
                    

class GazeDetector():

    def __init__(self):
        
        #Import haar cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.show = False
        self.histShow = False
        self.flip = True
        self.emotionNet = EmotionNetClass()

    def setCameraShow(self, show):
        self.show = show

    def setHistShow(self, histShow):
        self.histShow = histShow

    def setFlip(self, flip):
        self.flip = flip

##loop for the frames of the video and process image by image
    def getEyeRegion(self, frame):

        #   *! Implement dynamic thresholding (using histogram) for better pupil detection !*
        pixelThresh = 60 #Threshold for pupil detection

        if self.flip == True:
            frame = cv2.flip(frame, 1) #Horizontally filp frame
   
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #Convert frame to grey


        faces = self.face_cascade.detectMultiScale(grey, 1.3, 5) #Detect face(s)
        eyesDetected = 0
        facesDetected = 0
        eyePos = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        eyeRelPos = np.array([[0.0, 0.0], [0.0, 0.0]])
    
        for (x,y,w,h) in faces:

            if (facesDetected < 1):
                fx = x
                fy = y
                fw= w
                fh = h

            facesDetected += 1

            wMod = int(round(0.15*w))    #width modifier to shrink face frame size
            tMod = int(round(0.2*h))    #top heigh modifier
            bMod = int(round(0.4*h))    #bottom height modifier (to remove nostrils)
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)        #Draw face frame
            cv2.rectangle(frame,(x+wMod,y+tMod),(x+w-wMod,y+h-bMod),(255,100,0),2)  #Draw shrunk face frame

            cv2.line(frame,(x+(w//2),y+tMod),(x+(w//2),y+h-bMod),(0,255,0),2)   #Draw line down centre of reduced face frame
            
            roi_colour = frame[y+tMod:y+h-bMod, x+wMod:x+w-wMod]    #Get roi for eye detection from shrunk face region
            roi_grey = cv2.cvtColor(roi_colour, cv2.COLOR_BGR2GRAY)
            
            eyes = self.eye_cascade.detectMultiScale(roi_grey)   #Detect eyes within facE frame
            for (ex,ey,ew,eh) in eyes:
                eyesDetected += 1
          
        hor = -1
        ver = -1

        #Eye Crosshairs
        if eyesDetected > 1 and facesDetected == 1:    #If 2 or more eyes detected

            #   *! Error occurs with eyes.items if more than one face detected since it is tuple/empty array !*
            # facesDetected used to avoid this error

            ex = eyes.item(0, 0)
            ey = eyes.item(0, 1)
            ew = eyes.item(0, 2)
            eh = eyes.item(0, 3)

            eyePos[0, 0] = ex
            eyePos[0, 1] = ey
            
            eye_roi_colour = roi_colour[ey:ey+eh, ex:ex+eh]     #Get new roi for specific eye
            eye_roi_grey = cv2.cvtColor(eye_roi_colour, cv2.COLOR_BGR2GRAY)


            #Create red eye image for thresholding
            red = eye_roi_colour.copy()
       
            red[:,:,0] = 0
            red[:,:,1] = 0
            
    
            testRedGrey = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY) * 3
            #cv2.imshow('Test Red Grey', testRedGrey)

            nonZero = 0
            x = 10
            
            while nonZero < pixelThresh:
                x = x + 1    
                #Red threshold for pupil extraction
                _,redThresh=cv2.threshold(red,x,255,cv2.THRESH_BINARY_INV)


                redGrey = cv2.cvtColor(redThresh, cv2.COLOR_BGR2GRAY)
                _,redGreyThresh=cv2.threshold(redGrey,200,255,cv2.THRESH_BINARY)
                
                nonZero = cv2.countNonZero(redGreyThresh)

            kernel = np.array([[0,0,1,0,0],
                               [0,1,1,1,0],
                               [1,1,1,1,1],
                               [0,1,1,1,0],
                               [0,0,1,0,0]], np.uint8)

            redClose = cv2.morphologyEx(redGreyThresh, cv2.MORPH_CLOSE, kernel)
            redCloseOpen = cv2.morphologyEx(redClose, cv2.MORPH_OPEN, kernel)

            if self.histShow == True:
                cv2.imshow('eye',testRedGrey)
                
                hist = cv2.calcHist([testRedGrey], [0], None, [256], [0, 256])
                plt.plot(hist)
                plt.show()
            

            eyePos[0, 2], eyePos[0, 3] = self.detectPupils(eye_roi_colour, redCloseOpen, "Eye 1")                  
            
            ex = eyes.item(1, 0)
            ey = eyes.item(1, 1)
            ew = eyes.item(1, 2)
            eh = eyes.item(1, 3)

            eyePos[1, 0] = ex
            eyePos[1, 1] = ey
                
            eye_roi_colour = roi_colour[ey:ey+eh, ex:ex+eh]     #Get new roi for specific eye
            eye_roi_grey = cv2.cvtColor(eye_roi_colour, cv2.COLOR_BGR2GRAY)


            red = eye_roi_colour.copy()
            red[:,:,0] = 0
            red[:,:,1] = 0


            nonZero = 0
            x = 0
            
            while nonZero < pixelThresh:
                x = x + 1

                #Red threshold for pupil extraction
                _,redThresh=cv2.threshold(red,x,255,cv2.THRESH_BINARY_INV)

                redGrey = cv2.cvtColor(redThresh, cv2.COLOR_BGR2GRAY)
                _,redGreyThresh=cv2.threshold(redGrey,200,255,cv2.THRESH_BINARY)
                
                nonZero = cv2.countNonZero(redGreyThresh)
            
            redGrey = cv2.cvtColor(redThresh, cv2.COLOR_BGR2GRAY)
            _,redGreyThresh=cv2.threshold(redGrey,200,255,cv2.THRESH_BINARY)

            redClose = cv2.morphologyEx(redGreyThresh, cv2.MORPH_CLOSE, kernel)
            redCloseOpen = cv2.morphologyEx(redClose, cv2.MORPH_OPEN, kernel)


            eyePos[1, 2], eyePos[1, 3] =  self.detectPupils(eye_roi_colour, redCloseOpen, "Eye 2")

            eyePos = sortEyes(eyePos)       #Sort eyes left, right


            eyeRelPos = np.array([[0.0, 0.0], [0.0, 0.0]])

            fwr = fw - 2*wMod #reduced frame width

            #If first pupil detected
            if eyePos[0, 2] > 0:
                eyeRelPos[0, 0] = (fwr*0.5 - (eyePos[0,2] + eyePos[0, 0]))/(fwr*0.5)    #Get relative x coord of first eye to reduced frame

            #If second pupil detected
            if eyePos[1, 2] > 0:
                eyeRelPos[1, 0] = (fwr*0.5 - (fwr - (eyePos[1,0] + eyePos[1, 2])))/(fwr*0.5)    #Get relative x coord of second eye to reduced frame

            offset = 0.05
                
            #If both pupils detected
            if (eyePos[0,2] > 0 and eyePos[1,2] >0):
                if (eyeRelPos[1,0] + offset < eyeRelPos[0,0]): 
                    #print("Left")
                    hor = 0 #0 for left
                elif (eyeRelPos[0,0] + offset < eyeRelPos[1,0]):
                    #print("Right")
                    hor = 2 #2 for right
                else:
                    #print("Middle")
                    hor = 1 # 1 for middle

        if self.show==True:
            cv2.imshow('Processed Camera Feed',frame)

        return [hor]
    

    def detectPupils(self, roi, roi_grey, name):

        eye_roi_grey = roi_grey
        eye_roi_colour = roi
        
        dst = cv2.Canny(eye_roi_grey, 50, 200, None, 3)
            
        for p2 in range(30, 4, -1):

            #param2 = accumulator threshold - higher val, less circles detected but more accurate
            circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, 5, param1=50, param2=p2, minRadius=0, maxRadius=0)

            if circles is not None:
                cimg = eye_roi_colour.copy()
                    
                circles = np.uint16(np.around(circles))
                for i in circles[0,:]:
                    # draw the outer circle
                    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
                    # draw the center of the circle
                    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

                cv2.imshow(name, cimg)

                # draw the outer circle
                cv2.circle(eye_roi_colour,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(eye_roi_colour,(i[0],i[1]),2,(0,0,255),3)

                return i[0], i[1]

        return 0, 0

    def emotionClassification(self, img):
        self.emotionNet.prepare(img)
        return(self.emotionNet.imagePredict())


#Function to test Gaze tracking on webcam without the GUI
def testDetectionFeed():   
    gd = GazeDetector()
    gd.setCameraShow(True)
    #gd.setHistShow(True)

    cap=cv2.VideoCapture(0) #Begin webcam capture

    ret,frame=cap.read()
    
    gd.getEyeRegion(frame)

    ret = True


    while True:
        ret,frame=cap.read()

        if ret is False:
            break

        if cv2.waitKey(33) ==ord('g'):
            print(gd.getEyeRegion(frame))
            print("Emotion:", gd.emotionClassification(frame))

        #esc to break
        key=cv2.waitKey(30)
        if key==27:
           break

    cap.release()


#Function to test Gaze tracking on input images
def testDetectionImage(imagePath):
    gd = GazeDetector()
    gd.setCameraShow(True)
    gd.setFlip(False)
    
    path = os.path.join(imagePath)
    frame = cv2.imread(path)

    dimensions = frame.shape

    resize = dimensions[0]/720

    height = round(dimensions[0]/resize)
    width = round(dimensions[1]/resize)

    #Resize image to size of web camera frame
    resizedFrame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)


    classification = ["Left", "Middle", "Right", "No Classification"]
    print(imagePath, classification[gd.getEyeRegion(resizedFrame)[0]])


