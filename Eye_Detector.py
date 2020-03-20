import cv2
import numpy as np
from matplotlib import pyplot as plt


#Function to sort detected eye values according to their position in the face frame
def sortEyes(eyes):
    xPos = np.array([eyes[0,0], eyes[1,0]])
    xPosSort = np.sort(xPos, axis=0)
    #print("xPos:", xPos)
    #print("sort:", xPosSort)

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

    def setCameraShow(self, show):
        self.show = show

    def setHistShow(self, histShow):
        self.histShow = histShow

##loop for the frames of the video and process image by image
    def getEyeRegion(self, frame):

        #   *! Implement dynamic thresholding (using histogram) for better pupil detection !*
        thresh = 30 #Threshold for pupil detection


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

            wMod = int(round(0.2*w))    #width modifier to shrink face frame size
            tMod = int(round(0.2*h))    #top heigh modifier
            bMod = int(round(0.4*h))    #bottom height modifier (to remove nostrils)
            
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)        #Draw face frame
            cv2.rectangle(frame,(x+wMod,y+tMod),(x+w-wMod,y+h-bMod),(255,100,0),2)  #Draw shrunk face frame
            roi_colour = frame[y+tMod:y+h-bMod, x+wMod:x+w-wMod]    #Get roi for eye detection from shrunk face region
            roi_grey = cv2.cvtColor(roi_colour, cv2.COLOR_BGR2GRAY)
            
            eyes = self.eye_cascade.detectMultiScale(roi_grey)   #Detect eyes within facE frame
            for (ex,ey,ew,eh) in eyes:
                eyesDetected += 1
                cv2.rectangle(roi_colour,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)     #Draw detected eye regions


        hor = -1
        ver = -1

        #Eye Crosshairs
        if eyesDetected > 0 and facesDetected == 1:    #If 1 or more eyes detected

            #   *! Error occurs with eyes.items if more than one face detected since it is tuple/empty array !*
            # facesDetected used to avoid this error

            ex = eyes.item(0, 0)
            ey = eyes.item(0, 1)
            ew = eyes.item(0, 2)
            eh = eyes.item(0, 3)
            
            eye_roi_colour = roi_colour[ey:ey+eh, ex:ex+eh]     #Get new roi for specific eye
            eye_roi_grey = cv2.cvtColor(eye_roi_colour, cv2.COLOR_BGR2GRAY)


            cv2.imshow('eye',eye_roi_grey)

            if self.histShow == True:
                cv2.imshow('eye',eye_roi_grey)
                
                hist = cv2.calcHist([eye_roi_grey], [0], None, [256], [0, 256])
                plt.plot(hist)
                plt.show()
            

            rows,cols,_=eye_roi_colour.shape
            _,threshold=cv2.threshold(eye_roi_grey,thresh,255,cv2.THRESH_BINARY_INV)
            contours,hierarchy=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours=sorted(contours,key=lambda x: cv2.contourArea(x),reverse=True)

            for cnt in contours:
                (x,y,w,h)=cv2.boundingRect(cnt)

                cv2.rectangle(eye_roi_colour,(x,y),(x+w,y+h),(0,0,255),2)   #Colour pupil red
                cv2.line(eye_roi_colour,(x+(w//2),0),(x+(w//2),rows),(0,255,0),2)
                cv2.line(eye_roi_colour, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
                break

            #print("Eye 1")
            #print("Pupil: ", x, " ", y, " ", x+w, " ", y+h)
            #print("Face frame: ", fx, " ", fy, " ", fx+fw, " ", fy+fh)
            #print("Pupil position in reduced face frame: ", (x+w//2 + ex)/(fw - 2*wMod), (y+h//2 + ey)/(fh - tMod - bMod))
            #eyeRelPos[0, 0] = (x+w//2 + ex)/(fw - 2*wMod)    #Get relative x coord of first eye to reduced frame
            #eyeRelPos[0, 1] = (y+h//2 + ey)/(fh - tMod - bMod) #Get relative y coord of first eye to reduced frame
            eyePos[0, 0] = ex
            eyePos[0, 1] = ey
            eyePos[0, 2] = x+w//2
            eyePos[0, 3] = y+h//2

            
            if eyesDetected > 1:        #If 2 or more eyes detected
                ex = eyes.item(1, 0)
                ey = eyes.item(1, 1)
                ew = eyes.item(1, 2)
                eh = eyes.item(1, 3)
                
                eye_roi_colour = roi_colour[ey:ey+eh, ex:ex+eh]     #Get new roi for specific eye
                eye_roi_grey = cv2.cvtColor(eye_roi_colour, cv2.COLOR_BGR2GRAY)

                rows,cols,_=eye_roi_colour.shape
                _,threshold=cv2.threshold(eye_roi_grey,thresh,255,cv2.THRESH_BINARY_INV)
                contours,hierarchy=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                contours=sorted(contours,key=lambda x: cv2.contourArea(x),reverse=True)
                for cnt in contours:
                    (x,y,w,h)=cv2.boundingRect(cnt)

                    cv2.rectangle(eye_roi_colour,(x,y),(x+w,y+h),(0,0,255),2)   #Colour pupil red
                    cv2.line(eye_roi_colour,(x+(w//2),0),(x+(w//2),rows),(0,255,0),2)
                    cv2.line(eye_roi_colour, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
                    break


                #print("Eye 2")
                #print("Pupil: ", x, " ", y, " ", x+w, " ", y+h)
                #print("Face frame: ", fx, " ", fy, " ", fx+fw, " ", fy+fh)
                #print("Pupil position in reduced face frame: ", (x+w//2 + ex)/(fw - 2*wMod), (y+h//2 + ey)/(fh - tMod - bMod))
                #eyeRelPos[1, 0] = (x+w//2 + ex)/(fw - 2*wMod)    #Get relative x coord of first eye to reduced frame
                #eyeRelPos[1, 1] = (y+h//2 + ey)/(fh - tMod - bMod) #Get relative y coord of first eye to reduced frame
                eyePos[1,0] = ex
                eyePos[1,1] = ey
                eyePos[1,2] = x+w//2    #Pupil centre width 
                eyePos[1,3] = y=h//2    #Pupil centre height

            #print(eyePos)

            eyePos = sortEyes(eyePos)       #Sort eyes left, right
            #print(eyePos)
            #print("")

            eyeRelPos = np.array([[0.0, 0.0], [0.0, 0.0]])
            
            eyeRelPos[0, 0] = (eyePos[0,2] + eyePos[0, 0])/((fw - 2*wMod)*0.4)    #Get relative x coord of first eye to reduced frame
            eyeRelPos[0, 1] = (eyePos[0,3] + eyePos[0, 1])/(fh - tMod - bMod) #Get relative y coord of first eye to reduced frame
            eyeRelPos[1, 0] = ((eyePos[1,2] + eyePos[1, 0])-(fw - 2*wMod)*0.6)/((fw - 2*wMod)*0.4)    #Get relative x coord of first eye to reduced frame
            eyeRelPos[1, 1] = (eyePos[1,3] + eyePos[1, 1])/(fh - tMod - bMod) #Get relative y coord of first eye to reduced frame

            
        

            #print(eyeRelPos)
            if (eyeRelPos[0,0] < 0.45): #or (eyeRelPos[1,0] < 0.2)):
                print("Left")
                hor = 0 #0 for left
            elif (eyeRelPos[0,0] > 0.35):
                print("Right")
                hor = 1 #1 for right
            if (eyeRelPos[1,1] < 0.45): # or (eyeRelPos[1,1] < 0.45)):
                print("Top")
                ver = 0 #0 for top
            else:
                print("Bottom")
                ver = 1 #1 for bottom

            print(eyeRelPos)
            
            

        if self.show==True:
            cv2.imshow('eyes',frame)


        return [hor, ver]

def testDetection():   
    gd = GazeDetector()
    gd.setCameraShow(True)
    #gd.setHistShow(True)

    cap=cv2.VideoCapture(0) #Begin webcam capture

    ret,frame=cap.read()
    gd.getEyeRegion(frame)

    while True:
        ret,frame=cap.read()

        if ret is False:
            break

        if cv2.waitKey(33) ==ord('g'):
            gd.getEyeRegion(frame)

        #esc to break
        key=cv2.waitKey(30)
        if key==27:
           break

    cap.release()
    cv2.destroyAllWindows()


#testDetection()



    
