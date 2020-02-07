import cv2
import numpy as np

#Sorting function to eliminate false eye reads ------- (Not working ATM)
def sortEyes(eyes):

    #eyes.view('i8,i8,i8,i8')
    eyes.sort(order=['f1'], axis=0)
    eyes = np.flipud(eyes)

    return eyes

Thresh = 20 #Threshold for pupil detection


#Import haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap=cv2.VideoCapture(0) #Begin webcam capture

##loop for the frames of the video and process image by image
while True:
    ret,frame=cap.read()

    if ret is False:
        break
   
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #Convert frame to grey


    faces = face_cascade.detectMultiScale(grey, 1.3, 5) #Detect face(s)
    eyesDetected = 0
    for (x,y,w,h) in faces:

        wMod = int(round(0.2*w))    #width modifier to shrink face frame size
        hMod = int(round(0.2*h))    #heigh " " "
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)        #Draw face frame
        cv2.rectangle(frame,(x+wMod,y+hMod),(x+w-wMod,y+h-hMod),(255,100,0),2)  #Draw shrunk face frame
        roi_colour = frame[y+hMod:y+h-hMod, x+wMod:x+w-wMod]    #Get roi for eye detection from shrunk face region
        roi_grey = cv2.cvtColor(roi_colour, cv2.COLOR_BGR2GRAY)
        
        eyes = eye_cascade.detectMultiScale(roi_grey)   #Detect eyes within faca frame
        for (ex,ey,ew,eh) in eyes:
            eyesDetected += 1
            cv2.rectangle(roi_colour,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)     #Draw detected eye regions

    #Eye Crosshairs
    if eyesDetected > 0:    #If 1 or more eyes detected
        ex = eyes.item(0, 0)
        ey = eyes.item(0, 1)
        ew = eyes.item(0, 2)
        eh = eyes.item(0, 3)
        
        eye_roi_colour = roi_colour[ey:ey+eh, ex:ex+eh]     #Get new roi for specific eye
        eye_roi_grey = cv2.cvtColor(eye_roi_colour, cv2.COLOR_BGR2GRAY)

        rows,cols,_=eye_roi_colour.shape
        _,threshold=cv2.threshold(eye_roi_grey,Thresh,255,cv2.THRESH_BINARY_INV)
        contours,hierarchy=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours=sorted(contours,key=lambda x: cv2.contourArea(x),reverse=True)
        for cnt in contours:
            (x,y,w,h)=cv2.boundingRect(cnt)

            cv2.rectangle(eye_roi_colour,(x,y),(x+w,y+h),(0,0,255),2)   #Colour pupil red
            cv2.line(eye_roi_colour,(x+(w//2),0),(x+(w//2),rows),(0,255,0),2)
            cv2.line(eye_roi_colour, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
            break

        
        if eyesDetected > 1:        #If 2 or more eyes detected
            ex = eyes.item(1, 0)
            ey = eyes.item(1, 1)
            ew = eyes.item(1, 2)
            eh = eyes.item(1, 3)
            
            eye_roi_colour = roi_colour[ey:ey+eh, ex:ex+eh]     #Get new roi for specific eye
            eye_roi_grey = cv2.cvtColor(eye_roi_colour, cv2.COLOR_BGR2GRAY)

            rows,cols,_=eye_roi_colour.shape
            _,threshold=cv2.threshold(eye_roi_grey,Thresh,255,cv2.THRESH_BINARY_INV)
            contours,hierarchy=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours=sorted(contours,key=lambda x: cv2.contourArea(x),reverse=True)
            for cnt in contours:
                (x,y,w,h)=cv2.boundingRect(cnt)

                cv2.rectangle(eye_roi_colour,(x,y),(x+w,y+h),(0,0,255),2)   #Colour pupil red
                cv2.line(eye_roi_colour,(x+(w//2),0),(x+(w//2),rows),(0,255,0),2)
                cv2.line(eye_roi_colour, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
                break
        

    cv2.imshow('eyes',frame)

    #esc to break
    key=cv2.waitKey(30)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()

print(eyes)
eyes = sortEyes(eyes)
print(eyes)
print(type(eyes))




    
