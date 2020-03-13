import tkinter as tk
from Eye_Detector import GazeDetector
import cv2
#import os

HEIGHT = 100
WIDTH = 300

class GUI():
    
    def __init__(self, root):

        self.gazeDetector = GazeDetector()

        self.gazeDetector.setCameraShow(True) #Turn camera display on
        
        self.cap=cv2.VideoCapture(0) #Begin webcam capture
        
        self.root = root
        self.root.configure(background='DeepSkyBlue3')
        
        self.canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
        
        #Top Left Image
        self.topLeftImg = tk.Label(self.root, text="Top Left", font=("Helvetica", 20)) #, fg="white"
        self.topLeftImg.place(relwidth=0.5, relheight=0.5, relx=0, rely=0)
        self.topLeftImg.configure(background='DeepSkyBlue3')
        
        #How to add image
            #filePath = os.path.join("TwitterIcon.png")
            #print(filePath)
            #logo = tk.PhotoImage(file=filePath)
            #self.logoLabel = tk.Label(self.root, image=logo)  
            #self.logoLabel.image = logo
            #self.logoLabel.place(relwidth=0.2, relheight=0.2, relx=0.8, rely=0)
            #self.logoLabel.configure(image=logo)
        
        #Top Right Image
        self.topRightImg = tk.Label(self.root, text="Top Right", font=("Helvetica", 20))
        self.topRightImg.place(relwidth=0.5, relheight=0.5, relx=0.5, rely=0)
        self.topRightImg.configure(background='DeepSkyBlue3')
        
        #Bottom Left Image
        self.bottomLeftImg = tk.Label(self.root, text="Bottom Left", font=("Helvetica", 20))
        self.bottomLeftImg.place(relwidth=0.5, relheight=0.5, relx=0, rely=0.5)
        self.bottomLeftImg.configure(background='DeepSkyBlue3')

        #Bottom Right Image
        self.bottomRightImg = tk.Label(self.root, text="Bottom Right", font=("Helvetica", 20))
        self.bottomRightImg.place(relwidth=0.5, relheight=0.5, relx=0.5, rely=0.5)
        self.bottomRightImg.configure(background='DeepSkyBlue3')

        self.imageArray = [[self.topLeftImg, self.topRightImg],
                            [self.bottomLeftImg, self.bottomRightImg]]


        self.button = tk.Button(self.root, command=lambda:print("-------------Change---------------"))
        self.button.place(relwidth=0.2, relheigh=0.2, relx=0.4, rely=0.4)
        
        self.mainLoop()
        
    #Colours section of screen you are looking at
    def colourImage(self, hor,  ver):
        for h in range(2):
            for v in range(2):
                if h == hor and v == ver:
                    self.imageArray[h][v].configure(background='Red')
                else:
                    self.imageArray[h][v].configure(background='DeepSkyBlue3')        

    def mainLoop(self):

        #cap=cv2.VideoCapture(0) #Begin webcam capture
    
        ret,frame=self.cap.read()

        #if ret is False:
            #break

        hor, ver = self.gazeDetector.getEyeRegion(frame)

        self.colourImage(hor, ver)

        #cap.release()
        #cv2.destroyAllWindows()
        
        #esc to break
        #key=cv2.waitKey(30)
        #if key==27:
            #break

        self.root.after(300, self.mainLoop) #Don't use () after function name!!

root = tk.Tk()

GUI = GUI(root)

#root.after(10000, GUI.mainLoop())
root.mainloop()

GUI.cap.release()
cv2.destroyAllWindows()
