import tkinter as tk
from Eye_Detector import GazeDetector
import cv2
import os

HEIGHT = 100
WIDTH = 300

class GUI():
    
    def __init__(self, root):

        self.gazeDetector = GazeDetector()
        
        self.cap=cv2.VideoCapture(0) #Begin webcam capture
        
        self.root = root
        self.root.configure(background='DeepSkyBlue3')
        
        self.canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)


        #Left Frame
        self.leftFrame = tk.Frame(self.root) #, fg="white"
        self.leftFrame.place(relwidth=0.33, relheight=1, relx=0, rely=0)
        self.leftFrame.configure(background='DeepSkyBlue3')

        rocketPath = os.path.join("rocket.png")
        rocket = tk.PhotoImage(file=rocketPath)

        #Left Image
        self.leftImg = tk.Label(self.leftFrame, image=rocket)
        self.leftImg.image = rocket
        self.leftImg.place(relwidth=0.9, relheight=0.5, relx=0.05, rely=0.05)
        self.leftImg.configure(image=rocket)
        
        #Right Emotion Label
        self.leftLabel = tk.Label(self.leftFrame, text="", font=("Helvetica", 20))
        self.leftLabel.place(relwidth=0.9, relheight=0.25, relx=0.05, rely=0.65)
        self.leftLabel.configure(background='DeepSkyBlue3')



        #Middle Frame
        self.middleFrame = tk.Frame(self.root)
        self.middleFrame.place(relwidth=0.34, relheight=1, relx=0.34, rely=0)
        self.middleFrame.configure(background='DeepSkyBlue3')

        burgerPath = os.path.join("burger.png")
        burger = tk.PhotoImage(file=burgerPath)
        
        #Middle Image
        self.middleImg = tk.Label(self.middleFrame)
        self.middleImg.image = burger
        self.middleImg.place(relwidth=0.9, relheight=0.5, relx=0.05, rely=0.05)
        self.middleImg.configure(image=burger)

        #Middle Emotion Label
        self.middleLabel = tk.Label(self.middleFrame, text="", font=("Helvetica", 20))
        self.middleLabel.place(relwidth=0.9, relheight=0.25, relx=0.05, rely=0.65)
        self.middleLabel.configure(background='DeepSkyBlue3')


        
        #Right Frame
        self.rightFrame = tk.Frame(self.root)
        self.rightFrame.place(relwidth=0.33, relheight=1, relx=0.68, rely=0)
        self.rightFrame.configure(background='DeepSkyBlue3')


        puppyPath = os.path.join("puppy.png")
        puppy = tk.PhotoImage(file=puppyPath)
        
        #Right Image
        self.rightImg = tk.Label(self.rightFrame)
        self.rightImg.image = puppy
        self.rightImg.place(relwidth=0.9, relheight=0.5, relx=0.05, rely=0.05)
        self.rightImg.configure(image=puppy)
        
        #Right Emotion Label
        self.rightLabel = tk.Label(self.rightFrame, text="", font=("Helvetica", 20))
        self.rightLabel.place(relwidth=0.9, relheight=0.25, relx=0.05, rely=0.65)
        self.rightLabel.configure(background='DeepSkyBlue3')
                


        self.frameArray = [self.leftFrame, self.middleFrame, self.rightFrame]
        self.emotionArray = [self.leftLabel, self.middleLabel, self.rightLabel]


        
        self.mainLoop()
        
    #Colours section of screen you are looking at
    def colourImage(self, hor, emotion):
        for h in range(3):
            if h == hor: 
                self.frameArray[h].configure(background='Red')
                self.emotionArray[h].configure(background='Red')
                self.emotionArray[h]['text'] = "Your emotional response to this image is: " + emotion
            else:
                self.frameArray[h].configure(background='DeepSkyBlue3')   
                self.emotionArray[h].configure(background='DeepSkyBlue3') 
                self.emotionArray[h]['text'] = ""

    def mainLoop(self):

        ret,frame=self.cap.read()

        hor = self.gazeDetector.getEyeRegion(frame)
        emotion = self.gazeDetector.emotionClassification(frame)

        self.colourImage(hor[0], emotion)

        self.root.after(300, self.mainLoop) #Don't use () after function name!!

root = tk.Tk()

GUI = GUI(root)

root.mainloop()

GUI.cap.release()
cv2.destroyAllWindows()
