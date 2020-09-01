#!/usr/bin/env python
# coding: utf-8

# In[1]:

print ("hellow world")

from keras.models import load_model


# In[4]:


from tkinter import *
import tkinter as tk


# In[5]:


#import win32gui
from PIL import ImageGrab, Image
import numpy as np


# In[6]:


import os


# In[7]:


os.getcwd()


# In[8]:


os.chdir("models/")


# In[9]:


os.listdir()



# In[10]:


#LOAD MODEL
model = load_model('mnist.h5')


# In[11]:


def predict_digit(img):
    """f(x): Takes drawn image, processes to ML input and runs prediction"""

    #resize image to 28x28 pixels
    img = img.resize((28,28))
    #convert rgb to grayscale
    img = img.convert('L')
    img = np.array(img)
    #reshaping to support our model input and normalizing
    img = img.reshape(1,28,28,1)
    img = img/255.0
    #predicting the class
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


# In[14]:


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Draw..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command = self.classify_handwriting)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        print ("HWND", HWND)
        #coords_var=self.canvas.coords("all")
        print ("HWND coords", self.canvas.coords("all"))
        print ("HWND bbox", self.canvas.bbox("all"))
        bbox_var = self.canvas.bbox("all")
        a,b,c,d = bbox_var


        #CORRECT
        #rect = win32gui.GetWindowRect(HWND)
        #print ("rect", rect)  # get the coordinate of the canvas
        #a,b,c,d = rect
        rect=(a+4,b+4,c-4,d-4)
        im = ImageGrab.grab(rect)

        # Runs Classifier on captured input
        digit, acc = predict_digit(im)
        self.label.configure(text= "Prediction :"+str(digit)+',\n '+"Accuracy :"+ str(int(acc*100))+'%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
mainloop()
