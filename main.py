# Import necessary Libraries 
import numpy as np
import cv2
import os
import PIL
from PIL import ImageTk
import PIL.Image
import speech_recognition as sr
import pyttsx3
from itertools import count
import string
from tkinter import *
import time
import pyttsx3
try:
       import tkinter as tk
except:
       import tkinter as tk
import tkinter as tk
import numpy as np

#Initialize the text-to-speech engine

text_speech= pyttsx3.init()

#Define image dimensions

image_x, image_y = 64,64

#load the pre-trained Keras model

from keras.models import load_model

classifier = load_model('model.h5')

#Function to determine character similarity

def give_char():
    
    #check similarity of input with items in the file map 

    import numpy as np
    from keras.preprocessing import image
    from keras.preprocessing.image import load_img
    
    test_image = load_img('tmp1.png', target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    print(result)
    
    char_map ={i: chr(ord('A')+i) for i in range(26)}
    indx=np.argmax(result[0])

    return char_map.get(indx,' ')


def check_sim(i,file_map):
       for item in file_map:
              for word in file_map[item]:
                     if(i==word):
                            return 1,item
       return -1,""

#Define file paths for data and alphabet images 

op_dest="D:\\PANDA\\filtered_data\\"
alpha_dest="D:\\PANDA\\alphabet\\"

#Retrieve file listings and create a mapping of file name to words 

dirListing = os.listdir(op_dest)
editFiles = []
for item in dirListing:
       if ".webp" in item:
              editFiles.append(item)

file_map={}
for i in editFiles:
       tmp=i.replace(".webp","")
       # print(tmp)
       tmp=tmp.split()
       file_map[i]=tmp


#Function to convert input text to corresponding sing language images

def func(a):

       #Initialize variables 

       all_frames=[]
       final= PIL.Image.new('RGB', (380, 260))
       words=a.split()
       #Process each word in the input text 

       for i in words:
              flag,sim=check_sim(i,file_map)
              # If word not found, process individual characters

              if(flag==-1):
                     for j in i:
                            print(j)
                            im = PIL.Image.open(alpha_dest+str(j).lower()+"_small.gif")
                            frameCnt = im.n_frames
                            for frame_cnt in range(frameCnt):
                                   im.seek(frame_cnt)
                                   im.save("tmp.png")
                                   img = cv2.imread("tmp.png")
                                   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                   img = cv2.resize(img, (380,260))
                                   im_arr = PIL.Image.fromarray(img)
                                   for itr in range(15):
                                          all_frames.append(im_arr)
              else:
                     print(sim)
                     im = PIL.Image.open(op_dest+sim)
                     im.info.pop('background', None)
                     im.save('tmp.gif', 'gif', save_all=True)
                     im = PIL.Image.open("tmp.gif")
                     frameCnt = im.n_frames
                     for frame_cnt in range(frameCnt):
                            im.seek(frame_cnt)
                            im.save("tmp.png")
                            img = cv2.imread("tmp.png")
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, (380,260))
                            im_arr = PIL.Image.fromarray(img)
                            all_frames.append(im_arr)

        # Save the generated GIF image

       final.save("out.gif", save_all=True, append_images=all_frames, duration=100, loop=0)
       return all_frames     
 
# Initialize variables for image counter and text storage

img_counter = 0
img_text=''

# Class to manage the Tkinter application

class Tk_Manage(tk.Tk):
       def __init__(self, *args, **kwargs):     
              tk.Tk.__init__(self, *args, **kwargs)
              container = tk.Frame(self,bg="lightblue")
              container.pack(side="top", fill="both", expand = True)
              container.grid_rowconfigure(0, weight=1)
              container.grid_columnconfigure(0, weight=1)
              self.frames = {}
              for F in (StartPage, VtoS, StoV):
                     frame = F(container, self)
                     self.frames[F] = frame
                     frame.grid(row=0, column=0, sticky="nsew")
                     frame.configure(bg="lightblue")
              self.show_frame(StartPage)

       def show_frame(self, cont):
              frame = self.frames[cont]
              frame.tkraise()

# Class for the Start Page of the application
        
class StartPage(tk.Frame):

       def __init__(self, parent, controller):
              tk.Frame.__init__(self,parent,bg="lightblue")
              label = tk.Label(self, text="Project of Artifical Intelligence For Deaf and Aphonic",bg="lightblue" ,font=("Times New Roman", 32))
              label.pack(pady=10,padx=10)
              button = tk.Button(self, text="Deaf ",font=("Helvetica", 16),bg="lightyellow", command=lambda: controller.show_frame(VtoS))
              button.pack(pady=10)
              button2 = tk.Button(self, text="Apohnic ",font=("Helvetica", 16),bg="lightyellow",command=lambda: controller.show_frame(StoV))
              button2.pack(pady=10)
              # parent('-fullscreen',True)
              button.config(width=15, height=1)
              button2.config(width=15, height=1)

              load = PIL.Image.open("bg1.png")
              load = load.resize((800,500))
              render = ImageTk.PhotoImage(load)
              img = Label(self, image=render,bg="lightblue")
              img.image = render
              img.place(x=400, y=270) 
              
# Class for Voice to Sign functionality

class VtoS(tk.Frame):
       def __init__(self, parent, controller):
              cnt=0
              gif_frames=[]
              global inputtxt
              tk.Frame.__init__(self, parent)
              label = tk.Label(self, text="Voice to Sign",bg="lightblue" ,font=("Times New Roman", 32))
              label.pack(pady=10,padx=10)
              gif_box = tk.Label(self)
              
              button1 = tk.Button(self, text="Back to Home",font=("Helvetica", 20),bg="lightyellow",command=lambda: controller.show_frame(StartPage))
              button1.pack(pady=10)
              button2 = tk.Button(self, text="Sign to Voice",font=("Helvetica",22),bg="lightyellow",command=lambda: controller.show_frame(StoV))
              button2.pack(pady=10)
              def gif_stream():
                     global cnt
                     global gif_frames
                     if(cnt==len(gif_frames)):
                            return
                     img = gif_frames[cnt]
                     cnt+=1
                     imgtk = ImageTk.PhotoImage(image=img)
                     gif_box.imgtk = imgtk
                     gif_box.configure(image=imgtk)
                     gif_box.after(50, gif_stream)
              def hear_voice():
                     store = sr.Recognizer()
                     with sr.Microphone() as s:
                            try:
                                   print("listening ....")
                                   audio_input = store.listen(s,timeout=5)
                                   print("recording complete....")
                                   text_output = store.recognize_google(audio_input, language='en-US')
                                   inputtxt.insert(END,text_output)
                            except sr.WaitTimeoutError:
                                   print("Timeout error. No speech detected")
                            except sr.RequestError as e : 
                                   print(f"Could not request results from Google Web Speech API;{e}")
                            except sr.UnknownValueError:
                                   print("Unable to recoginze speech")
                            except Exception as e :
                                   print(f"An error occured:{e}")
                                   inputtxt.insert(END, '')
              def Take_input():
                     input = inputtxt.get("1.0", "end-1c")
                     INPUT= input.lower()
                     print(INPUT)
                     text_speech.say(INPUT)
                     text_speech.runAndWait()
                     global gif_frames
                     gif_frames=func(INPUT)
                     global cnt
                     cnt=0
                     gif_stream()
                     gif_box.place(x=520,y=350)
                     inputtxt.delete("1.0", tk.END)
              
              l = tk.Label(self,text = "Enter your Text :",bg="lightblue",font=("Helvetica", 16))
              inputtxt = tk.Text(self, height = 6,width = 36)
              l1 = tk.Label(self,text = "OR",bg="lightblue",font=("Helvetica", 18))
              voice_button= tk.Button(self,height = 3,bg="lightpink",width = 30,text="Record ",command=lambda: hear_voice(),relief=tk.GROOVE, borderwidth=2)
              voice_button.place(x=180,y=420)
              Display = tk.Button(self,height = 2,width = 20,text ="Convert",bg="lightgreen",command = lambda:Take_input())
              l.place(x=50, y=220)
              l1.place(x=255, y=370)
              inputtxt.place(x=150, y=250)
              Display.pack()

# Class for Sign to Voice functionality

class StoV(tk.Frame):

       def __init__(self, parent, controller):
              tk.Frame.__init__(self, parent)
              label = tk.Label(self, text="Sign to Voice", bg="lightblue" ,font=("Times New Roman", 32))
              label.pack(pady=10,padx=10)
              button1 = tk.Button(self, text="Back to Home",font=("Helvetica", 20),bg="lightyellow",command=lambda: controller.show_frame(StartPage))
              button1.pack(padx=10,pady=10)
              button2 = tk.Button(self, text="Voice to Sign",font=("Helvetica", 22),bg="lightyellow",command=lambda: controller.show_frame(VtoS))
              button2.pack(pady=10,padx=10)
              disp_txt = tk.Text(self, height = 4,width = 25)
              self.cam=False
              # start_vid = tk.Button(self, height=2, width=20, text="Start Video",disp_txt=None, command=lambda: start_video(self))                      
              # start_vid.pack()

              def start_video():
                     video_frame = tk.Label(self)
                     cam = cv2.VideoCapture(0)
                     global img_counter
                     img_counter = 0
                     global img_text
                     img_text = ''
                     def video_stream():
                            global img_text
                            global img_counter
                            if(img_counter>50):
                                   cam.release()
                                   return None
                            img_counter+=1
                            ret, frame = cam.read()
                            frame = cv2.flip(frame,1)
                            img=cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8, shift=0)
                            lower_blue = np.array([35,10,0])
                            upper_blue = np.array([160,230,255])
                            imcrop = img[102:298, 427:623]
                            hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
                            mask = cv2.inRange(hsv, lower_blue, upper_blue)
                            cv2.putText(frame, img_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
                            img_name = "tmp1.png"
                            save_img = cv2.resize(mask, (image_x, image_y))
                            cv2.imwrite(img_name, save_img)
                            time.sleep(0.1)
                            tmp_text=img_text[0:]
                            img_text = give_char()
                            if(tmp_text!=img_text):
                                   print(tmp_text)
                                   disp_txt.insert(tk.END, tmp_text)
                            img = PIL.Image.fromarray(frame)
                            imgtk = ImageTk.PhotoImage(image=img)
                            video_frame.imgtk = imgtk
                            video_frame.configure(image=imgtk)
                            video_frame.after(1, video_stream)
                     video_stream()
                     disp_txt.pack()
                     video_frame.pack()
              start_vid = tk.Button(self,height = 2,width = 20, text="Start Video",command=lambda: start_video())
              start_vid.pack()


# Start the Tkinter application

app = Tk_Manage()
app.geometry("800x750")
app.mainloop()