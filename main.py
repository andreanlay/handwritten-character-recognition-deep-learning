import tkinter as tk
import tkinter.messagebox
import win32gui
import numpy as np
import tensorflow as tf
from PIL import ImageGrab

def initialize():
    top = tk.Tk()
    top.geometry("300x350")
    top.title("HCR")
    model = tf.keras.models.load_model("model_v2.h5")
    return top, model

def decode(num):
    return num if num <= 9 else int(num) + 55

def clear():
    canvas.delete("all")

def predict():
    # Step 1 : Getting the canvas ID
    canvas_handle = canvas.winfo_id()
    # Step 2 : Get the canvas from ID
    canvas_rect = win32gui.GetWindowRect(canvas_handle)
    # Step 3 : Get the canvas content
    img = ImageGrab.grab(canvas_rect)
    # Step 4 : Resize the content for CNN input
    img = img.resize((28, 28)).convert("L")
    img = np.array(img)
    img = img.reshape((1, 28, 28, 1))
    img = img / 255.0
    # Step 5 : Predict the image drawn
    Y = model.predict([img])[0]
    tkinter.messagebox.showinfo("Prediction", "it's a " + str(decode(np.argmax(Y))))


def mouse_event(event):
    x, y = event.x, event.y
    canvas.create_oval(x, y, x, y, fill='white', outline='white', width=25)


(root, model) = initialize()
button_frame = tk.Frame(root)

canvas = tk.Canvas(root, bg="black", height=300, width=300)
canvas.bind('<B1-Motion>', mouse_event)
clear_button = tk.Button(button_frame, text="Clear", command=clear)
predict_button = tk.Button(button_frame, text="Predict", command=predict)

canvas.pack()
clear_button.pack(side="left")
predict_button.pack(side="right")

button_frame.pack()
root.mainloop()
