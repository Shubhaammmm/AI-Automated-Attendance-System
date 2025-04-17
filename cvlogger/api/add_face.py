import cv2
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import mysql.connector
import numpy as np
import io

#pretrained model to detect faces.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#create connection to database.
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    database='Face'
)
cursor = conn.cursor()


#function to detect and capture.
def detect_and_capture():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(image=img)

    panel.config(image=img)
    panel.image = img

    panel.after(10, detect_and_capture)

root = tk.Tk()
root.title("Live Camera Feed")

panel = tk.Label(root)
panel.pack()

cap = cv2.VideoCapture(1)

def close_window():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

btn_capture = tk.Button(root, text="Capture Face", command=lambda: capture_face(cap))
btn_capture.pack(pady=10)



#function to capture and ask userinput for name.
def capture_face(cap):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_crop = frame[y:y+h, x:x+w]

        name = simpledialog.askstring("Input", "Enter name for the captured face:")
        if name:
            img_byte_arr = io.BytesIO()
            Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)).save(img_byte_arr, format='JPEG')
            img_blob = img_byte_arr.getvalue()

            insert_image_to_db(name, img_blob)

#insert name and image to database.
def insert_image_to_db(name, img_blob):
    insert_query = "INSERT INTO images (name, image) VALUES (%s, %s)"
    cursor.execute(insert_query, (name, img_blob))
    conn.commit()
    print("Image inserted into database")

btn_exit = tk.Button(root, text="Exit", command=close_window)
btn_exit.pack(pady=10)

detect_and_capture()

root.mainloop()
