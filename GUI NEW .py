from tkinter import *
from tkinter import filedialog
import os
from PIL import Image,ImageTk
import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("/Users/drbk/Downloads/1.yolov4.weights", "/Users/drbk/Downloads/1.yolov4.cfg")
classes = []
with open("/Users/drbk/Downloads/coco.names1", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    #print(classes)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

def upload_file():
    global img
    global filename
    f_types = [('jpg Files', '*.jpg'),('png Files','*.png')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    print(filename)
    img = ImageTk.PhotoImage(file=filename)
    b2 =Button(tk,image=img) # using Button
    b2.pack()



def object_detection():
    img = cv2.imread(filename)
    
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)


    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    img_2=ImageTk.PhotoImage(Image.open(img))
    canvas.create_image(20,20,anchor=NW,image=img_2)
    root.mainloop()

tk = Tk()
tk.title("Object Detection App")
tk.geometry("500x600")
Lbl=Label(text='OBJECT DETECTION',bd = 7,font=("comic sans",30,"bold",),bg='blue',fg='white',relief='raised')
Lbl.pack()
btn=Button(tk,text='Upload Image' , command=upload_file,bd = 7,font=("comic sans",30,"bold",),bg='blue',fg='black',relief='raised')
btn.pack()
btn=Button(tk,text='Press to detect' , command=object_detection, bd = 7,font=("comic sans",30,"bold",),bg='blue',fg='black',relief='raised')
btn.pack()
tk.mainloop()
