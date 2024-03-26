# import tkinter as tk
# from tkinter import filedialog
# from tkinter import ttk
# from subprocess import call  # To call your Jupyter Notebook
# from tkinter import LEFT, END
# from PIL import Image , ImageTk 
# from tkinter.filedialog import askopenfilename
# import cv2
# import numpy as np
# import time
# import h5py
# import pickle
# from skimage import feature

# root = tk.Tk()
# root.configure(background="seashell2")
# #root.geometry("1300x700")
# root.title("Melanoma Cancer Detection using Deep Learning")

# # img=ImageTk.PhotoImage(Image.open("img1.jpg"))

# logo_label=tk.Label()
# logo_label.place(x=0,y=0)

# x = 1

# def select_image():
#     global filename
#     filename = filedialog.askopenfilename(initialdir="C:/Users/AJAY/Desktop/project/melanoma-detection-main/final", title="Select Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
#     if filename:
#         image_label.config(text=f"Selected Image: {filename}")

# # Function to choose model (replace with your actual model selection logic)
# def choose_model(model_num):
#     global selected_model
#     selected_model = model_num  # Update based on your model selection logic (e.g., dictionary)
#     model_label.config(text=f"Selected Model: {selected_model}")  # Display selected model

# def detect_cancer():
#     if not filename:
#         return ms.showerror("Error", "Please select an image file!")
#     if not selected_model:
#         return ms.showerror("Error", "Please choose a model!")

# # Call your Jupyter Notebook with arguments (filename, model)
# # call(["jupyter", "nbconvert", "--execute", "--to", "python", "your_notebook.ipynb", f"--args {filename} {selected_model}"])  # Replace with your notebook name

# # Optionally, display results after notebook execution (read from a file/variable)

# # Initialize variables (optional)
# filename = ""
# selected_model = None


# # --- Enhanced UI elements ---
# root = tk.Tk()
# root.title("Melanoma Cancer Detection")
# root.configure(background="#F5F5F5")  # Lighter background color

# style = ttk.Style()
# style.configure("TButton", font=("Arial", 14), foreground="#333")  # Button styling
# style.configure("TLabel", font=("Arial", 12), foreground="#333")  # Label styling

# # Get screen dimensions
# w, h = root.winfo_screenwidth()/2, root.winfo_screenheight()/2

# # Set the window to full screen
# root.geometry("%dx%d+0+0" % (w, h))
# # --- Main content ---

# # Select Image section
# image_frame = ttk.LabelFrame(root, text="Image Selection")
# image_frame.pack(padx=15, pady=15)

# image_label = ttk.Label(image_frame, text="No Image Selected")
# image_label.pack()

# select_button = ttk.Button(image_frame, text="Select Image")
# select_button.pack(pady=5)

# # Model Selection section
# model_frame = ttk.LabelFrame(root, text="Model Selection")
# model_frame.pack(padx=15, pady=15)

# model_label = ttk.Label(model_frame, text="No Model Selected")
# model_label.pack()

# model_buttons = []
# for i in range(6):
#     button = ttk.Button(model_frame, text=f"Model {i+1}")
#     button.pack(pady=3)
#     model_buttons.append(button)

# # Detect Cancer button
# detect_button = ttk.Button(root, text="Detect Cancer", style="Accent.TButton")  # Accentuated button
# detect_button.pack(pady=20)

# # --- Functionality (unchanged) ---
# # ... (rest of your code functions remain the same)

# # --- Run the main loop ---
# root.mainloop()

import tkinter as tk
from tkinter import ttk, filedialog
from subprocess import call
from PIL import Image, ImageTk
from tkinter import messagebox as ms
import imageio
import cv2
import numpy as np
from scipy import ndimage  # To call your Jupyter Notebook

# Function to select image file
image_labels = []

# Function to choose model (replace with your actual model selection logic)
def choose_model(model_num):
    global selected_model
    selected_model = model_num  # Update based on your model selection logic (e.g., dictionary)
    model_label.config(text=f"Selected Model: {selected_model}")  # Display selected model

# Function to launch detection (call your Jupyter Notebook)
def detect_cancer():
    if not filename:
        return ms.showerror("Error", "Please select an image file!")
    if not selected_model:
        return ms.showerror("Error", "Please choose a model!")

    # Call your Jupyter Notebook with arguments (filename, model)
    call(["jupyter", "nbconvert", "--execute", "--to", "python", "your_notebook.ipynb", f"--args {filename} {selected_model}"])  # Replace with your notebook name

    # Optionally, display results after notebook execution (read from a file/variable)

# Initialize variables (optional)
    
# Model Selection (replace with your model options)
# model_label = tk.Label(root, text="No Model Selected")
# model_label.pack()
# model_buttons = [
#     tk.Button(root, text=f"Model {i+1}", command=lambda m=i+1: choose_model(m)) for i in range(6)  # Create buttons for 6 models
# ]
# for button in model_buttons:
#     button.pack()    
filename = ""
selected_model = None

# Create the main window
root = tk.Tk()
root.configure(background="seashell2")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Melanoma Cancer Detection using Deep Learning")

logo_label=tk.Label()
logo_label.place(x=0,y=0)

lbl = tk.Label(root, text="Melanoma Cancer Detection using Deep Learning", font=('times', 35,' bold '), height=1, width=65,bg="violet Red",fg="Black")
lbl.place(x=0, y=0)

threshold = 149.0
# Detect Cancer Button
def remove_hairs_im(image):
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    _, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    dst = cv2.inpaint(image, thresh2, 1, cv2.INPAINT_TELEA)
    return dst

def remove_hairs_img():
    global fn
    if fn:
        img = cv2.imread(fn)
        processed_img = remove_hairs_im(img)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for PIL
        processed_img = Image.fromarray(processed_img)
        processed_img = ImageTk.PhotoImage(processed_img)

        img3 = tk.Label(root, image=processed_img, height=200, width=200)
        img3.image = processed_img
        img3.place(x=550, y=100)
        image_labels.append(img3)
        
    else:
        print("Please select an image first.")        

def convertToGreyScale_im(image):
    image = remove_hairs_im(image)
    def getWeightedAvg(pixel):
        return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]        
    grey = np.zeros(image.shape[0:-1])
    for rownum in range(len(image)):
        for colnum in range(len(image[rownum])):
            grey[rownum][colnum] = getWeightedAvg(image[rownum][colnum])        
    return grey

def convertToGreyScale_img():
    global fn
    if fn:
        img = cv2.imread(fn)
        processed_img = convertToGreyScale_im(img)
        # processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for PIL
        processed_img = Image.fromarray(processed_img)
        processed_img = ImageTk.PhotoImage(processed_img)

        img3 = tk.Label(root, image=processed_img, height=200, width=200)
        img3.image = processed_img
        img3.place(x=800, y=100)
        image_labels.append(img3)
        
    else:
        print("Please select an image first.")

def threshold_im(image, threshold):
    assert len(image.shape) == 2, "Must be grayscale image"
    thresh = np.zeros(image.shape)
    for rownum in range(len(image)):
        for colnum in range(len(image[rownum])):    
            if image[rownum][colnum] > threshold:
                thresh[rownum][colnum] = 0 
            else:
                thresh[rownum][colnum] = 255
    return thresh

def threshold_img():
    global fn
    if fn:
        img = cv2.imread(fn)
        processed_img = threshold_im(convertToGreyScale_im(img),threshold)
        # processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for PIL
        processed_img = Image.fromarray(processed_img)
        processed_img = ImageTk.PhotoImage(processed_img)

        img4 = tk.Label(root, image=processed_img, height=200, width=200)
        img4.image = processed_img
        img4.place(x=1200, y=100)
        image_labels.append(img4)
        
    else:
        print("Please select an image first.") 

def getHistGray_im(image):
    assert len(image.shape) == 2, "Must be grayscale image"
    hist = np.zeros(255)
    for row in image:
        for col in row:
            hist[int(col)] += 1
    return hist

def getHistGray_img():
    global fn
    if fn:
        img = cv2.imread(fn)
        processed_img = getHistGray_im(convertToGreyScale_im(img))
        # processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for PIL
        processed_img = Image.fromarray(processed_img)
        processed_img = ImageTk.PhotoImage(processed_img)

        img4 = tk.Label(root, image=processed_img, height=200, width=200)
        img4.image = processed_img
        img4.place(x=600, y=400)
        image_labels.append(img4)
        
    else:
        print("Please select an image first.") 

def otsu_im(image):
    assert len(image.shape) == 2, "Must be grayscale image"
    th = _getOtsuThreshold_im(image)
    return threshold_im(image, th)

def otsu_img():
    global fn
    if fn:
        img = cv2.imread(fn)
        processed_img = otsu_im(convertToGreyScale_im(img))
        # processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for PIL
        processed_img = Image.fromarray(processed_img)
        processed_img = ImageTk.PhotoImage(processed_img)

        img4 = tk.Label(root, image=processed_img, height=200, width=200)
        img4.image = processed_img
        img4.place(x=1050, y=100)
        image_labels.append(img4)
        
    else:
        print("Please select an image first.")     
    
def _getOtsuThreshold_im(image):
    s = 0
    histogram = getHistGray_im(image)
    for i in range(len(histogram)):
        s += i * histogram[i]
    sumB = 0
    wB = 0
    wF = 0
    mB = None
    mF = None
    m = 0.0
    between = 0.0
    threshold1 = 0.0
    threshold2 = 0.0
    total = len(image.ravel())
    for i in range(len(histogram)):
        wB += histogram[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * histogram[i]
        mB = sumB / wB
        mF = (s - sumB) / wF
        between = wB * wF * ((mB - mF) ** 2)
        if between >= m:
            threshold1 = i
            if between > m:
                threshold2 = i
            m = between
    return (threshold1 + threshold2) / 2.0

def masking_im(image, mask):
    new_img = np.array(image)
    new_img = remove_hairs_im(new_img)
    for row in range(len(image)):
        for col in range(len(image[row])):
            if mask[row,col]==0:
                new_img[row,col] = 0
            if mask[row,col]==1:
                new_img[row,col] = image[row,col]
    return new_img

def masking_img():
    global fn
    if fn:
        img = cv2.imread(fn)
        processed_img = remove_hairs_im(masking_im(img,opened_im(img)))
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for PIL
        processed_img = Image.fromarray(processed_img)
        processed_img = ImageTk.PhotoImage(processed_img)

        img4 = tk.Label(root, image=processed_img, height=200, width=200)
        img4.image = processed_img
        img4.place(x=1300, y=100)
        image_labels.append(img4)
        
    else:
        print("Please select an image first.") 

def opened_im(image):
    grey = convertToGreyScale_im(image)
    thresh = otsu_im(grey)
    dilimg = ndimage.binary_dilation(thresh)
    for x in range(25):
        dilimg = ndimage.binary_dilation(dilimg)
    erimg = ndimage.binary_erosion(dilimg)
    for x in range(25):
        erimg = ndimage.binary_erosion(erimg)
    return erimg

def opened_img():
    global fn
    if fn:
        img = cv2.imread(fn)
        processed_img = opened_im(img)
        # processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for PIL
        processed_img = Image.fromarray(processed_img)
        processed_img = ImageTk.PhotoImage(processed_img)

        img4 = tk.Label(root, image=processed_img, height=200, width=200)
        img4.image = processed_img
        img4.place(x=600, y=400)
        image_labels.append(img4)
        
    else:
        print("Please select an image first.")        


def openimage():
   
    global fn
    global image_labels
    for label in image_labels:
        label.destroy()
    image_labels = []
    fileName = filedialog.askopenfilename(initialdir='C:/Users/AJAY/Desktop/project/melanoma-detection-main/final', title='Select image for Aanalysis ',
                               filetypes=[("all files", "*.*")])
    IMAGE_SIZE=200
    imgpath = fileName
    fn = fileName


#        img = Image.open(imgpath).convert("L")
    img = Image.open(imgpath)
    
    img = img.resize((IMAGE_SIZE,200))
    img = np.array(img)
#        img = img / 255.0
#        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)


    x1 = int(img.shape[0])
    y1 = int(img.shape[1])


#
#        gs = cv2.cvtColor(cv2.imread(imgpath, 1), cv2.COLOR_RGB2GRAY)
#
#        gs = cv2.resize(gs, (x1, y1))
#
#        retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(im)
    img = tk.Label(root, image=imgtk, height=200, width=200)
    
    #result_label1 = tk.Label(root, image=imgtk, width=250,height=250)
    #result_label1.place(x=300, y=100)
    img.image = imgtk
    img.place(x=300, y=100)
    image_labels.append(img)
    

frame_alpr = tk.LabelFrame(root, text=" --Process-- ", width=220, height=300, bd=5, font=('times', 14, ' bold '),bg="lawn Green")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=40, y=80)

button1 = tk.Button(frame_alpr, text=" Select_Image ", command=openimage,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button1.place(x=10, y=20)

remove_hairs_button = tk.Button(frame_alpr, text="Remove Hairs", command=remove_hairs_img,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
remove_hairs_button.place(x=10,y=70)

greyscale_button = tk.Button(frame_alpr, text="Grey Scale", command=convertToGreyScale_img,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
greyscale_button.place(x=10,y=120)

otsu_img_button = tk.Button(frame_alpr, text="Threshold", command=otsu_img,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
otsu_img_button.place(x=10,y=170)

final_img_button = tk.Button(frame_alpr, text="Final Image", command=masking_img,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
final_img_button.place(x=10,y=220)

frame_alpr = tk.LabelFrame(root, text=" --Models-- ", width=220, height=350, bd=5, font=('times', 14, ' bold '),bg="lawn Green")
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=40, y=400)

button1 = tk.Button(frame_alpr, text="SVM CLF", command=openimage,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
button1.place(x=10, y=20)

remove_hairs_button = tk.Button(frame_alpr, text="LR CLF", command=remove_hairs_img,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
remove_hairs_button.place(x=10,y=70)

greyscale_button = tk.Button(frame_alpr, text="DT CLF", command=convertToGreyScale_img,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
greyscale_button.place(x=10,y=120)

final_img_button = tk.Button(frame_alpr, text="RF CLF", command=masking_img,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
final_img_button.place(x=10,y=170)

final_img_button = tk.Button(frame_alpr, text="Bagging CLF", command=masking_img,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
final_img_button.place(x=10,y=220)

final_img_button = tk.Button(frame_alpr, text="XGBoost CLF", command=masking_img,width=15, height=1, font=('times', 15, ' bold '),bg="white",fg="black")
final_img_button.place(x=10,y=270)






# Run the main loop
root.mainloop()
