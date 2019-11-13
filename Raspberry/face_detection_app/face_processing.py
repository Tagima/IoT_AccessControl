#! /usr/bin/python3

# FACE DETECTION FOR DATA GATHERING
# ---------------------------------
# This algorithm gets frames from the webcam and wait for a face to appear.
# It checks if there's 2 eyes, normalize the picture and passes it to the neural network
# The neural network then classifies the picture and gets its label.
# This information is saved to be processed by the LoRa script.

import numpy as np		# Provides math functions to manipulate arrays, matrices and more
import cv2				# OpenCV2 library
import os.path          # Some system function
import subprocess       # For running bash commands
import time

data_path = '/var/tmp_app'  # Pasta montada na RAM
#data_path = 'face_data'

# ======================================================================================
#                                        FUNCTIONS                                      
# ======================================================================================



# ======================================================================================
#                                          MAIN                                         
# ======================================================================================
face_detected = 0

camera = cv2.VideoCapture(0)                # Camera object
sm_capture = 0                              # Controls state machine for capturing samples
sample_count = 1                            # Controls the amount of samples captured
padding_1 = 25                              # Padding value to let some space when straighting face
padding_2 = 5                               # Padding value to let some space when saving face

# ------------------------- Getting some properties from camera ------------------------

camera.set(3,640)                           # Setting width
camera.set(4,480)                           # Setting height
camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)    # Setting brightness

# ---------------------------------------- Setup ---------------------------------------

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')	    # Trained classifier object
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# ----------------------------------- Program's logic ----------------------------------

while (1):
    # Initilizing variables

    ret,frame = camera.read()                               # Read a frame from camera
    frame = cv2.flip(frame, flipCode=1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)          # Convert it to grayscale
    faces = face_cascade.detectMultiScale(                  # Setting detector
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(20,20)
    )

    rows_f, cols_f = frame.shape[:2]

    for (x,y,w,h) in faces:                         # Get coordinates from face detected
    
        cv2.rectangle(frame,(x-padding_1,y-padding_1),(x+w+padding_1,y+h+padding_1),(0,0,255),2)    # Draw the rectangle

        if y < padding_1 or y + h + padding_1 > rows_f or x < padding_1 or x + w + padding_1 > cols_f:
            print("Out of bounds")
            continue

        else:
            roi_gray = gray[y-padding_1:y+h+padding_1, x-padding_1:x+w+padding_1]
            roi_color = frame[y-padding_1:y+h+padding_1, x-padding_1:x+w+padding_1]

            # Getting face size for normalization processing
            face_size = [w, h]

            # Getting eyes positions for normalization processing
            eyes = eyes_cascade.detectMultiScale(frame[y:y+h, x:x+w], minSize=(50,50))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 255, 255), 1)

            print("Face detected")
            face_detected = 1

    time.sleep(0.5)
    
    if face_detected == 1:
        if roi_gray.all():

            # --------------------------------------------------------------------------
            # Processing image in order to feed our CNN
            # --------------------------------------------------------------------------
            if len(eyes) != 2:
                print("Not enough eyes in the image!")
                continue;

            print("Two eyes detected")

            rows, cols = roi_gray.shape[:2]

            # compute the angle between the eye centroids
            i = 0
            center_x = [0, 0]
            center_y = [0, 0]

            for (ex, ey, ew, eh) in eyes:
                center_x[i] = ex + ew/2;
                center_y[i] = ey + eh/2;
                i += 1;

            print("Center of eyes:")
            print("center_x: ", center_x[0], "  ", center_x[1])
            print("center_y: ", center_y[0], "  ", center_y[1])

            dY = center_y[1] - center_y[0];
            dX = center_x[1] - center_x[0];

            angle = np.degrees(np.arctan2(dY, dX))
            print("angle ", angle)

            # Correcting the angle
            if angle < -90:
                angle += 180
            else:
                if angle > 90:
                    angle -= 180

            # Rotating the picture
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)

            if cols == 0 or rows == 0:
                print("Could not retrieve the image dimensions.")
                print("Getting another sample.")
                continue
                
            img_normalized = cv2.warpAffine(roi_gray, M, (cols,rows))

            # Cropping image from center of picture
            rows_in, cols_in = img_normalized.shape[:2]
            y = int(round(rows_in/2));
            y_offset = int(round(face_size[1]/2));
            x = int(round(cols_in/2));
            x_offset = int(round(face_size[0]/2));

            print("information: ", y, " ", y_offset, " ", x, " ", x_offset)

            img_cropped = img_normalized[y-y_offset-padding_2:y+y_offset+padding_2, x-x_offset-padding_2:x+x_offset+padding_2];

            # Resizing to 224x224 since this is the size of all dataset of the chosen pre trained network
            img_resized = cv2.resize(img_cropped, dsize=(224,224), interpolation=cv2.INTER_CUBIC)

            print("Face normalized!")

            cv2.imwrite(data_path+"/sample_"+str(sample_count)+".jpg", img_resized)
            sample_count += 1;

            if sample_count == 6:   # Captura 5 amostras
                break;

            print("\n")

        roi_gray = np.zeros([224, 224])   # Zeroing the matrix in order to not get the same sample
        face_detected = 0

camera.release()
cv2.destroyAllWindows()

exit(0)