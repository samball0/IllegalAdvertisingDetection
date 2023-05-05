import pytesseract
import numpy as np
import argparse
import cv2
import pytesseract
from pytesseract import Output
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt
import os


def detect_and_recognise_text(filename):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    confThreshold = 0.2
    model = "frozen_east_text_detection.pb"
    net = cv2.dnn.readNet(model)

    directory = os.path.dirname(filename)
    file = os.path.splitext(os.path.basename(filename))[0]
    frame = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    orig = frame.copy()
    inpWidth = 640
    inpHeight = 640
    height_ = frame.shape[0]
    width_ = frame.shape[1]
    rW = width_ / float(inpWidth)
    rH = height_ / float(inpHeight)

    frame = cv2.resize(frame, (inpWidth, inpHeight))
    (H, W) = frame.shape[:2]

    #blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (42, 164, 26), swapRB=True, crop=False)

    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")

    net.setInput(blob)
    output = net.forward(outputLayers)
    scores = output[0]
    geometry = output[1]

    height = scores.shape[2]
    width = scores.shape[3]
    rects = []
    confidences = []
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]

        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < confThreshold):
                continue

            # Calculate offset
            offsetX = x * 4
            offsetY = y * 4
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = np.cos(angle)
            sinA = np.sin(angle)

            # Use geometry volume to derive width and height of bounding box
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Compute starting and ending (x, y)-coordinates for text prediction bounding box
            endX = int(offsetX + (cosA * x1_data[x]) + (sinA * x2_data[x]))
            endY = int(offsetY - (sinA * x1_data[x]) + (cosA * x2_data[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Add bounding box coordinates and probability score to lists
            rects.append((startX, startY, endX, endY))
            confidences.append(score)

    boxes = non_max_suppression(np.array(rects), probs = confidences)
    results = []
    text = ""
    #indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold, nmsThreshold)
    for (startX, startY, endX, endY) in boxes:
        margin = 4
        # Scale bounding box coordinates based on ratios
        startX = int(startX * rW - margin)
        startY = int(startY * rH - margin)
        endX = int(endX * rW + margin)
        endY = int(endY * rH + margin)

        # Draw bounding box on image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (70, 215, 50), 2)

        r = orig[startY:endY, startX:endX]
        #r = cv2.resize(r, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        if r.size == 0:
            break 

        r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY) #Convert image to greyscale
        #kernel = np.ones((1, 1), np.uint8)
        #r = cv2.dilate(r, kernel, iterations=1)    
        #r = cv2.erode(r, kernel, iterations=1)
        #r = cv2.threshold(r, 190, 255, cv2.THRESH_BINARY)[1] #Apply threshold effect
        r = cv2.adaptiveThreshold(r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        #configuration setting to convert image to string.  
        configuration = ("-l eng --oem 1 --psm 8")
        ##This will recognize the text from the image of bounding box
        text = pytesseract.image_to_string(r, config=configuration)
        #print(text)
        results.append(((startX, startY, endX, endY), text))

    # cv2.imshow("Text detect",orig)
    # cv2.imwrite("output3.png",orig)
    # cv2.waitKey(0)

    #Display the image with bounding box and recognized text
    orig_image = orig.copy()

    full_text = ""
    # Moving over the results and display on the image
    for ((start_X, start_Y, end_X, end_Y), text) in results:
        # display the text detected by Tesseract
        #print("{}\n".format(text))

        # Displaying text
        text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
        cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
            (0, 0, 255), 2)
        cv2.putText(orig_image, text, (start_X, start_Y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255, 255, 51), 2)
        full_text += text
        
    new_filename = os.path.join(directory, "{}_text.png".format(file))
    cv2.imwrite(new_filename, orig_image)
    return full_text, new_filename