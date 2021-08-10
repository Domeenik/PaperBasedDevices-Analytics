import numpy as np
import cv2

#### info ####
# Run with python version 3
# Stop script via 'ESC' (key=27)
# It is important to select an area that contains only the paper and no border


#### settings ####
# save data to csv-file
SAVE_DATA = True
# name of the video relative to the script path
INPUT_FILENAME = "./pbd_1000.mp4"
# name of the output csv-file relative to the script path
OUTPUT_FILENAME = f"./output_{INPUT_FILENAME.split('./')[1].split('.mp4')}.csv"
# length of the observed paper in mm (ruler)
AREA_WIDTH = 18
# different orientations can be archieved by: 0,1,2,3
ROTATION = 0
# threshold for the front-line-detection
# experiment with this one. A change of 20 is a lot.
THRESHOLD = 190
# prescaling for roi selection
PRESCALE = 1


#### script ####
# load video
cap = cv2.VideoCapture(INPUT_FILENAME)

# create csv-writer
f = open(OUTPUT_FILENAME, "w")

# select region of interest (roi)
if cap.grab():
    flag, frame = cap.retrieve()
    if not ROTATION == 0:
        frame = cv2.rotate(frame, ROTATION)
    if not PRESCALE == 1:
        frame = cv2.resize(frame, (frame.shape[1]*PRESCALE, frame.shape[0]*PRESCALE))
    region_of_interest = cv2.selectROI("selection", frame)
    r = [0,0,0,0]
    for i in range(len(r)):
        r[i] = int(region_of_interest[i] / PRESCALE)
    r_height = r[3]
    cv2.destroyWindow("selection")

frame_counter = 0
while True:
    # get frame
    if cap.grab():
        flag, frame = cap.retrieve()
        frame_counter += 1

        # rotate image
        if not ROTATION == 0:
            frame = cv2.rotate(frame, ROTATION)

        # crop frame to roi
        frame_cropped = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        if frame_cropped.shape[0] < 10:
            frame_cropped = cv2.resize(frame_cropped, (frame_cropped.shape[1], frame_cropped.shape[0]*4))

        # blur image
        frame_blurred = frame_cropped#cv2.medianBlur(frame_cropped, 5)
        frame_blurred = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    
        # get a clear border via threshholding
        ret, frame_thresh = cv2.threshold(frame_blurred, THRESHOLD, 255, cv2.THRESH_BINARY)

        # find contours and deep copy image
        contours, hierarchy = cv2.findContours(frame_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        frame_contour = frame_cropped.copy()
        
        # draw rect of roi
        frame = cv2.rectangle(frame, r, (255,0,255))
        if contours:
            contour_index = 0
            contour_size = 0

            # check for largest area
            for i in range(len(contours)):
                if len(contours[i]) > contour_size:
                    contour_index = i
                    contour_size = len(contours[i])
            
            # draw largest area and line
            cv2.drawContours(frame_contour, contours, 0, (0,0,255))
            bounding_box = cv2.boundingRect(contours[contour_index])
            frame_contour = cv2.rectangle(frame_contour, bounding_box, (0,255,255))
            frame = cv2.line(frame, (bounding_box[0] + r[0], bounding_box[1] + r[1]), (bounding_box[0] + r[0], bounding_box[1] + r[1] + r_height), (255,0,0))
        
        # stack the background-process images to one image
        frame_stitched = np.vstack((frame_cropped, cv2.cvtColor(frame_blurred, cv2.COLOR_GRAY2BGR), cv2.cvtColor(frame_thresh, cv2.COLOR_GRAY2BGR), frame_contour))
        frame_stitched = cv2.resize(frame_stitched, (frame_stitched.shape[1]*2, frame_stitched.shape[0]*2))

        # get current state
        max_width = r[2]
        if SAVE_DATA:
            if contours:
                data = round((AREA_WIDTH*(bounding_box[0])/max_width), 2)
                f.write(f"{frame_counter};{data};\n".replace(".", ","))
                print(frame_counter, f"  {data} mm")

    # check for keyboard input and show image
    if cv2.waitKey(1) == 27:
        break
    else:
        cv2.imshow('source', frame)
        cv2.imshow('backend', frame_stitched)

# close file
f.close()