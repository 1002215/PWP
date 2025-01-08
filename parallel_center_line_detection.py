import cv2 as cv
import numpy as np
import math

#Emma Chetan Parallel Line and Finding Center Line PWP H


# Return the average slope and average intercept given an array of tuples with format (index, slope, intercept)
def average_slope_intercept(lines_si):

    average_slope = 0

    average_intercept = 0

    if len(lines_si) == 0:

        return None
    
    for l in lines_si:

        average_slope += l[1]

        average_intercept += l[2]

    return average_slope/len(lines_si), average_intercept/len(lines_si)


# Return the coordinates of the points for the extremes of each line. 
# Function takes in lines_si, an array of tuples of form (index, slope, intercept), vertical lines is True or False
# Function takes in an array of lines given by the Hough line detection (lines)
def points(lines_si, lines, vertical_lines):

    if len(lines_si) == 0:

        return None
    
    slope, intercept = average_slope_intercept(lines_si)

    x1, x2, y1, y2 = extremes_for_lines(lines_si, lines)

    if vertical_lines:
        # Replace x's with intercepts to rotate the lines back into vertical lines.
        return ((int(intercept), int(y1)), (int(intercept), int(y2)))
    
    else:
        # Calculate the y coordinate using a line equation: y = mx + b
        y1 = slope*x1+intercept

        y2 = slope*x2+intercept

        return ((int(x1), int(y1)), (int(x2), int(y2)))

# Return all the highest and lowest coordinates of each extreme detected by Hough lines
# Function takes in an array of lines given by the Hough line detection (lines)
# Function takes in an array of tuples of form (index, slope, intercept), vertical lines is Bool (lines_si)
def extremes_for_lines(lines_si, lines):
    # List of all x coordinates for all extremes of each Hough line detected
    xes = []
    # List of all x coordinates for all extremes of each Hough line detected
    ys = []

    for l in lines_si:

        idx = l[0]
        # Uses the index to find the x coordinates of all extremes of each Hough line detected, same procedure for y
        x1, x2 = lines[idx][0][0], lines[idx][0][2]

        y1, y2 = lines[idx][0][1], lines[idx][0][3]

        xes.append(x1)

        xes.append(x2)

        ys.append(y1)

        ys.append(y2)
    # Find all the highest and lowest extremes of each coordinate
    min_x = min(xes)

    max_x = max(xes)

    min_y = min(ys)

    max_y = max(ys)

    return min_x, max_x, min_y, max_y


# Return the points needed to draw the upper, lower, and center lines
# Function takes in an array of lines given by the Hough line detection
def make_lines(lines):
    # Format (index, slope, intercept), si standing for slope intercept
    line_si = []
    # Format (index, slope, intercept), si standing for slope intercept
    vertical_line_si = []

    vertical_lines = False

    upper_line = []

    lower_line = []

    idx = 0

    if lines is None:
            
            return [None, None]
    
    for line in lines:

        for x1, y1, x2, y2 in line:
            # Case if the line is vertical
            if abs(x1-x2) < 9:
                # To avoid dealing with an undefined slope, rotate the vertical lines into horizontal lines. Switch back the points later on.
                slope = 0

                intercept = x1

                vertical_line_si.append((idx, slope, intercept))

            else:
                # Equation to calculate slope
                slope = (y2 - y1) / (x2 - x1)
                # Calculating intercept of a line using line equation: y = mx + b
                intercept = y1 - (slope * x1)

            line_si.append((idx, slope, intercept))

        idx += 1
    # Case determining whether the lines are vertical
    if len(vertical_line_si) > len(lines)/2:

        line_si = vertical_line_si

        vertical_lines = True

    m_slope, m_intercept = average_slope_intercept(line_si)
    #Group each line detected into upper or lower lines based on their position above or below the average x intercept.
    for l in line_si:

        if m_intercept > l[2]:

            lower_line.append(l)
        else:

            upper_line.append(l)

    u_points = points(upper_line, lines, vertical_lines)

    l_points = points(lower_line, lines, vertical_lines)

    c_points = points(line_si, lines, vertical_lines)

    if l_points != None and u_points != None and c_points != None:

        x1, y1, x2, y2 = u_points[0][0], u_points[0][1], u_points[1][0], u_points[1][1]

        X1, Y1, X2, Y2 = l_points[0][0], l_points[0][1], l_points[1][0], l_points[1][1]
        # Center coordinates
        mx1 = int(x1+X1)/2

        my1 = int(y1+Y1)/2

        mx2 = int(x2+X2)/2

        my2 = int(y2+Y2)/2

        min_x, max_x, min_y, max_y = extremes_for_lines(line_si, lines)

        m1, b1 = average_slope_intercept(upper_line)

        m2, b2 = average_slope_intercept(lower_line)
        # Case if the upper and lower lines are not parallel
        if abs(m1-m2) > 6:
            # Normalization
            n1 = 1/(math.sqrt(m1*m1+1))

            n2 = 1/(math.sqrt(m2*m2+1))

            center_slope = (n2*m2-n1*m1)/(n2-n1)
            
            if m1 > m2:

                center_slope = -1/center_slope

            center_intercept = (n2*b2-n1*b1)/(n2-n1)
            
            mx1 = min_x

            mx2 = max_x

            my1 = mx1*center_slope+center_intercept

            my2 = mx2*center_slope+center_intercept

            c_points = ((int(mx1), int(my1)), (int(mx2), int(my2)))

        else:
            c_points = ((int(mx1), int(my1)), (int(mx2), int(my2)))

    return u_points, l_points, c_points


# Using points of the form ((x1, y1), (x2, y2)) representing each line, draw the line
# Function takes in the image, array of Hough lines detected, color of the parallel lines, and line thickness
def draw_lines(image, lines, color=[0, 0, 255], thickness=12):

    line_image = np.zeros_like(image)

    if lines is None:

        return None

    for i in range(len(lines)):

        line = lines[i]

        if line is not None:
            # Center line case since the center line is returned last by the make_lines function
            if i == len(lines)-1:
                # Center line is drawn in green
                cv.line(line_image, *line, [0,255,0], thickness+10)

            else:

                cv.line(line_image, *line, color, thickness)

        else:

            continue

    return cv.addWeighted(image, 1.0, line_image, 1.0, 0.0)



def frame_processor(img):

    canny_img = cv.Canny(img, 50, 200, None, 3)
    #https://www.learningaboutelectronics.com/Articles/Region-of-interest-in-an-image-Python-OpenCV.php used to create a square region of interest
    vertices = np.array([[(100, 100), (400, 100), (400,400), (100, 400)]], dtype=np.int32)

    mask = np.zeros_like(canny_img)

    ROI = cv.fillPoly(mask, vertices,255)

    region_of_interest = cv.bitwise_and(canny_img, ROI)

    lines = cv.HoughLinesP(region_of_interest, 1, np.pi / 180, 50, None, 50, 10)

    final_no_roi_drawn = draw_lines(img, make_lines(lines))
    # https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html used to draw the outline of the region of interest
    final = cv.polylines(final_no_roi_drawn, [vertices], True, (0,0,255), 1)

    return final


# https://stackoverflow.com/questions/2601194/displaying-a-webcam-feed-using-opencv-and-python/11449901#11449901 used to display the webcam feed and lines
camera = cv.VideoCapture(0)

cv.namedWindow("Emma Chetan Parallel and Centerline Detection PWP")

if camera.isOpened():

    success, frame = camera.read()

    frame = frame_processor(frame)

else:

    success = False

while success:

    cv.imshow("Emma Chetan Parallel and Centerline Detection PWP", frame)

    success, frame = camera.read()

    frame = frame_processor(frame)

    key = cv.waitKey(20)
    # Use esc button to exit
    if key == 27:

        break


cv.destroyWindow("Emma Chetan Parallel and Centerline Detection PWP")

camera.release()
