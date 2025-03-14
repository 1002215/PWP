'''Emma Chetan PWP Image Overlay Project Q3

The program will begin by taking in a live video stream and applying image processing to each frame. First, 
each frame shall be blurred using Gaussian blur, reducing the noise in the image and making it easier to process
in further steps. Each frame shall be converted into Hue, saturation, and Value (HSV) color space using an OpenCV 
function, allowing for the differentiation of colors based on their brightness. Since the lines of the road are 
all either yellow or white in the United States, OpenCV will be harnessed to create a mask so the program only 
executes on spaces detected as white or yellow. This is done by defining the upper and lower HSV values of white
and yellow. The frame is converted to grayscale to help simplify any algorithms applied from here on out. Canny
edge detection is then utilized, returning the edges of the image. Afterward, the program will invoke a triangular
region of interest to reduce noise and focus on the road lines. This mimics the perspective of a car driving down a road.

This region of interest will be altered to apply perspective transform. Perspective transform straightens objects that were
recorded at an angle, correcting any distortions as a result of the perspective. This will return an image of the lane 
lines as two straight lines, making it much easier to process and detect. Hough transform will be exercised, detecting any 
straight lines, even accounting for slight distortion. Using the points returned by the Hough function, the lines will be 
grouped and drawn in red based on their slope and intercept. The lane lines have been detected and drawn on the original frame. 
The Guo Hall skeletonizing algorithm will use these two lines and draw a pixel-wide centerline between them. Using polyfit., 
the pixels can be used to draw the complete centerline in green on the original frame. This iterates for each frame processed 
by the camera. For the compass rose, trigonometry will determine the direction of the vehicle. A right triangle will be drawn 
with the hypotenuse being the centerline. Using the angle of elevation and the tangent function, the exact degree can be 
calculated. The quadrant the angle is in will determine the direction. For example, if the degree measures about 10°, then 
it is apparent that the vehicle is turning right since the degree measure will be in the first quadrant close to the x-axis.

'''


import cv2 as cv
import numpy as np
import math
import logging
from datetime import datetime
from PIL import Image
import requests
from flask import Flask, request, jsonify, render_template, redirect, Response



app = Flask(__name__, template_folder="templates")



# Reads the file and appends information to a list.
with open("filename.txt", "r") as f:
    log_messages = []
    for line in f:
        log_messages.append(line.strip())



def save_list(lst, filename="filename.txt"):
    """Write a given list on a given file."""
    with open(filename, "w") as f:
        for item in lst:
            f.write(f"{item}\n")



def log_action(message):
    logging.info(message)
    log_messages.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
    save_list(log_messages)



def send_request(url):  
   x = requests.post(url)
   return x



@app.route('/logs', methods=['GET'])
def get_logs():
    return jsonify(log_messages)



@app.route('/buttons', methods=['GET', 'POST'])
def buttons():
    """
       Listens for POST requests at /buttons and performs different
       actions depending on the button pressed by the user.
       It also logs each action.
    """
    if request.method == 'POST':
       # Get JSON data from the buttons
       jsondata = request.get_json()
       action = jsondata.get('action')
       if action == 'fwd':
           send_request('http://192.168.240.20:5123/fwd')
           log_action(request)
       if action == 'bwd':
           send_request('http://192.168.240.20:5123/bwd')
           log_action(request)
       if action == 'right':
           send_request('http://192.168.240.20:5123/right')
           log_action(request)
       if action == 'left':
           send_request('http://192.168.240.20:5123/left')
           log_action(request)
       if action == 'stop':
           send_request('http://192.168.240.20:5123/stop')
           log_action(request)
    return render_template('buttons.html') # Use the buttons html file for the aesthetics
   


@app.route('/logs', methods = ['GET', 'POST'])
def console():
    "Show the front end displaying the logs on the console."
    return render_template('logs.html')
    


@app.route('/screen/', methods = ['GET', 'POST'])
def screen():
    """
        Show the screen separating the webite into four parts: one with the buttons, 
        one with the console log, and the rest as placeholders.
    """
    return render_template('screen.html')



def average_slope_intercept(lines_si):
    """
        Arguments: 
        lines_si, where each index is a tuple with the index, slope, 
        and intercept, of a line. (index, slope, intercept)

        Returns: 
        The average slope and intercept of the given list of line
        slopes and intercepts.
 
    """

    average_slope = 0

    average_intercept = 0

    if len(lines_si) == 0:
        return None

    for l in lines_si:
        average_slope += l[1]

        average_intercept += l[2]

    return average_slope / len(lines_si), average_intercept / len(lines_si)



def points(img, lines_si, lines, vertical_lines):
    """
        Arguments:
        img, an image of NumPy array format. 
        lines_si, an array of tuples of form (index, slope, intercept) with
        each index representing a different line.
        vertical_lines is True or False.

        Returns:
        Return the coordinates of the points for the start and end of the lines.

    """
    if len(lines_si) == 0:
        return None

    slope, intercept = average_slope_intercept(lines_si)

    x1, x2, y1, y2 = extremes_for_lines(lines_si, lines)

    if vertical_lines:
        # Replace x's with intercepts to rotate the lines back into vertical lines.
        return ((int(intercept), int(y1)), (int(intercept), int(y2)))

    else:
        x1 = 0
        x2 = img.shape[1]
        # Calculate the y coordinate using the line equation: y = mx + b
        y1 = slope * x1 + intercept

        y2 = slope * x2 + intercept

        return ((int(x1), int(y1)), (int(x2), int(y2)))



def extremes_for_lines(lines_si, lines):
    """
        Arguments:
        lines_si, an array of tuples of form (index, slope, intercept) with
        each index representing a different line.
        lines, array of lines given by the Hough line detection.

        Returns:
        Return all the highest and lowest coordinates of each extreme detected by Hough lines.

    """
    # List of all x coordinates for all extremes of each Hough line detected
    xes = []
    # List of all y coordinates for all extremes of each Hough line detected
    ys = []

    for l in lines_si:
        idx = l[0]
        # Uses the index to find the x coordinates of all extremes of each Hough line detected, 
        # same procedure for y
        x1, x2 = lines[idx][0][0], lines[idx][0][2]
        y1, y2 = lines[idx][0][1], lines[idx][0][3]
        xes.append(x1)
        xes.append(x2)
        ys.append(y1)
        ys.append(y2)
    
    min_x = min(xes)
    max_x = max(xes)
    min_y = min(ys)
    max_y = max(ys)

    return min_x, max_x, min_y, max_y

def perspective_transform(img):
    """
        Transforms a road image to a birds-eye view.
    
        Argument:
        img, an image of the NumPy array format.

        Returns:
        Returns the image with perspective transform applied, warped
        Returns the inverse matrix that allows for reversal
        of transformation.

    """
    img_size = (img.shape[1], img.shape[0])
    offset = 300
    src = np.float32([
        (190, 720),  # bottom-left corner
        (596, 447),  # top-left corner
        (685, 447),  # top-right corner
        (1125, 720)  # bottom-right corner
    ])

    # Destination points are to be parallel, taking into account the image size
    dst = np.float32([
        [offset, img_size[1]],  # bottom-left corner
        [offset, 0],  # top-left corner
        [img_size[0] - offset, 0],  # top-right corner
        [img_size[0] - offset, img_size[1]]  # bottom-right corner
    ])
    M = cv.getPerspectiveTransform(src, dst)
    M_inv = cv.getPerspectiveTransform(dst, src)
    warped = cv.warpPerspective(img, M, img_size)

    return warped, M_inv



def make_lines(img, lines):
    """
        Generates the points for the right and left lanes as
        well as the centerline for drawing.

        Arguments:
        img, an image of the NumPy array format.
        A list returned by the OpenCV Hough Lines transformation.

        Returns:
        Points of left lane (l_points), right lane, and centerline needed
        for line drawing. 
    """
    
    line_si = []
    vertical_line_si = []
    vertical_lines = False
    upper_line = []
    lower_line = []
    idx = 0

    if lines is None:
        return [None, None]

    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            # Case if the line is vertical
            if abs(x1 - x2) < 9:
                # To avoid dealing with an undefined slope, rotate the 
                # vertical lines into horizontal lines. 
                # Switch back the points later on.
                slope = 0
                intercept = x1
                vertical_line_si.append((idx, slope, intercept))

            else:
                slope = fit[0]
                intercept = fit[1]

            line_si.append((idx, slope, intercept))

        idx += 1
    
    # Case determining whether the lines are vertical by evaluating
    # whether the majority of lines are vertical
    if len(vertical_line_si) > len(lines) / 2:
        line_si = vertical_line_si
        vertical_lines = True

    m_slope, m_intercept = average_slope_intercept(line_si)
    # Group each line detected into upper or lower lines based on their 
    # position above or below the average x intercept.
    for l in line_si:

        if m_intercept > l[2]:
            lower_line.append(l)

        else:
            upper_line.append(l)

    u_points = points(img, upper_line, lines, vertical_lines)
    l_points = points(img, lower_line, lines, vertical_lines)
    c_points = points(img, line_si, lines, vertical_lines)

    if l_points != None and u_points != None and c_points != None:

        x1, y1, x2, y2 = u_points[0][0], u_points[0][1], u_points[1][0], u_points[1][1]
        X1, Y1, X2, Y2 = l_points[0][0], l_points[0][1], l_points[1][0], l_points[1][1]

        # Center coordinates
        mx1 = int(x1 + X1) / 2
        my1 = int(y1 + Y1) / 2
        mx2 = int(x2 + X2) / 2
        my2 = int(y2 + Y2) / 2

        min_x, max_x, min_y, max_y = extremes_for_lines(line_si, lines)

        m1, b1 = average_slope_intercept(upper_line)
        m2, b2 = average_slope_intercept(lower_line)

        # Case if the upper and lower lines are not parallel
        if abs(m1 - m2) >0.5:
            # Given two line equations, the normalization of lines
            # can be used to determine the slope and intercept of the centerline.
            # Equation is n = 1/(sqrt{m^2+1})
            n1 = 1 / (math.sqrt(m1 * m1 + 1))
            n2 = 1 / (math.sqrt(m2 * m2 + 1))
            center_slope = (n2 * m2 - n1 * m1) / (n2 - n1)

            if m1 > m2:
                center_slope = -1 / center_slope

            center_intercept = (n2 * b2 - n1 * b1) / (n2 - n1)

            mx1 = min_x
            mx2 = max_x
            my1 = mx1 * center_slope + center_intercept
            my2 = mx2 * center_slope + center_intercept

            c_points = ((int(mx1), int(my1)), (int(mx2), int(my2)))

        else:
            c_points = ((int(mx1), int(my1)), (int(mx2), int(my2)))

    return u_points, l_points, c_points



def draw_lines(image, lines, color=[0, 0, 255], thickness=12):

    """
        Use the given lines to draw the red lane lines and the green
        centerline on a given image.

        Arguments:
        img, an image of the NumPy array format.
        lines, list of points of the form ((x1, y1), (x2, y2)) representing each line. Returned 
        by make_lines.
        color is set to red and the thickness is set to 12.

        Returns:
        Returns the NumPy formatted image with the red lane lines drawn and the green
        centerline drawn.
    """

    line_image = np.zeros_like(image)

    if lines is None:
        return None

    for i in range(len(lines)):

        line = lines[i]

        if line is not None:
            # Center line case since the center line is returned last by the make_lines function
            if i == len(lines) - 1:
                # Center line is drawn in green
                cv.line(line_image, *line, [0, 255, 0], thickness + 10)
                pass

            else:
                cv.line(line_image, *line, color, thickness)

        else:
            continue

    return cv.addWeighted(image, 1.0, line_image, 1.0, 0.0)



def make_mask(img):
    """ Create a mask used for filtering white and yellow objects in a given image."""
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    yellow_lower = np.array([10, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    mask_yellow = cv.inRange(hsv, yellow_lower, yellow_upper)

    lower_white = np.array([0, 0, 237])
    upper_white = np.array([255, 50, 255])

    mask_white = cv.inRange(hsv, lower_white, upper_white)
    # Combine the white and yellow mask into one mask.
    mask_yw = cv.bitwise_or(mask_white, mask_yellow)

    return mask_yw



def roi(img):
    """ Create a region of interest in the shape of a triangle."""
    height = img.shape[0]
    width = img.shape[1]

    # Coordinates for vertices.
    lower_left = (0-350, height)
    lower_right = (width+300, height)
    top_left = (width - 300, int(height / 1.5))

    vertices = np.array([[lower_left, 
                          lower_right, 
                          top_left]], dtype=np.int32)

    mask = np.zeros_like(img)

    ROI = cv.fillPoly(mask, vertices, (255, 255, 255))

    region_of_interest = cv.bitwise_and(img, ROI)

    return region_of_interest, vertices



def paste_arrow(frame, arrow_img):
    "Paste a given arrow image onto the frame."
    frame_pil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)).convert("RGBA")
    frame_pil.paste(arrow_img, (0, 0), mask=arrow_img)
    frame_bgr = cv.cvtColor(np.array(frame_pil), cv.COLOR_RGBA2BGR)

    return frame_bgr



def frame_processor(img):
    """
        This is the main pipeline for the image overlay. frame_processor
        adds an image overlay with red lane lines and a green centerline on a road.

        Argument:
        img, an image or frame to be processed.

        Returns:
        An image of NumPy format with the red lane lines and green centerline drawn.
    """
    img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
    img = cv.resize(img, (540,960))
    rows, columns = img.shape[0], img.shape[1]
    img = img[10:rows - 200, 10:columns - 10]

    region_of_interest, vertices = roi(img)
    
    mask_yw = make_mask(region_of_interest)

    canny_img = cv.Canny(mask_yw, 50, 200, None, 3)

    lines = cv.HoughLinesP(canny_img, 1, np.pi / 180, 50, None, 50, 10)

    # Get points of right lane line, left lane line, and centerline
    last_lines = make_lines(img, lines)

    if last_lines != None :

        # Points of the centerline are the last index
        c_points = last_lines[-1]

        if c_points != None:
            ((x1, y1), (x2, y2)) = c_points

            # Create vectors to determine direction of the y-coordinates
            # and x-coordinates for inverse tangent.
            diry = y2 - y1
            dirx = x2 - x1

            # atan2 is the math library's inverse tangent function.
            angle = math.atan2(diry, dirx)

            # If the value is negative, the direction of left or right will be
            # flipped due to unit circle properties.
            negative = False

            if angle < 0:
                angle += math.pi
                negative = True

            if angle > math.pi/4 and angle < 3*math.pi/4:
                if negative :
                    dir = "up"
                else:
                    dir = "up"

            elif angle > 3*math.pi/4:
                if negative :
                    dir = "right"
                else:
                    dir = "left"  

            elif angle < math.pi/4:
                if negative :
                    dir = "right"
                else:
                    dir = "left" 

            else:
                dir = "up"

            img = paste_arrow(img, arrows[dir])

    final = draw_lines(img, make_lines(img, lines))

    return final



arrows ={"up": "up.png", 
          "left": "left.png",
          "right": "right.png"}

for arrow in arrows:
    img = Image.open(arrows[arrow]).convert("RGBA")
    arrows[arrow] = img.resize((100,100))

  

camera = cv.VideoCapture("Video.mov")

def normal_video():
    """Encode each frame of a constant, raw video stream."""
    while True:

        success, frame = camera.read()
        if not success:
            break

        frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
        if frame is not None:    
        # Encode the grayscale frame
            ret2, buffer2 = cv.imencode('.jpg', frame)

            if not ret2:
                print("Error: Failed to encode grayscale frame")
                break
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer2.tobytes() + b'\r\n')



def frames():
    """Apply image overlay on each frame of a video stream and encode each frame."""
    while True:

        success, frame = camera.read()

        if not success:
            break

        frame = frame_processor(frame)
        if frame is not None:    
        # Encode the frame with image overlay.
            ret2, buffer2 = cv.imencode('.jpg', frame)

            if not ret2:
                print("Error: Failed to encode grayscale frame")
                break
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer2.tobytes() + b'\r\n')
        # resize image



@app.route('/')
def index():
   """Render the main webpage."""
   return render_template('video_feed.html')



@app.route('/color_feed')
def color_feed():
   """Endpoint for the color video feed."""
   return Response(frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/gray_feed')
def gray_feed():
   """Endpoint for the grayscale video feed."""
   return Response(normal_video(), mimetype='multipart/x-mixed-replace; boundary=frame')



# Runs the API to access the webservers.
if __name__=="__main__":
  app.run(host='127.0.0.1', debug=True, port=5000, use_reloader=False)
