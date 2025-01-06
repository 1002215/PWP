import cv2 as cv
import numpy as np
import math
from flask import Flask, render_template, Response

#src_img = cv.imread("testlines.jpeg")

camera = cv.VideoCapture(0)

app = Flask(__name__)


def average_slope_intercept(lines_si): # Where this is an array of tuples (index, slope, intercept)
    average_slope = 0
    average_intercept = 0
    if len(lines_si) == 0:
        return None
    for l in lines_si:
        average_slope += l[1]
        average_intercept += l[2]
    return average_slope/len(lines_si), average_intercept/len(lines_si)

def points(lines_si, lines, vertical_lines): # Where lines_si is an array of tuples of form (index, slope, intercept), vertical lines is Bool
    if len(lines_si) == 0:
        return None
    slope, intercept = average_slope_intercept(lines_si)
    x1, x2, y1, y2 = extremes_for_lines(lines_si, lines)
    if vertical_lines:
        return ((int(intercept), int(y1)), (int(intercept), int(y2)))
    else:
        y1 = slope*x1+intercept
        y2 = slope*x2+intercept
        return ((int(x1), int(y1)), (int(x2), int(y2)))

def extremes_for_lines(lines_si, lines):
    xes = []
    ys = []
    for l in lines_si:
        idx = l[0]
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
        

def make_lines(image, lines):
    #BTDUBS if you know upper and lower just mathematically do middle, so transversals won't affect much
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    line_si = [] #(index, slope, intercept)
    vertical_line_si = []
    vertical_lines = False
    upper_line = []
    lower_line = []
    idx = 0
    if lines is None:
            return [None, None]
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x1-x2) < 6:
                slope = 0
                intercept = x1
                vertical_line_si.append((idx, slope, intercept))
            else:
                slope = (y2 - y1) / (x2 - x1)
                # calculating intercept of a line
                intercept = y1 - (slope * x1)
            line_si.append((idx, slope, intercept))
        idx += 1
    
    if len(vertical_line_si) > len(lines)/2:
        line_si = vertical_line_si
        vertical_lines = True

    m_slope, m_intercept = average_slope_intercept(line_si)
    for l in line_si:
        if m_intercept > l[2]:
            lower_line.append(l)
        else:
            upper_line.append(l)

    u_points = points(upper_line, lines, vertical_lines)
    l_points = points(lower_line, lines, vertical_lines)
    c_points = points(line_si, lines, vertical_lines)

    return u_points, l_points, c_points

def draw_lines( image, lines, color=[0, 0, 255], thickness=12):
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image (video frame in our case).
            lines: The output lines from Hough Transform.
            color (Default = red): Line color.
            thickness (Default = 12): Line thickness.
    """
    line_image = np.zeros_like(image)
    if lines is None:
        return None

    for i in range(len(lines)):
        line = lines[i]
        if line is not None:
            if i == len(lines)-1:
                cv.line(line_image, *line, [0,255,0], thickness+10)
            else:
                cv.line(line_image, *line, color, thickness)
        else:
            continue

    return cv.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def frame_processor(img):
    canny_img = cv.Canny(img, 50, 200, None, 3)

    lines = cv.HoughLinesP(canny_img, 1, np.pi / 180, 50, None, 50, 10)

    '''for i in range(0, len(lines)):
        lin = lines[i][0]
        cv.line(img, (lin[0], lin[1]), (lin[2], lin[3]), (0, 0, 255), 3, cv.LINE_AA)
'''

    final = draw_lines(img, make_lines(img, lines))

    return final

def generate_gray_frames():
    #Generate grayscale video frames.
    while True:
        success, frame = camera.read()
        if not success:
            break
        # Convert to grayscale
        fixed = frame_processor(frame)

        if fixed is not None:    
        # Encode the grayscale frame
            ret2, buffer2 = cv.imencode('.jpg', fixed)
            if not ret2:
                print("Error: Failed to encode grayscale frame")
                break


            # Yield grayscale frame
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer2.tobytes() + b'\r\n')

def generate_color_frames():
    """Generate color video frames."""
    while True:
        success, frame = camera.read()
        if not success:
            break
        # Encode the color frame
        ret1, buffer1 = cv.imencode('.jpg', frame)
        if not ret1:
            print("Error: Failed to encode color frame")
            break


        # Yield color frame
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer1.tobytes() + b'\r\n')


@app.route('/')
def index():
   """Render the main webpage."""
   return render_template('video.html')




@app.route('/color_feed')
def color_feed():
   """Endpoint for the color video feed."""
   return Response(generate_color_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/gray_feed')
def gray_feed():
   """Endpoint for the grayscale video feed."""
   return Response(generate_gray_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
 app.run(host='0.0.0.0', debug=True, port=8000)




'''

frame_processor(floor)


# Takes in a line, its starting point (y1), and its ending point (y2)
def points_from_y(y1, y2, lines, line):
    if line is None:
        return None
    slope, intercept = line[0], line[1]
    if abs(slope) < 0.0000000000001:
        x1 = lines[0][0][0]
        x2 = lines[0][0][2]
        y1 = int(y1)
        y2 = int(y2)
    else:
        x1 = int((y1 - intercept) / slope) # Derived from line equation y = slope*x + intercept, where x1 corresponds to y1
        x2 = int((y2 - intercept) / slope)
        y1 = int(y1)
        y2 = int(y2)
    return ((x1, y1), (x2, y2))


def slope_and_intercept(lines):
    line1 = []
    line2 = []

    l,r = lines[0][0], lines[-1][0]
    x1, y1, x2, y2 = l
    X1, Y1, X2, Y2 = r
    slope1 = (y2 - y1) / (x2 - x1)
    intercept1 = y2 - (slope1*x1)
    line1.append(slope1)
    line1.append(intercept1)

    slope2 = (Y2 - Y1) / (X2 - X1)
    intercept2 = Y2 - (slope2*X2)
    line2.append(slope2)
    line2.append(intercept2)

    return line1, line2

'''
