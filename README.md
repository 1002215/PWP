The program takes in a constant video stream. For each frame, the program uses Hough Lines to identify two parallel lines. 
Given these two parallel lines, the program will calculate and display parallel lines (in red) and the center line (in green).
The Hough line detection is run on the frame after applying Canny edge detection. For each line detected using the Hough line
transformation, its slope and intercept is computed. Once the average y intercept of all the lines is calculated, each line is 
grouped into two lists containing upper or lower lines based on their position above or below the average y intercept. The upper 
and lower lines are drawn based off the average slope, intercept, and the point extremes for the lines in these two lists. The 
center line is computed by averaging out the coordinates of the extremes of the upper and lower line. Each of these three lines
are drawn with their respective colors using their coordinates. The square region of interest prevents the detection of lines 
outside of the desired parallel lines.
