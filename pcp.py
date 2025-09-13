import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Load custom background image
background_img = cv2.imread("background.jpg")  # Replace with your image path
background_img = cv2.resize(background_img, (640, 480))  # Resize to match video frame

print("Camera warming up...")
while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        break

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define green color range
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])


    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Refine the mask (remove noise)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    # Invert mask
    mask_inv = cv2.bitwise_not(mask)

    # Segment out green area from custom background and rest of frame
    res1 = cv2.bitwise_and(background_img, background_img, mask=mask)  # green replaced with background
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)                 # rest of the frame

    # Combine both results
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    cv2.imshow("Green Screen Replacement", final_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
