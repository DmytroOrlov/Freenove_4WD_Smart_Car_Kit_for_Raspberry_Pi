import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)

# Capture frame
ret, frame = cap.read()
if ret:
  cv2.imwrite('image.jpg', frame)

cap.release()
