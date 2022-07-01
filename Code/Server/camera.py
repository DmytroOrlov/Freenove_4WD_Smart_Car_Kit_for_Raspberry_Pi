from picamera import PiCamera
from time import sleep
camera = PiCamera()
camera.resolution = (2592,1944)
#camera.exposure_compensation = 18
camera.capture('image.jpg')
