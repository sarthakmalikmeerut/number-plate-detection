# from ultralytics import YOLO
# import cv2
# import cvzone
#
# import math
#
# vid = cv2.VideoCapture("C:\\Users\\sarth\\PycharmProjects\\OCR\\car_highway.mp4")
# model = YOLO("yolov8n")
#
# while True:
#     success, img = vid.read()
#     # succ , im = vid1.read()
#     results = model(img, stream=True)
#     # results1 = model(im,stream=True)
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#
#             w, h = x2 - x1, y2 - y1
#             cvzone.cornerRect(img, (x1, y1, w, h))
#
#             conf = math.ceil((box.conf[0] * 100)) / 100
#
#             cvzone.putTextRect(img, f'{conf}', (max(0, x1), max(35, y1)))
#         cv2.imshow("image", img)
#
#         cv2.waitKey(1)
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr
import time



img = cv2.imread('Cars52.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(cv2.cvtColor(gray,cv2.COLOR_BGR2RGB))
# plt.show()

noise_red = cv2.bilateralFilter(gray,11,17,17)
edge_detection = cv2.Canny(noise_red,30,200)
plt.imshow(cv2.cvtColor(edge_detection,cv2.COLOR_BGR2RGB))



keypoints = cv2.findContours(edge_detection.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

# print(location)



mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))


(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))



reader = easyocr.Reader(['en'])
results = reader.readtext(cropped_image)
print(results)
text = ''
for result in results:
    text += result[1]+''


print(text)
plt.show()


