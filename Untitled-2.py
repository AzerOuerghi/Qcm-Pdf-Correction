import cv2
import numpy as np
import os
from wand.image import Image

pdf_path = os.path.join(os.getcwd(), 'images', 'exam paper.pdf')

# convert the second page of the PDF to an image
with Image(filename=pdf_path, resolution=300) as img:
    img.compression_quality = 99
    img.background_color = 'white'
    img.alpha_channel = 'remove'
    img = img.sequence[1]
    img = Image(img)


# load the processed image and detect shapes
img = cv2.imread('image_processed.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 50, 255, 0)
contours, hierarchy = cv2.findContours(thresh, 1, 2)

qcm = 0  # initialize the counter for rectangles with squares inside

print("Number of contours detected:", len(contours))

height, width, _ = img.shape
test_starts = 0
test_starts = 0
for i, cnt in enumerate(contours):
   x1, y1 = cnt[0][0]
   approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
   if len(approx) == 4:
      x, y, w, h = cv2.boundingRect(cnt)
      ratio = float(w)/h
      if ratio >= 0.9 and ratio <= 1.1:
         img = cv2.drawContours(img, [cnt], -1, (0, 255, 255), 3)
         cv2.putText(img, 'Square', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
      else:
         # check if the current QCM has a shared border with at least 2 other QCMs from at least 2 sides
         shared_cnts = []
         for j, other_cnt in enumerate(contours):
            if i == j:
               continue
            x2, y2 = other_cnt[0][0]
            other_approx = cv2.approxPolyDP(other_cnt, 0.01*cv2.arcLength(other_cnt, True), True)
            if len(other_approx) == 4:
               x_, y_, w_, h_ = cv2.boundingRect(other_cnt)
               if (abs(x1 - x_) < w and abs(y1 - y_) < h_) or (abs(x1 - x_ - w_) < w and abs(y1 - y_) < h_) or (abs(x1 - x_) < w and abs(y1 - y_ - h_) < h) or (abs(x1 - x_ - w_) < w and abs(y1 - y_ - h_) < h_):
                  shared_cnts.append(other_cnt)
         if len(shared_cnts) >= 2:
            cv2.putText(img, 'Test Start', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            test_starts += 1
         else:
            cv2.putText(img, 'Rectangle', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
         img = cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)

print("Number of test starts:", test_starts)



cv2.imshow("Shapes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()