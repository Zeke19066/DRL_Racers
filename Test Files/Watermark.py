import cv2
import os
import numpy as np

# Screenshot functions do not capture mouse; it must be added.
def add_watermark(image, watermark):
    #Note: Watermak img must be loaded as cv2.imread("watermark.png", -1) #-1 is neccessary.
    h_img, w_img, _ = image.shape
    h_wm, w_wm, _ = watermark.shape

    y = 0
    x = int(w_img/2)-int(w_wm/2) #center of img offset by the center of the watermark.

    image_overlay = watermark[:, :, 0:3]
    alpha_mask = watermark[:, :, 3] / 255.0

    # Image ranges
    y1, y2 = max(0, y), min(image.shape[0], y + image_overlay.shape[0])
    x1, x2 = max(0, x), min(image.shape[1], x + image_overlay.shape[1])
    # Overlay ranges
    y1o, y2o = max(0, -y), min(image_overlay.shape[0], image.shape[0] - y)
    x1o, x2o = max(0, -x), min(image_overlay.shape[1], image.shape[1] - x)
    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return print('ScreenGrab.add_Mouse FAILED')
    channels = image.shape[2]
    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha
    for c in range(channels):
        image[y1:y2, x1:x2, c] = (alpha * image_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * image[y1:y2, x1:x2, c])

    return image
     
watermark = cv2.imread("1_up.png", -1) #-1 is neccessary
img = cv2.imread(f'mini_img_sample1.png')

output_img = add_watermark(img, watermark)
print(img.shape, output_img.shape)

filename = "output_img_watermark.png"
cv2.imwrite(filename, output_img)
cv2.imshow("Watermarked Image", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()