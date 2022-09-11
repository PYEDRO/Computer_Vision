import cv2

from pd_func import histogramCalc




image_content = 'cameraman'

img_path = 'figs\lua.tif.jpg'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)




histogram = histogramCalc(image_content, img)




histogram_image = cv2.imread(histogram)

resized_histogram = cv2.resize(
    histogram_image,
    (int(histogram_image.shape[0]*0.6), int(histogram_image.shape[1] * 0.3)),
    interpolation=cv2.INTER_AREA)

cv2.imshow('unput image', img)
cv2.imshow('{} hisrogram'.format(image_content), resized_histogram)

cv2.waitKey(0)
cv2.destroyAllWindows()
