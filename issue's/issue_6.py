import cv2

from pd_func import sobelFilter



image_content = 'lavotanovo'

img_path = 'figs\lua.tif.jpg'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)



sobel_h, sobel_v, sobel_image = sobelFilter(img)



cv2.imshow('h sobel', sobel_h)
cv2.imshow('v sobel', sobel_v)
cv2.imshow('sobel image', sobel_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Image saving ======================================================================================

cv2.imwrite(
    './enhancement_filetrs/results_sobel_filtering/{}_horizontal_sobel.png'.format(image_content), sobel_h)
cv2.imwrite(
    './enhancement_filetrs/results_sobel_filtering/{}_vertical_sobel.png'.format(image_content), sobel_v)
cv2.imwrite(
    './enhancement_filetrs/results_sobel_filtering/{}_sobel_image.png'.format(image_content), sobel_image)
