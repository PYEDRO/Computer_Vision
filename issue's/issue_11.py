import cv2

from pd_func import multiThreshold


# Image loading =====================================================================================

image_content = 'lena'

img_path = 'figs\lena_color_256.tif'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


# Thresholding ======================================================================================

th_image, th1_value, th2_value = multiThreshold(img, 30, 160)


# Image displaying ==================================================================================

cv2.imshow('th image', th_image)

cv2.waitKey(0)

cv2.destroyAllWindows()


# Image saving ======================================================================================

cv2.imwrite('threshhold/results_multi_th/{}_th1_{}_th2_{}_threshold.png'.
            format(image_content, th1_value, th2_value), th_image)
