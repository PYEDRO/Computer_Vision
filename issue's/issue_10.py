import cv2


from pd_func import threshold


# Image loading =====================================================================================

image_content = 'lua'

img_path = 'figs\lua.tif.jpg'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


# Thresholding ======================================================================================

th_image, th_value = threshold(img, 120)


# Image displaying ==================================================================================

cv2.imshow('th image', th_image)

cv2.waitKey(0)

cv2.destroyAllWindows()


# Image saving ======================================================================================

cv2.imwrite(
    'threshhold/results_th/{}_th{}_threshold.png'.format(image_content, th_value), th_image)
