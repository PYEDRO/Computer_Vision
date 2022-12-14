
import cv2

from pd_func import prewittFilter



image_content = 'lua'

img_path = 'figs\lua.tif.jpg'

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

cv2.imshow('laplacian', img)

cv2.waitKey(0)
cv2.destroyAllWindows()


prewitt_h, prewitt_v, prewitt_image = prewittFilter(img)



cv2.imshow('h', prewitt_h)
cv2.imshow('v', prewitt_v)
cv2.imshow('prewitt', prewitt_image)

cv2.waitKey(0)
cv2.destroyAllWindows()



cv2.imwrite(
    './enhancement_filetrs/results_prewitt_filter/{}_horizontal_prewitt.png'.format(image_content), prewitt_h)
cv2.imwrite(
    './enhancement_filetrs/results_prewitt_filter/{}_vertical_prewitt.png'.format(image_content), prewitt_v)
cv2.imwrite(
    './enhancement_filetrs/results_prewitt_filter/{}_prewitt_image.png'.format(image_content), prewitt_image)

