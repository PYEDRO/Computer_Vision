
import cv2

import numpy as np

from pd_func import meanFilter



img = cv2.imread('figs\lua.tif.jpg', cv2.IMREAD_GRAYSCALE)

image_content = 'lua'



kernel_size_list = [5]

number_of_iterations_list = [1, 5]

for k in kernel_size_list:
    for n in number_of_iterations_list:

        filtered_image, filtering_parameters = meanFilter(image_content, img, k, n, 1)

        print('filtering parameters:', filtering_parameters)



        file_name = './smoothing_filters/results_mean_filtering/{}_k{}n{}pdd{}_mean_filter.jpg'

        salved_img = cv2.imwrite(file_name.format(
            filtering_parameters[0],  # image content
            filtering_parameters[1],  # kernel dimensions
            filtering_parameters[2],  # number of iterations
            filtering_parameters[3]), # padding
            filtered_image)
        
        cv2.imshow('laplacian', filtered_image)

        cv2.waitKey(0)

        print(salved_img, 'ok')

       