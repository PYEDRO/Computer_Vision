import cv2

from pd_func import medianFilter

# Image loading =====================================================================================

img = cv2.imread(
    'figs\lua.tif.jpg', cv2.IMREAD_GRAYSCALE)

image_content = 'dbz'

# Filtering and Saving ==============================================================================

kernel_size_list = [3]

number_of_iterations_list = [1]

for k in kernel_size_list:
    for n in number_of_iterations_list:
        filtered_image, filtering_parameters = medianFilter(
            image_content, img, k, n, 1)

        print('filtering parameters:', filtering_parameters)

        # image saving

        file_name = './smoothing_filters/results_median_filtering/{}_k{}n{}pdd{}_median_filter.jpg'

        cv2.imwrite(file_name.format(
            filtering_parameters[0],  # image content
            filtering_parameters[1],  # kernel dimensions
            filtering_parameters[2],  # number of iterations
            filtering_parameters[3]),  # padding
            filtered_image)

        cv2.imshow('laplacian', filtered_image)

        cv2.waitKey(0)
