import cv2

from pd_func import gaussianFilter


img = cv2.imread('figs\images.jpg', cv2.IMREAD_GRAYSCALE)

image_content = 'menina'


kernel_size_list = [3]  # k

number_of_iterations_list = [3]  # n

for k in kernel_size_list:
    for n in number_of_iterations_list:
        filtered_image, filtering_parameters = gaussianFilter(
            image_content, img, k, n, 1)

        print('filtering parameters:', filtering_parameters)


        file_name = './smoothing_filters/results_gaussian_filtering/{}_k{}n{}pdd{}_gaussian_filter.jpg'

        cv2.imwrite(file_name.format(
            filtering_parameters[0],  # image content
            filtering_parameters[1],  # kernel dimensions
            filtering_parameters[2],  # number of iterations
            filtering_parameters[3]),  # padding
            filtered_image)

        cv2.imshow('laplacian', filtered_image)

        cv2.waitKey(0)
