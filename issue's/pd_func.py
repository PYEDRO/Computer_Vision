
import matplotlib.pyplot as plt
import numpy as np
import math as m

# MEAN FILTER =======================================================================================

def meanFilter(image_content, img, k, n, padding):
       

    filtering_parameters_list = []
    
    filtering_parameters_list.append(image_content)
    filtering_parameters_list.append(k)
    filtering_parameters_list.append(n)
    filtering_parameters_list.append(padding)

    kernel = (1/k**2)*np.ones((k, k))

    c = int(k/2) # Just do simplify; k is the kernel dimension (k x k)
 

    if padding == 1:

        line, column = (img.shape)

        holdpdd = np.zeros(((line + 2 * c), (column + 2 * c))).astype(np.uint8)  

        new_line, new_column = (holdpdd.shape)

        # Based on the Professor Navar's lecture.

        holdpdd[c: new_line - c, c: new_column - c] = img
        
        mean_image = holdpdd.copy()
       
        #print('Padded image dimensions:', holdpdd.shape)


        # Convolution

        for i in range(n): 
            for x in range(c,mean_image.shape[0]-c):
                for y in range(c,mean_image.shape[1]-c):
            
                    lol = mean_image[ x - c : x + c + 1 , y - c : y + c + 1 ]
                    
                    mean = (lol * kernel).sum()
                    
                    mean_image [x,y] = round(mean)

        # To remove the padding

        final_image = np.zeros((img.shape[0] , img.shape[1]))

        final_image = mean_image[ c : new_line - c , c : new_column - c ] 


    # If padding is not wanted

    else:

        mean_image = img.copy()

        for i in range(n): 
            for x in range(c,mean_image.shape[0]-c):
                for y in range(c,mean_image.shape[1]-c):
            
                    lol = mean_image[ x - c : x + c + 1 , y - c : y + c + 1 ]
                    
                    mean = (lol * kernel).sum()
                                            
                    mean_image [x,y] = round(mean)
        
        final_image = np.zeros((img.shape[0] , img.shape[1]))
        final_image = mean_image

    
    
    return final_image, filtering_parameters_list


# MEDIAN FILTER =====================================================================================

def medianFilter(image_content, img, k, n, padding):

    filtering_parameters_list = []
    
    filtering_parameters_list.append(image_content)
    filtering_parameters_list.append(k)
    filtering_parameters_list.append(n)
    filtering_parameters_list.append(padding)

    kernel = np.ones((k, k))

    c = int(k/2) # Just do simplify; k is the kernel dimension (k x k)
 

    if padding == 1:

        line, column = (img.shape)

        holdpdd = np.zeros(((line + 2 * c), (column + 2 * c))).astype(np.uint8)  

        new_line, new_column = (holdpdd.shape)

        # Based on the Professor Navar's lecture.

        holdpdd[c: new_line - c, c: new_column - c] = img
        
        median_image = holdpdd.copy()
       
        #print('Padded image dimensions:', holdpdd.shape)


        # Convolution

        for i in range(n): 
            for x in range(c,median_image.shape[0]-c):
                for y in range(c,median_image.shape[1]-c):
            
                    lol = median_image[ x - c : x + c + 1 , y - c : y + c + 1 ]
                    
                    median = np.median(lol * kernel)
                    
                    median_image[x,y] = m.ceil(median)

        # To remove the padding

        final_image = np.zeros((img.shape[0] , img.shape[1]))

        final_image = median_image[ c : new_line - c , c : new_column - c ] 


    # If padding is not wanted

    else:

        median_image = img.copy()

        for i in range(n): 
            for x in range(c,median_image.shape[0]-c):
                for y in range(c,median_image.shape[1]-c):
            
                    lol = median_image[ x - c : x + c + 1 , y - c : y + c + 1 ]
                    
                    median = lol * kernel

                    median_list = []

                    #Alternative to the np.median() method

                    for h in range(median.shape[0]):
                        for j in range(median.shape[1]):

                            median_list.append(median[h][j]) # Could have used numpy reshape
                            
                            sorted_list = sorted(median_list)


                    median_pixel_value = sorted_list[m.floor(len(median_list)/2)]

                    median_image [x,y] = round(median_pixel_value)
        

        final_image = np.zeros((img.shape[0] , img.shape[1]))
        final_image = median_image
   
    
    return final_image, filtering_parameters_list


# GAUSSIAN FILTER ===================================================================================

# To generate the gaussian kernel. Based on the Professor Navar's code and Gonzalez and Woods (2008)

def gaussianKernel(d1, d2):
  
  x, y = np.mgrid[0:d2, 0:d1]
  x = x-d2/2
  y = y-d1/2
  sigma = 4 # std deviation
  a = np.exp( -(x**2 + y**2) / (2 * sigma**2) )
  return a / a.sum()


def gaussianFilter(image_content, img, k, n, padding):

    filtering_parameters_list = []
    
    filtering_parameters_list.append(image_content)
    filtering_parameters_list.append(k)
    filtering_parameters_list.append(n)
    filtering_parameters_list.append(padding)

    gaussian_kernel = gaussianKernel(k, k)

    c = int(k/2) # Just do simplify; k is the kernel dimension (k x k)

    # With Padding

    if padding == 1:

        line, column = (img.shape) 

        holdpdd = np.zeros( ((line + 2 * c), (column + 2 * c)) ).astype(np.uint8)   

        new_line, new_column = (holdpdd.shape)

        holdpdd [ c : new_line - c , c : new_column - c ] = img # Based on the Professor Navar's lecture.

        gaussian_image = holdpdd.copy()
            
        print(holdpdd.shape)

        # Convolution

        for i in range(n): 
            for x in range(c,gaussian_image.shape[0]-c):
                for y in range(c,gaussian_image.shape[1]-c):
                    
                    lol = gaussian_image[ x - c : x + c + 1 , y - c : y + c + 1 ]
                    
                    gaussian_conv = (lol * gaussian_kernel).sum()
                    
                    gaussian_image [x,y] = round(gaussian_conv)

        # To remove the padding

        final_image = np.zeros((img.shape[0] , img.shape[1]))

        final_image = gaussian_image[ c : new_line - c , c : new_column - c ] 


    # Without padding

    else:
 
        gaussian_image = img.copy()

        for i in range(n): 
            for x in range(c,gaussian_image.shape[0]-c):
                for y in range(c,gaussian_image.shape[1]-c):
                
                    lol = gaussian_image[ x - c : x + c + 1 , y - c : y + c + 1 ]
                    
                    gaussian_conv = (lol * gaussian_kernel).sum()
                        
                    gaussian_image [x,y] = round(gaussian_conv)
        
        final_image = np.zeros((img.shape[0] , img.shape[1]))
        final_image = gaussian_image

    return final_image, filtering_parameters_list


def normalizeImage(v):

  v = (v - v.min()) / (v.max() - v.min())

  result = (v * 255).astype(np.uint8)

  #cv2_imshow(result)

  return result

# LAPLACIAN FILTER ==================================================================================


def laplacianFilter(image_content, img, kernel_type, n, padding):

    c = 1

    filtering_parameters_list = []

    filtering_parameters_list.append(image_content)
    filtering_parameters_list.append(kernel_type)
    filtering_parameters_list.append(n)
    filtering_parameters_list.append(padding)

    if kernel_type == 0:

        kernel_laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    elif kernel_type == 1:
        kernel_laplacian = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

    elif kernel_type == 2:
        kernel_laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    else:
        kernel_laplacian = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    # Padding
    if padding == 1:
        line, column = (img.shape)

        holdpdd = np.zeros(((line + 2 * c), (column + 2 * c)))

        new_line, new_column = (holdpdd.shape)

        holdpdd[c: new_line - c, c: new_column - c] = img

    else:

        line, column = (img.shape)

        holdpdd = img.copy()

        new_line, new_column = (holdpdd.shape)

    # Convolution

    laplacian_image = holdpdd.copy()  # Based on the Professor Navar's code.

    for i in range(n):
        for x in range(c, holdpdd.shape[0]-c):
            for y in range(c, holdpdd.shape[1]-c):

                lol = holdpdd[x - c: x + c + 1, y - c: y + c + 1]

                laplacian_conv = (lol * kernel_laplacian).sum()

                laplacian_image[x, y] = round(laplacian_conv)

    # Removing padding and image saving

    final_image = np.zeros((line, column))

    final_image = laplacian_image[c: new_line - c, c: new_column - c]

    print('img shape:', final_image.shape, ';',
          'laplacian image shape:', laplacian_image.shape)

    # range [0,255]

    for i in range(0, final_image.shape[0]):
        for j in range(0, final_image.shape[1]):

            if final_image[(i, j)] > 255:

                final_image[(i, j)] = 255

            elif final_image[(i, j)] < 0:
                final_image[(i, j)] = 0

    final_image = normalizeImage(final_image)

    return final_image, filtering_parameters_list


# PREWITT FILTER ====================================================================================

def prewittFilter(img):

    k = 3
    c = 1

    horizontal_prewitt_kernel = np.ones((k, k))

    for x in range(k):
        horizontal_prewitt_kernel[(x, 0)] = 1

    for y in range(k):
        horizontal_prewitt_kernel[(y, 1)] = 0

    vertical_prewitt_kernel = np.ones((k, k))

    for i in range(k):
        vertical_prewitt_kernel[(0, i)] = 1
    for j in range(k):
        vertical_prewitt_kernel[(1, j)] = 0

    # Padding

    line, column = img.shape

    holdpdd = np.zeros(((line + 2 * c), (column + 2 * c)))

    new_line, new_column = (holdpdd.shape)

    # Based on the Professor Navar's lecture.
    holdpdd[c: new_line - c, c: new_column - c] = img

    #  Convolution

    # Based on the Professor Navar's lecture.
    holdpdd_copy_horizontal = holdpdd.copy()

    holdpdd_copy_vertical = holdpdd.copy()

    # Horizontal Mask

    for x in range(c, holdpdd_copy_horizontal.shape[0]-c):
        for y in range(c, holdpdd_copy_horizontal.shape[1]-c):

            lol = holdpdd[x - c:x + c + 1, y - c: y + c + 1]

            horizontal_mask = (lol*horizontal_prewitt_kernel).sum()

            holdpdd_copy_horizontal[x, y] = horizontal_mask

    print('h mask:', holdpdd_copy_horizontal.min(),
          holdpdd_copy_horizontal.max())

    # Vertical Mask

    for x in range(c, holdpdd_copy_vertical.shape[0]-c):
        for y in range(c, holdpdd_copy_horizontal.shape[1]-c):

            lol = holdpdd[x - c:x + c + 1, y - c: y + c + 1]

            vertical_mask = (lol*vertical_prewitt_kernel).sum()

            holdpdd_copy_vertical[x, y] = vertical_mask

    print('v mask:', holdpdd_copy_vertical.min(), holdpdd_copy_vertical.max())

    # Normalized images

    prewitt_h_normalized = normalizeImage(holdpdd_copy_horizontal)
    prewitt_v_normalized = normalizeImage(holdpdd_copy_vertical)

    # Prewitt image adjustment

    prewitt_image = np.sqrt(
        (holdpdd_copy_horizontal**2 + holdpdd_copy_vertical**2))

    prewitt_adjusted = normalizeImage(prewitt_image.copy())

    # To remove the padding

    final_image = np.zeros((line, column))

    final_image = prewitt_adjusted[c: new_line - c, c: new_column - c]

    print(prewitt_image.shape, final_image.shape)

    return prewitt_h_normalized, prewitt_v_normalized, final_image


# SOBEL FILTER ======================================================================================

def sobelFilter(img):
    k = 3

    c = 1  # Just do simplify; k is the kernel dimension (k x k)

    horizontal_sobel_kernel = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))

    vertical_sobel_kernel = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))

    # Padding

    line, column = (img.shape)

    holdpdd = np.zeros(((line + 2 * c), (column + 2 * c)))

    new_line, new_column = (holdpdd.shape)

    # Based on the Professor Navar's lecture.
    holdpdd[c: new_line - c, c: new_column - c] = img

    # Convolution

    # Based on the Professor Navar's code.
    holdpdd_copy_horizontal = holdpdd.copy()

    # Based on the Professor Navar's code.
    holdpdd_copy_vertical = holdpdd.copy()

    # Horizontal Mask

    for x in range(c, holdpdd_copy_horizontal.shape[0]-c):
        for y in range(c, holdpdd_copy_horizontal.shape[1]-c):

            lol = holdpdd[x - c:x + c + 1, y - c: y + c + 1]

            horizontal_mask = (lol*horizontal_sobel_kernel).sum()

            holdpdd_copy_horizontal[x, y] = round(horizontal_mask)

    # Vertical Mask

    for x in range(c, holdpdd_copy_horizontal.shape[0]-c):
        for y in range(c, holdpdd_copy_horizontal.shape[1]-c):

            lol = holdpdd[x - c: x + c + 1, y - c: y + c + 1]

            vertical_mask = (lol*vertical_sobel_kernel).sum()

            holdpdd_copy_vertical[x, y] = round(vertical_mask)

    # Sobel image and intensity adjustment

    sobel_image = np.sqrt((holdpdd_copy_horizontal**2 +
                          holdpdd_copy_vertical**2))  # sobel image calc

    sobel_adjusted = sobel_image.copy()

    sobel_h_normalized = normalizeImage(holdpdd_copy_horizontal)

    sobel_v_normalized = normalizeImage(holdpdd_copy_vertical)

    final_image = normalizeImage(sobel_image)

    return sobel_h_normalized, sobel_v_normalized, final_image

# Histogram functions

def hist_equa(img, img_context):
    h,w= (img.shape[:2])
    print('height', h)
    print('weight', w)

    h = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            h[img[i, j]] += 1

    eixo_x = list(range(256))
    histograma = (plt.figure(2))
    
    histograma.savefig('histogram/results_histogram/{}_histogram.png'.format(img_context), dpi = 150)
    hist_save = 'histogram/results_histogram/{}_histogram.png'.format(img_context)
    
    return hist_save