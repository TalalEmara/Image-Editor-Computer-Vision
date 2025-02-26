import numpy as np
from Core.gray import rgb_to_grayscale
# from gray import rgb_to_grayscale

def add_LowPass_filter(image, cutOff_Freq):

    image = rgb_to_grayscale(image)

    # Compute the Fourier Transform and shift the DC component to the center
    fourier = np.fft.fftshift(np.fft.fft2(image))
    
    rows, columns = fourier.shape
    low_pass_filter = np.zeros_like(fourier)

    # Create a circular mask for low-pass filtering
    for row in range(rows):
        for column in range(columns):
            frequency = np.sqrt((row - rows / 2) ** 2 + (column - columns / 2) ** 2)
            if frequency <= cutOff_Freq:
                low_pass_filter[row, column] = 1

    filtered_fourier = fourier * low_pass_filter

    lowPass_filter_Image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fourier))).astype(np.uint8)

    return lowPass_filter_Image


def add_HighPass_filter(image, cutOff_Freq):
    image = rgb_to_grayscale(image)

    fourier = np.fft.fftshift(np.fft.fft2(image))
    
    rows, columns = fourier.shape
    high_pass_filter = np.ones_like(fourier)  

    # Create a circular mask for high-pass filtering 
    for row in range(rows):
        for column in range(columns):
            frequency = np.sqrt((row - rows / 2) ** 2 + (column - columns / 2) ** 2)
            if frequency <= cutOff_Freq:
                high_pass_filter[row, column] = 0  # Block low frequencies


    filtered_fourier = fourier * high_pass_filter

    highPass_filter_Image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_fourier))).astype(np.uint8)

    return highPass_filter_Image


#TEST
# import cv2
# image = cv2.imread("images/colored2.jpg")


# low_pass_result = add_LowPass_filter(image, cutOff_Freq=70)
# high_pass_result = add_HighPass_filter(image, cutOff_Freq=1)

# cv2.imshow("Low-Pass Filter", low_pass_result)
# cv2.imshow("High-Pass Filter", high_pass_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()










# # using kernal size

# import numpy as np

# def add_LowPass_filter(image, kernel_size):
#     image = rgb_to_grayscale(image)
#     kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    
#     h, w = image.shape
#     filtered_image = np.zeros((h, w), dtype=np.float32)
#     padded_image = np.pad(image, kernel_size // 2, mode='constant', constant_values=0)
    
#     for i in range(h):
#         for j in range(w):
#             region = padded_image[i:i + kernel_size, j:j + kernel_size]
#             filtered_image[i, j] = np.sum(region * kernel)
    
#     return np.clip(filtered_image, 0, 255).astype(np.uint8)

# def add_HighPass_filter(image, kernel_size):
#     image = rgb_to_grayscale(image)
    
#     kernel = -1 * np.ones((kernel_size, kernel_size), dtype=np.float32)
#     kernel[kernel_size // 2, kernel_size // 2] = (kernel_size ** 2) - 1
    
#     h, w = image.shape
#     filtered_image = np.zeros((h, w), dtype=np.float32)
#     padded_image = np.pad(image, kernel_size // 2, mode='constant', constant_values=0)
    
#     for i in range(h):
#         for j in range(w):
#             region = padded_image[i:i + kernel_size, j:j + kernel_size]
#             filtered_image[i, j] = np.sum(region * kernel)
    
#     return np.clip(filtered_image, 0, 255).astype(np.uint8)

# # TEST
# import cv2
# image = cv2.imread("images/colored2.jpg")

# low_pass_result = add_LowPass_filter(image, kernel_size=5)
# high_pass_result = add_HighPass_filter(image, kernel_size=2)

# cv2.imwrite("low_pass_result.jpg", low_pass_result)
# cv2.imwrite("high_pass_result.jpg", high_pass_result)

# cv2.imshow("Low-Pass Filter", low_pass_result)
# cv2.imshow("High-Pass Filter", high_pass_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
