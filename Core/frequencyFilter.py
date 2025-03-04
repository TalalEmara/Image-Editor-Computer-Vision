import numpy as np
from Core.gray import rgb_to_grayscale
# from gray import rgb_to_grayscale

def add_LowPass_filter(image, cutOff_Freq):

    image = rgb_to_grayscale(image)

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

# def generate_hybrid_image(image1, image2, weight=0.5, low_cutoff=70, high_cutoff=1):
#     """
#     Generate a hybrid image using custom low-pass and high-pass filtering functions.
#
#     Args:
#     - image1: Image for low frequencies (background).
#     - image2: Image for high frequencies (details).
#     - weight: Blending weight (0-1) between low and high frequencies.
#     - low_cutoff: Cutoff frequency for low-pass filtering.
#     - high_cutoff: Cutoff frequency for high-pass filtering.
#
#     Returns:
#     - Hybrid image combining low and high frequencies.
#     """
#     # Resize both images to the smallest common size
#     min_shape = (min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1]))
#     image1_resized = cv2.resize(image1, (min_shape[1], min_shape[0]))
#     image2_resized = cv2.resize(image2, (min_shape[1], min_shape[0]))
#
#     # Apply custom filters
#     low_frequencies = add_LowPass_filter(image1_resized, low_cutoff).astype(np.float32)
#     high_frequencies = add_HighPass_filter(image2_resized, high_cutoff).astype(np.float32)
#
#     # Normalize both images to range 0-1 for better blending
#     low_frequencies = (low_frequencies - np.min(low_frequencies)) / (np.max(low_frequencies) - np.min(low_frequencies) + 1e-8)
#     high_frequencies = (high_frequencies - np.min(high_frequencies)) / (np.max(high_frequencies) - np.min(high_frequencies) + 1e-8)
#
#     # Blend images using weight
#     hybrid_image = ((1 - weight) * low_frequencies + weight * high_frequencies) * 255
#
#     # Convert to 8-bit image
#     hybrid_image = np.clip(hybrid_image, 0, 255).astype(np.uint8)
#
#     return hybrid_image
#
# import cv2
# import numpy as np
#
# def gaussian_pyramid(image, levels=5):
#     """ Generate a Gaussian pyramid for an image. """
#     pyramid = [image]
#     for _ in range(1, levels):
#         image = cv2.pyrDown(image)  # Downsample while applying Gaussian blur
#         pyramid.append(image)
#     return pyramid
#
# def laplacian_pyramid(image, levels=5):
#     """ Generate a Laplacian pyramid from a Gaussian pyramid. """
#     gauss_pyr = gaussian_pyramid(image, levels)
#     laplacian_pyr = []
#
#     for i in range(levels - 1):
#         upsampled = cv2.pyrUp(gauss_pyr[i + 1], dstsize=gauss_pyr[i].shape[:2][::-1])
#         laplacian = cv2.subtract(gauss_pyr[i], upsampled)  # Laplacian = Difference
#         laplacian_pyr.append(laplacian)
#
#     laplacian_pyr.append(gauss_pyr[-1])  # Append the last Gaussian level
#     return laplacian_pyr
#
# def blend_pyramids(lap_pyr1, lap_pyr2, N1, N2):
#     """ Blend two Laplacian pyramids using first N1 levels from image1
#         and last N2 levels from image2. """
#     blended_pyr = []
#     levels = len(lap_pyr1)
#
#     for i in range(levels):
#         if i < N1:
#             blended_pyr.append(lap_pyr1[i])  # Take from image1
#         elif i >= levels - N2:
#             blended_pyr.append(lap_pyr2[i])  # Take from image2
#         else:
#             alpha = (i - N1) / (levels - N1 - N2 + 1)
#             blended_pyr.append((1 - alpha) * lap_pyr1[i] + alpha * lap_pyr2[i])
#
#     return blended_pyr
#
# def reconstruct_from_pyramid(lap_pyr):
#     """ Reconstruct an image from a Laplacian pyramid. """
#     image = lap_pyr[-1]  # Start with the coarsest level
#
#     for i in range(len(lap_pyr) - 2, -1, -1):
#         image = cv2.pyrUp(image, dstsize=lap_pyr[i].shape[:2][::-1])  # Upsample
#         image = cv2.add(image, lap_pyr[i])  # Add back the details
#
#     return np.clip(image, 0, 255).astype(np.uint8)
#
# def generate_hybrid_imageK(image1, image2, N1=3, N2=5):
#     """ Generate a hybrid image using Laplacian pyramids. """
#     # Convert to grayscale if needed
#     if len(image1.shape) == 3:
#         image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     if len(image2.shape) == 3:
#         image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
#
#     # Resize both images to the same smallest dimensions
#     min_shape = (min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1]))
#     image1 = cv2.resize(image1, (min_shape[1], min_shape[0]))
#     image2 = cv2.resize(image2, (min_shape[1], min_shape[0]))
#
#     # Generate pyramids
#     lap_pyr1 = laplacian_pyramid(image1)
#     lap_pyr2 = laplacian_pyramid(image2)
#
#     # Blend the Laplacian pyramids
#     blended_pyr = blend_pyramids(lap_pyr1, lap_pyr2, N1, N2)
#
#     # Reconstruct final hybrid image
#     hybrid_image = reconstruct_from_pyramid(blended_pyr)
#
#     return hybrid_image


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
