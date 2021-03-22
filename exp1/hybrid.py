import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    img_array = np.array(img)  # 创建像素值数组
    img_row = img_array.shape[0]  # 获取图像的行
    img_col = img_array.shape[1]  # 获取图像的列


    kernel_row = kernel.shape[0]  # 获取核的行
    kernel_col = kernel.shape[1]  # 获取核的列

    Hedge = np.zeros((img_row, kernel_col//2))  # 计算水平边缘
    Vedge = np.zeros((kernel_row//2, img_col+Hedge.shape[1]*2))  # 计算垂直边缘

# 判断图像维度
    if img_array.ndim == 3:
        img_vim = img_array.shape[2]  # 获取图像维度
        conv = np.zeros((img_row, img_col, img_vim))
        for i in range(3):
            temp_array = np.hstack([Hedge, np.hstack([img_array[:, :, i], Hedge])])  # 为图像填充水平边缘
            new_array = np.vstack([Vedge, np.vstack([temp_array, Vedge])])  # 为图像填充垂直边缘
            #计算相关
            for j in range(img_row):
                for k in range(img_col):
                    conv[j][k][i] = (new_array[j:j+kernel_row, k:k+kernel_col]*kernel).sum()
        return conv
    else:
        conv = np.zeros((img_row, img_col))
        temp_array = np.hstack([Hedge, np.hstack([img_array, Hedge])])  # 为图像填充水平边缘
        new_array = np.vstack([Vedge, np.vstack([temp_array, Vedge])])  # 为图像填充垂直边缘
        for j in range(img_row):
            for k in range(img_col):
                conv[j][k] = (new_array[j:j + kernel_row - 1, k:k + kernel_col - 1]*kernel).sum()
        return conv
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    kernel1 = np.flipud(np.fliplr(kernel))  # 图像左右上下翻转
    return cross_correlation_2d(img, kernel1)
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # 初始化高斯核
    gaussian_kernel = np.zeros((height, width), dtype='double')
    # 计算中心位置
    centerRow = height//2
    centerCol = width//2
    sum = 0
    for i in range(height):
        for j in range(width):
            x = i - centerRow
            y = j - centerCol
            gaussian_kernel[i][j] = (1.0/(2*np.pi*(sigma**2)))*np.exp(-float(x**2+y**2)/(2*(sigma**2)))
            sum = sum+gaussian_kernel[i][j]
    return gaussian_kernel/sum  # 返回归一化后的高斯核
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    height = width = size
    kernel = gaussian_blur_kernel_2d(sigma, height, width)  # 设置高斯核
    return convolve_2d(img, kernel)  # 进行卷积
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    img_array = np.array(img)
    height = width = size
    return (img_array - low_pass(img, sigma, size))
    # TODO-BLOCK-BEGIN
    raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
                        high_low2, mixin_ratio):
    '''
    This function adds two images to create a hybrid image, based on
    parameters specified by the user.
    '''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)


    else:
        img1 = high_pass(img1, sigma1, size1)



    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)


    else:
        img2 = high_pass(img2, sigma2, size2)


    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
   #return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
    min_h=np.amin(hybrid_img);
    max_h=np.amax(hybrid_img);
    return ((hybrid_img-min_h)/(max_h-min_h) * 255).clip(0, 255).astype(np.uint8)



img1 = cv2.imread("D:/2017/Project1/cat.jpg")
img2 = cv2.imread("D:/2017/Project1/dog.jpg")
Nmimg1 = np.array(img1)
Nmimg2 = np.array(img2)
ratio = 0.65
Image = create_hybrid_image(Nmimg1, Nmimg2, 10, 30, "high", 12, 50, "low", ratio)
cv2.imshow("hybrid", Image)
#cv2.imwrite("hybrid2.jpg", Image)
cv2.waitKey(0)