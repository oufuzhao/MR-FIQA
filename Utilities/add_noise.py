import numpy as np
from scipy.ndimage import filters, measurements, interpolation
from PIL import Image

def add_noise_img(image, low_scale, blur_level, out_size):
    lr_img = blur_downscale(image, low_scale, blur_level, output_shape=None)
    resized_image = lr_img.resize(out_size)
    return resized_image

def blur_downscale(im, low_scale, blur_level, output_shape=None):
    if blur_level>1:
        im = np.array(im).astype(np.uint8)
        scale_factor = np.array([low_scale, low_scale])  # choose scale-factor
        avg_sf = np.mean(scale_factor)  # this is calculated so that min_var and max_var will be more intutitive
        min_var = avg_sf * 0.175 # default = 0.175   # variance of the gaussian kernel will be sampled between min_var and max_var
        max_var = avg_sf * blur_level  #avg_sf * (low_scale/1.6) # default = 2.5
        k_size = np.array([21, 21])  # size of the kernel, should have room for the gaussian
        noise_level = 0.1 * blur_level  #default = 0.4  # this option allows deviation from just a gaussian, by adding multiplicative noise noise
        # Set random eigen-vals (lambdas) and angle (theta) for COV matrix
        lambda_1 = min_var + np.random.rand() * (max_var - min_var);
        lambda_2 = min_var + np.random.rand() * (max_var - min_var);
        theta = np.random.rand() * np.pi
        noise = -noise_level + np.random.rand(*k_size) * noise_level * 2
        # Set COV matrix using Lambdas and Theta
        LAMBDA = np.diag([lambda_1, lambda_2]);
        Q = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
        SIGMA = Q @ LAMBDA @ Q.T
        INV_SIGMA = np.linalg.inv(SIGMA)[None, None, :, :]
        # Set expectation position (shifting kernel for aligned image)
        MU = k_size // 2  + 0.5 * (scale_factor - k_size % 2)
        MU = MU[None, None, :, None]
        # Create meshgrid for Gaussian
        [X,Y] = np.meshgrid(range(k_size[0]), range(k_size[1]))
        Z = np.stack([X, Y], 2)[:, :, :, None]
        # Calcualte Gaussian for every pixel of the kernel
        ZZ = Z-MU
        ZZ_t = ZZ.transpose(0,1,3,2)
        raw_kernel = np.exp(-0.5 * np.squeeze(ZZ_t @ INV_SIGMA @ ZZ)) * (1 + noise)
        # shift the kernel so it will be centered
        shift_vec = np.array(raw_kernel.shape) / 2 + 0.5 * (scale_factor - (raw_kernel.shape[0] % 2)) - measurements.center_of_mass(raw_kernel)
        # Finally shift the kernel and return
        raw_kernel_centered = interpolation.shift(raw_kernel, shift_vec)
        # raw_kernel_centered = kernel_shift(raw_kernel, scale_factor)
        # Normalize the kernel and return
        kernel = raw_kernel_centered / np.sum(raw_kernel_centered)
        # output shape can either be specified or, for simple cases, can be calculated.
        # see more details regarding this at: https://github.com/assafshocher/Resizer
        scale_factor = np.array([low_scale, low_scale])  # choose scale-factor
        if output_shape is None:
            output_shape = np.array(im.shape[:-1]) / np.array(scale_factor)
        # First run a correlation (convolution with flipped kernel)
        out_im = np.zeros_like(im)
        for channel in range(np.ndim(im)):
            out_im[:, :, channel] = filters.correlate(im[:, :, channel], kernel)
        image = Image.fromarray(out_im)
    else:
        image = im
    low_scale += 1
    lr_img = image.resize((int(image.size[0]/low_scale), int(image.size[1]/low_scale)))
    return lr_img