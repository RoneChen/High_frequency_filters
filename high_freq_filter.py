import torch
import torchvision
import torch

import torchvision.transforms as transforms
import time
import cv2

from torch import nn
import torch
import numpy as np

'''
1. Read a RGB iamge.
2. Transfer the RGB image to YUV, and split three channels.
3. Get high-frequency information from Y channel.
'''

# Load the image
image_rgb = cv2.imread('img.png')


def transfer_split_yuv_channels(img):
    '''
    Transfer a RGB image to YUV, and split each channel as a tensor.

    img: RGB PIL image
    '''
    image_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    image_y = torch.tensor(image_yuv[:,:,0])
    image_u = torch.tensor(image_yuv[:,:,1])
    image_v = torch.tensor(image_yuv[:,:,2])

    # Add a channel dimension to the U and V channels
    image_y = image_y.unsqueeze(0)
    image_u = image_u.unsqueeze(0)
    image_v = image_v.unsqueeze(0)

    return image_y, image_u, image_v

def combine_channels(y, u, v, type):
    '''
    Reconstruct a YUV image from three tensors of channels.

    y: Y channel.
    u: U channel.
    v: V channel.
    '''

    # Add the U and V channels back to the high-frequency image
    reconstruct_image = torch.cat((y, u, v), dim=0)

    # Convert YUV to RGB
    reconstruct_image = reconstruct_image.permute(1, 2, 0).numpy()
    reconstruct_image = cv2.cvtColor(reconstruct_image, cv2.COLOR_YUV2RGB)
    # Convert PIL image back to tensor
    reconstruct_image = transforms.ToTensor()(reconstruct_image)

    # Convert tensor to PIL image
    reconstruct_image = transforms.ToPILImage()(reconstruct_image)

    # Save the PIL image
    if type == 'g':
        reconstruct_image.save('g_combined.png')
    elif type == 'du':
        reconstruct_image.save('du_combined.png')


def gaussion_high_frequency_filter(y_channel):
    '''
    Get Y channel of an image, and split high and low-frequency information.

    y_chnanel: Y channel of an image.
    '''

    # Apply Gaussian blur transform
    total = 0

    for _ in range(30):
        start = time.time()
        gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=1)
        low_freq = gaussian_blur(y_channel)
        # Get the high-frequency information
        high_freq = y_channel - low_freq
        end = time.time()
        total +=  end-start

    print(total/30)

    return high_freq, low_freq


def down_up_high_freq_filter(y_channel):

    y_channel = y_channel.float()/255
    # Downsample the Y channel using average pooling

    downsampled_y = nn.functional.avg_pool2d(y_channel, kernel_size=2)
 
    downsampled_y = downsampled_y.unsqueeze(0)
    print(downsampled_y.shape)

    # # Upsample the downsampled Y channel using bilinear interpolation

    low_freq = nn.functional.interpolate(downsampled_y, size=(1080, 1920), mode='bilinear', align_corners=False)

    # Add a channel dimension to the upsampled Y channel
    # upsampled_y = upsampled_y.unsqueeze(0)
    low_freq = low_freq.squeeze(0)

    # Get the high-frequency information
    high_freq = y_channel - low_freq

    return (high_freq * 255).type(torch.uint8), (low_freq * 255).type(torch.uint8)

def save_as_png(high_freq, low_freq, type):
    '''
    save the high_freq and low_freq tensor to png image.
    '''

    # Convert tensor to PIL image
    high_frequency_img = transforms.ToPILImage()(high_freq)
    low_frequency_img = transforms.ToPILImage()(low_freq)

    if type == 'g':
    # Save the PIL image
        high_frequency_img.save('g_high_freq.png')
        low_frequency_img.save('g_low_freq.png')
    elif type == 'du':
        high_frequency_img.save('du_high_freq.png')
        low_frequency_img.save('du_low_freq.png')

if __name__ == '__main__':

    y_channel, u_channel, v_channel = transfer_split_yuv_channels(image_rgb)

    # Save each channel as a PNG image
    y_channel_img = transforms.ToPILImage()(y_channel)
    u_channel_img = transforms.ToPILImage()(u_channel)
    v_channel_img = transforms.ToPILImage()(v_channel)

    y_channel_img.save('y_channel.png')
    u_channel_img.save('u_channel.png')
    v_channel_img.save('v_channel.png')

    high_feq_g, low_freq_g = gaussion_high_frequency_filter(y_channel=y_channel)
    print(high_feq_g)
    save_as_png(high_freq=high_feq_g, low_freq=low_freq_g, type='g')

    high_feq_du, low_freq_du = down_up_high_freq_filter(y_channel=y_channel)
    print(high_feq_du)
    save_as_png(high_freq=high_feq_du, low_freq=low_freq_du, type='du')
    
    combine_channels(low_freq_g, u_channel, v_channel, 'g')
    combine_channels(low_freq_du, u_channel, v_channel, 'du')
 