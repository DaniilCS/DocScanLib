import PIL.Image as Image
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, GaussianBlur, RandomApply, ColorJitter, \
    Grayscale

from model import AutoEncoder

PATH = 'state.pth'


def cut_image_to_pieces(image):
    image_pieces = []
    for i in range(8):
        for j in range(8):
            image_pieces.append(image[i * 160: (i + 1) * 160, j * 115: (j + 1) * 115])
    return image_pieces


def merge_pieces_to_image(img_list):
    width, height = 920, 1280
    big_img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            x_start, y_start = i * 160, j * 115
            x_end, y_end = (i + 1) * 160, (j + 1) * 115
            big_img[x_start:x_end, y_start:y_end, :] = img_list[i * 8 + j]

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    big_img = cv2.filter2D(big_img, -1, kernel)

    return big_img


def denoise_image(image_to_denoise):
    dim = (115, 160)
    cleaned_images = []
    images = cut_image_to_pieces(image_to_denoise)

    transform = transforms.Compose([Resize(size=(255, 255)), Grayscale(), ToTensor()])
    model_for_noise = AutoEncoder()
    model_for_noise.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model_for_noise = model_for_noise.eval()
    max_pix_val = 0
    min_pix_val = 255

    pieces = []

    for k, image in enumerate(images):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        x = transform(im_pil)
        x = x.unsqueeze(0)
        y = model_for_noise(x)
        output_array = y.detach().cpu().numpy()
        output_array = output_array.squeeze()
        pieces.append(output_array)
        if output_array.min() < min_pix_val:
            min_pix_val = output_array.min()
        if output_array.max() > max_pix_val:
            max_pix_val = output_array.max()

    for output_array in pieces:
        output_array = (output_array - min_pix_val) / (max_pix_val - min_pix_val) * 255.0
        output_array = output_array.astype(np.uint8)
        output_image = cv2.cvtColor(output_array, cv2.COLOR_GRAY2BGR)
        output_image = cv2.resize(output_image, dim, interpolation=cv2.INTER_AREA)
        cleaned_images.append(output_image)

    return merge_pieces_to_image(cleaned_images)
