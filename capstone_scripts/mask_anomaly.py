import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from matplotlib import cm

sys.path.append('..')

import models_mae

#   Globals assosciated with the downloaded model
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

#   Prepares the model in this folder (should be a file ending in .pth)
def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    # print(msg)
    return model

#   Makes an inference on one image and shows the 4 different images (original, orignal with mask, reconstruction, visible parts of original + reconstruction)
def run_one_image(img, labels, model, og_size):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    # loss, y, mask = model(x.float(), mask_ratio=0.1)
    loss, y, mask = model.custom_forward(x.float(), mask_ratio=0.1, labels=labels)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    np_img = torch.Tensor.numpy(torch.clip((im_paste[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    save_img = Image.fromarray(np_img.astype('uint8'), 'RGB')
    return save_img

#   This prepares our mdoel
# This is an MAE model trained with an extra GAN loss for more realistic generation (ViT-Large, training mask ratio=0.75)
# wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth

chkpt_dir = 'mae_visualize_vit_large_ganloss.pth'
model_mae_gan = prepare_model('mae_visualize_vit_large_ganloss.pth', 'mae_vit_large_patch16')
print('Model loaded.')

#   This function performs random masking on an image given its path
#   (does not always covers anomalies)
def path_to_img(path, seed=None):
    if(seed):
        torch.manual_seed(3)
    print('MAE with extra GAN loss:')

    # load an image
    myImg = Image.open(path)
    myImg = myImg.resize((224, 224))
    myImg = np.array(myImg) / 255.

    # assert myImg.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    myImg = myImg - imagenet_mean
    myImg = myImg / imagenet_std

    run_one_image(myImg, model_mae_gan)

#   This function performs random masking and also gaurantees an images anomalies will be masked
#   given the path to an image and a path to its label folder
def path_label_to_img(img_path, label_path, seed=None):
    if(seed):
        torch.manual_seed(3)

    # load an image
    myImg = Image.open(img_path)
    og_size = myImg.size
    myImg = myImg.resize((224, 224))
    myImg = np.array(myImg) / 255.

    assert myImg.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    myImg = myImg - imagenet_mean
    myImg = myImg / imagenet_std

    #   Open and manipulate the label
    f = open(label_path, 'r')
    lines = f.readlines()
    
    labels = []

    for line in lines:
        label = np.asarray(line.strip('\n').split(' ')[1:5]).astype(np.float32)
        labels.append(label)
    labels = np.asarray(labels)

    return run_one_image(myImg, labels, model_mae_gan, og_size)

# path_label_to_img('../myImages/newAnomaly.jpg', '../myImages/newAnomaly.txt')
# plt.show()

def main():
    #   Check inputs

    if(len(sys.argv) != 2):
        print("Please run this program as follows\n$ python ./mask_anomaly.py {INPUT_DIR_PATH}")
        exit()

    INPUT_DIR = sys.argv[1]

    if not os.path.isdir(INPUT_DIR + '/images'):
        print("Input directory has no images subdirectory")
    if not os.path.isdir(INPUT_DIR + '/labels'):
        print("Input directory has no images subdirectory")

    #   Get name of all files

    images_names = os.listdir(INPUT_DIR + '/images')

    #   Create result directory if it doesn't already exist

    if not os.path.isdir('./results'):
        print("Created new directory with results named './results'")
        os.mkdir('./results')

    if not os.path.isdir('./results/images'):
        print("Created new directory with results named './results/images'")
        os.mkdir('./results/images')

    if not os.path.isdir('./results/labels'):
        print("Created new directory with results named './results/labels'")
        os.mkdir('./results/labels')

    for image in images_names:
        image_path = INPUT_DIR + '/images/' + image
        label_path = INPUT_DIR + '/labels/' + image[:len(image) - 4] + '.txt'

        masked_img = path_label_to_img(image_path, label_path)

        masked_img.save('./results/images/masked_' + image)
        f = open("./results/labels/masked_" + image[:len(image) - 4] + '.txt', 'w')
        f.write("\n")
        f.close()

main()