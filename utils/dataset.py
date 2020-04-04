from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import cv2


# class notes:
# modifications (Aadarsh):
'''
Class now takes a boolean flag upon init to specify if
certain image pre-processing steps need to be taken
false -> no processing,
true -> process and augment data.
Same augementations apply to both images and masks; to both train and val.
'''



class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, flag, scale=1):

        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.flag = flag

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, flag):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))
        if len(pil_img.getbands()) == 4: # RGBA image
            pil_img = pil_img.convert('RGB')

        img_nd = np.array(pil_img)

        if flag is True:
            '''
            1. Flipping the Mask (Flipped Vertically and Horizontally)
            2. Rotate the Image 90
            3. Gaussian Noise
            '''
            img_nd = np.flip(img_nd)
            img_nd = np.rot90(img_nd)

            gaus_noise = np.random.normal(0, 1, img_nd.shape)
            noise_img = img_nd + gaus_noise
            img_ind = noise_img

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        #yuankai add
        mask = np.array(mask)
        mask[mask>0] = 1
        mask = Image.fromarray(mask)


        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, self.flag)
        mask = self.preprocess(mask, self.scale, self.flag)

        #yuankai add
        mask = mask[0,:]

        if self.flag is True:
            return {'image': torch.from_numpy(np.ascontiguousarray(img, dtype=np.float64)), 'mask': torch.from_numpy(np.ascontiguousarray(mask, dtype=np.uint8))}

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
