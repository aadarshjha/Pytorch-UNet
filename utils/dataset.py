from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    # modifications @here for dataset.
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'

        pil_img = pil_img.resize((newW, newH))

        '''
        # Aadarsh Preprocess steps here.

        # Using Pillow Library to do preprocessing.
        # horizontal flip
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)

        #vertical flip.
        pil_img = pil_img.transpose(Image.FLIP_TOP_BOTTOM)

        #np image array.


        gaus_noise = np.random.normal(mean, std, image.shape)
        image = image.astype("int16")
        noise_img = image + gaus_noise
        image = ceil_floor_image(image)
        return noise_img
        '''

        img_nd = np.array(pil_img)


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

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        #yuankai add
        mask = mask[0,:]

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
