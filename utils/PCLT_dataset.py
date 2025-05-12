import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
from utils.image_augmentation import *
def prepare_PETCT_dataset(args,transforms=None):
    print("")
    print("img_dir: ", args.img_dir)

    with open(os.path.join(args.split_train_val_test, 'train.txt'), 'r') as f:
        train_list = [x[:-1] for x in f]
    with open(os.path.join(args.split_train_val_test, 'test.txt'), 'r') as f:
        test_list = [x[:-1] for x in f]
    print("train_num:", len(train_list))
    print("test_num:", len(test_list))

    train_dataset = PET_CT_Dataset(train_list, args.img_dir, transforms)
    test_dataset = PET_CT_Dataset(test_list, args.img_dir, transforms=False)

    return train_dataset, test_dataset
class PET_CT_Dataset(Dataset):
    def __init__(self, image_list, img_root, transforms=None):
        super(PET_CT_Dataset, self).__init__()
        self.image_list = image_list

        self.img_root = img_root

        self.pet_suffix = "_PET.png"
        self.mask_suffix = "_mask.png"
        self.ct_suffix = "_CT.png"

        self.transforms =transforms

    def _read_data(self, image_id):
        pet_path = os.path.join(self.img_root, "{0}/{1}{2}").format(image_id.split("_")[0], image_id, self.pet_suffix)
        mask_path = os.path.join(self.img_root, "{0}/{1}{2}").format(image_id.split("_")[0], image_id, self.mask_suffix)
        ct_path = os.path.join(self.img_root, "{0}/{1}{2}").format(image_id.split("_")[0], image_id, self.ct_suffix)

        pet = cv2.imread(pet_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        ct = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)

        assert (pet is not None), pet_path
        assert (mask is not None), mask_path
        assert (ct is not None), ct_path

        return pet, mask, ct

    def _concat_images(self, image1, image2):
        if image1 is not None and image2 is not None:
            img = np.concatenate([image1, image2], 2)
        elif image1 is None and image2 is not None:
            img = image2
        elif image1 is not None and image2 is None:
            img = image1
        else:
            print("[ERROR] Both images are empty.")
            exit(1)
        return img

    def _data_augmentation(self, pet, mask, ct):
        if pet.ndim == 2:
            pet = np.expand_dims(pet, axis=2)
        if ct.ndim == 2:
            ct = np.expand_dims(ct, axis=2)

        if self.transforms:
            img = self._concat_images(pet, ct)
            img, mask = randomShiftScaleRotate(img, mask)
            img, mask = randomHorizontalFlip(img, mask)
            img, mask = randomcrop(img, mask)
        else:
            img = self._concat_images(pet, ct)

        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=2)

        try:
            img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
            mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
        except Exception as e:
            print(e)
            print(img.shape, mask.shape)

        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0
        return img, mask

    def __getitem__(self, index):

        image_id = self.image_list[index]
        pet, mask, ct = self._read_data(image_id)
        pet, mask = self._data_augmentation(pet, mask, ct)
        pet, mask = torch.Tensor(pet), torch.Tensor(mask)
        return pet, mask
    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

