from torch.utils.data import Dataset
import albumentations
from PIL import Image
import numpy as np
import re
import os
import scipy.io as scio
from scipy.signal import hilbert
import torchvision.transforms as transforms
import torch

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        # if not image.mode == "RGB":
        #     image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        # image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example
    


class RFDataPaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

    def __len__(self):
        return self._length

    def preprocess_image(self, rf_path):
        path = re.sub("us_image", "rf_data", rf_path)
        path = re.sub(".png", "", path)
        files = os.listdir(path)
        files.sort(key=lambda x: int(x[5:-4]))

        D = 10  # Sampling frequency decimation factor
        fs = 100e6 / D  # Sampling frequency  [Hz]
        min_sample = 0

        tstarts = []
        rf_envs = []
        max_len = 0
        data_len = 1024
        for file in files:
            mat = scio.loadmat(os.path.join(path, file))
            tstart = mat['tstart']
            rf_data = mat['rf_data']
            # rf_data = np.resize(rf_data, (1, len(rf_data)))[0]

            # size_x = int(np.round(tstart * fs - min_sample))
            # if size_x > 0:
            #     rf_data = np.concatenate((np.zeros((size_x)), rf_data))
            
            # rf_data = rf_data[:, np.newaxis]

            rf_env = np.abs(hilbert(rf_data, axis=0))

            D = int(np.floor(rf_env.shape[0] / data_len))
            rf_env = rf_env[slice(0, data_len * D, D)]

            tstarts.append(tstart)
            rf_envs.append(rf_env)


        env = np.concatenate(rf_envs, axis=1)


        dB_Range=50
        env = env - np.min(env)
        env = env / np.max(env)
        env = env + 0.00001
        log_env = 20 * np.log10(env)
        log_env=255/dB_Range*(log_env+dB_Range)
        [N, M] = log_env.shape
        D = int(np.floor(N/1024))
        # env_disp = 255 * log_env[1:N:D, :] / np.max(log_env)
        env_disp = 255 * log_env / np.max(log_env)
        env_disp = env_disp.astype(np.uint8)
        # img = Image.fromarray(env_disp)
        # img = img.resize((self.size, self.size))
        # img = img.rotate(90)
        # if not img.mode == "RGB":
        #     img = img.convert("RGB")
        # img = np.array(img)
        img = env_disp
        img = np.rot90(img, 1)

        img = (img/127.5 - 1.0).astype(np.float32)

        return img

    def __getitem__(self, i):
        example = dict()
        example["coord"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example
    

class RFDataUsImageTrain(Dataset):
    def __init__(self, size=None, training_images_list_file=None, us_transforms=None, rf_transforms=None):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.rf = RFDataPaths(paths=paths, size=size, random_crop=False)

        if us_transforms:
            self.us_transforms = transforms.Compose(us_transforms)
        if rf_transforms:
            self.rf_transforms = transforms.Compose(rf_transforms)

        self._length = len(paths)

    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        example = {}
        example["us"] = self.us_transforms(self.data[index]["image"])
        example["rf"] = self.rf_transforms(self.rf[index]["coord"])

        return example
    


class RFDataUsImageValidation(Dataset):
    def __init__(self, size=None, test_images_list_file=None, us_transforms=None, rf_transforms=None):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)
        self.rf = RFDataPaths(paths=paths, size=size, random_crop=False)

        if us_transforms:
            self.us_transforms = transforms.Compose(us_transforms)
        if rf_transforms:
            self.rf_transforms = transforms.Compose(rf_transforms)

        self._length = len(paths)

    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        example = {}
        example["us"] = self.us_transforms(self.data[index]["image"])
        example["rf"] = self.rf_transforms(self.rf[index]["coord"])

        return example
    
    

class RFDataTrain(Dataset):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        with open(training_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = RFDataPaths(paths=paths, size=size, random_crop=False)

        self._length = len(paths)
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        return self.data[index]

    


class RFDataValidation(Dataset):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        with open(test_images_list_file, "r") as f:
            paths = f.read().splitlines()
        self.data = RFDataPaths(paths=paths, size=size, random_crop=False)

        self._length = len(paths)
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        return self.data[index]
    


# ================================TEST============================================


def GetRFData(dataPath,opt):

    D = 10  # Sampling frequency decimation factor
    fs = 100e6 / D  # Sampling frequency  [Hz]

    #  Read the data and adjust it in time

    min_sample = 0

    env = []

    env_max_list = []
    env_min_list = []

    for i in range(128):
        dataMatName = f'rf_ln{i + 1}.mat'
        dataFile = os.path.join(dataPath, dataMatName)
        data = scio.loadmat(dataFile)
        rf_data = data["rf_data"]
        rf_data = np.resize(rf_data, (1, len(rf_data)))[0]
        t_start = data["tstart"]

        size_x = int(np.round(t_start * fs - min_sample))

        if size_x > 0:
            rf_data = np.concatenate((np.zeros((size_x)), rf_data))

        rf_env = hilbert(rf_data)

        rf_env = np.abs(rf_env)

        env_max_list.append(max(rf_env))
        env_min_list.append(min(rf_env))

        env.append(rf_env)

    max_env = max(env_max_list)
    min_env = min(env_min_list)
    log_env = []
    # data_len = opt.rfdata_len
    data_len = 1024
    log_env_max_list = []
    log_env_min_list = []

    for i in range(len(env)):

        D = int(np.floor(len(env[i]) / data_len))

        env[i] = env[i] - min_env

        tmp_env = env[i][slice(0, data_len * D, D)]/max_env

        for i in range(len(tmp_env)):
            if tmp_env[i] != 0.0:
                tmp_env[i] = np.log(tmp_env[i])

        log_env_max_list.append(max(tmp_env))
        log_env_min_list.append(min(tmp_env))
        log_env.append(tmp_env)

    log_env_max = max(log_env_max_list)
    log_env_min = min(log_env_min_list)
    # print((log_env_min, log_env_max))
    log_env_max = log_env_max - log_env_min

    disp_env = []

    for i in range(len(log_env)):

        tmp_env = log_env[i]

        tmp_env = ((tmp_env - log_env_min) / log_env_max)

        disp_env.append(tmp_env)


    disp_env = np.asarray(disp_env)

    return disp_env

class MyDataSet(Dataset):
    def __init__(self,root,opt=None,img_transform=None, rf_transform=None,mode="train",use_input="rf_data",use_embedding = False):

        self.rf_dir = os.path.join(root, "rf_data")
        self.us_dir = os.path.join(root, "us_image")
        self.img_transform = None
        self.rf_transform = None

        if img_transform:
            self.img_transform = transforms.Compose(img_transform)
        if rf_transform:
            self.rf_transform = transforms.Compose(rf_transform)

        if mode == "train":
            f = open(os.path.join(root, "us_image_train.txt"), "r", encoding='utf-8')
            us_train_list = f.read().splitlines()
            f.close()
            self.us_list = us_train_list
        else:
            f = open(os.path.join(root, "us_image_val.txt"), "r", encoding='utf-8')
            us_train_list = f.read().splitlines()
            f.close()
            self.us_list = us_train_list

        self.use_input = use_input

        self.opt = opt

        self.use_embedding = use_embedding

        self.mode = mode

    def __len__(self):
        return len(self.us_list)

    def __getitem__(self, idx):

        loss = 0
        us_name = self.us_list[idx].replace("some/us_image/","")

        us_path = os.path.join(self.us_dir, us_name)

        us = Image.open(us_path)
        if self.img_transform:
            us = self.img_transform(us)
        
        rf_name = us_name.replace(".png","")
        rf_path = os.path.join(self.rf_dir, rf_name)

        rf_data = GetRFData(rf_path, self.opt)

        if self.rf_transform:
            rf_data = self.rf_transform(rf_data)

        return {"rf_data": rf_data, "us": us, "name": us_name.replace(".png", "")}
    

if __name__ == '__main__':
    training_images_list_file = 'some/us_image_train.txt'
    test_images_list_file = 'some/us_image_val.txt'

    rf_us = RFDataUsImageTrain(256, training_images_list_file)


    item = rf_us[0]
    image = ((item['image'] + 1)*125).astype(np.uint8)
    rfdata = ((item['coord'] + 1)*125).astype(np.uint8)
    print(image.shape)
    print(rfdata.shape)

    image = Image.fromarray(image)
    image.save('image_ori.png')

    rfdata = Image.fromarray(rfdata)
    rfdata.save('rfdata_ori.png')

    print('================TEST===============')

    rf_transforms = [
            transforms.ToTensor(),
        ]
    img_transforms = [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(0.5,0.5),
        ]

    data = MyDataSet(root='some', img_transform=img_transforms, rf_transform=rf_transforms)

    print('================DATA===============')
    item = data[0]

    image = item['us'].numpy()
    print(np.min(image), np.max(image))
    image = ((image + 1) * 150).astype(np.uint8)
    rfdata = item['rf_data'].numpy()
    print(np.min(rfdata), np.max(rfdata))
    rfdata = (rfdata * 255).astype(np.uint8)
    image = np.squeeze(image, axis=0)
    rfdata = np.squeeze(rfdata, axis=0)
    print(image.shape)
    print(rfdata.shape)

    image = Image.fromarray(image)
    image.save('image.png')

    rfdata = Image.fromarray(rfdata)
    rfdata.save('rfdata.png')

