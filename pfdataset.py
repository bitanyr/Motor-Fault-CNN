#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Zhenglong Sun
# Data : 2022-3-15 22:43
#
# نکته‌ی مهم: این فایل باید دقیقا با همین اسم (pfdataset.py) کنار
# motor-fault-cnn.py ذخیره شود، چون آن اسکریپت با دستور
#     import pfdataset as pfd
# آن را صدا می‌زند. اگر این فایل روی دیسک با اسم دیگری (مثلا
# generate_dataset.py) ذخیره شده باشد، اجرای motor-fault-cnn.py همان
# ابتدای کار با خطای
#     ModuleNotFoundError: No module named 'pfdataset'
# متوقف می‌شود و اصلا وارد آموزش نمی‌شود.

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch.nn.functional as F


def wgn(x, snr_min, snr_max):
    snr = np.random.randint(snr_min, snr_max)
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


'''
feature is a tuple, define which features are extracted from the original dataset.
if all the three phase voltages, features = (0,1,2)
داده‌ها ی ورودی
0'Voltage_0',1'Voltage_1',2'Voltage_2',3'Current_0',
4'Current_1',5'Current_2',6'Rotor_Current',7'Speed',
'Failde'
'''
class MFataset(Dataset):

    def __init__(self, df_path, pre_disturbance, post_disturbance, features, transform=None):
        # Initialize data, download, etc.
        # read with pandas

        data = np.load(df_path, allow_pickle='TRUE').item()

        data_Test_Data_Rotor_Current_Faild = data[0,0][:,(400-pre_disturbance):(400+post_disturbance),features]
        label_Test_Data_Rotor_Current_Faild = data[0,1]
        data_Disconnect_Phase_10_11_21 = data[1,0][:,(400-pre_disturbance):(400+post_disturbance),features]
        label_Disconnect_Phase_10_11_21 = data[1,1]
        data_Rotor_Current_Failed_R = data[2,0][:,(400-pre_disturbance):(400+post_disturbance),features]
        label_Rotor_Current_Failed_R = data[2,1]
        data_Short_between_two_phases = data[3,0][:,(400-pre_disturbance):(400+post_disturbance),features]
        label_Short_between_two_phases = data[3,1]
        data_Test_Data_Short_phases_Ln_G = data[4,0][:,(400-pre_disturbance):(400+post_disturbance),features]
        label_Test_Data_Short_phases_Ln_G = data[4,1]
        data_No_failed = data[5,0][:,(400-pre_disturbance):(400+post_disturbance),features]
        label_No_failed = data[5,1]

        all_exa = np.concatenate((data_Test_Data_Rotor_Current_Faild,data_Disconnect_Phase_10_11_21,
                                  data_Rotor_Current_Failed_R,data_Short_between_two_phases,
                                  data_Test_Data_Short_phases_Ln_G,data_No_failed),axis=0)

        all_lables = np.concatenate((label_Test_Data_Rotor_Current_Faild,label_Disconnect_Phase_10_11_21,
                               label_Rotor_Current_Failed_R,label_Short_between_two_phases,
                               label_Test_Data_Short_phases_Ln_G,label_No_failed),axis=0)
        self.n_samples = all_exa.shape[0]
        # all_exa, all_labels is np array
        self.x_data = all_exa.astype(np.float64)  # size [n_samples, time_len, n_features]
        self.y_data = all_lables.astype(int)  # size [n_samples, 1]

        self.transform = transform

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):

        sample = self.x_data[index], self.y_data[index].squeeze()

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


# we can also use the transforms in torch vision
class ToTensor:
    # Convert ndarrays to Tensors
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs).float(), torch.from_numpy(targets)


class ToImage:
    # سیگنال [time, feature] را به یک "تصویر" [feature, size, time/size] تبدیل می‌کند
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        inputs, targets = sample
        if self.size == 0:
            # قبلا اینجا هیچ مقداری برای figures ست نمی‌شد و خط return
            # با UnboundLocalError کرش می‌کرد. حالا سیگنال دست‌نخورده برمی‌گردد.
            return inputs, targets

        n_features = inputs.shape[1]
        time_len = inputs.shape[0]
        if time_len % self.size != 0:
            # قبلا اینجا فقط یک خطای مبهم reshape از torch می‌گرفتید.
            # این حالت دقیقا زمانی پیش می‌آید که pre_disturbance+post_disturbance
            # بر size بخش‌پذیر نباشد (مثلا برای طول‌های 400 یا 800 نمونه‌ای که
            # در گزارش امتحان شده‌اند، وقتی size=30 باشد).
            raise ValueError(
                f"طول سیگنال ({time_len}) بر size ({self.size}) بخش‌پذیر نیست. "
                f"pre_disturbance + post_disturbance را طوری انتخاب کنید که "
                f"باقی‌مانده‌اش بر size صفر شود، یا size را عوض کنید."
            )

        inputs1 = inputs.transpose(0, 1)
        figures = inputs1.reshape(n_features, self.size, -1)
        return figures, targets


class MulTransform:
    # multiply inputs with a given factor
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        # قبلا این خط به‌صورت درجا (inputs *= self.factor) بود. چون
        # torch.from_numpy در ToTensor حافظه را با آرایه‌ی numpy اصلی
        # self.x_data به اشتراک می‌گذارد، ضرب درجا داده‌ی خام دیتاست را
        # برای همیشه (در هر epoch دوباره) دستکاری می‌کرد. الان خروجی
        # جدید ساخته می‌شود و دیتای اصلی دست‌نخورده می‌ماند.
        inputs = inputs * self.factor
        return inputs, targets


class LpNormalize:
    # Performs Lp normalization of inputs over specified dimension,默认按行。
    def __init__(self, p=2, dim =1):
        self.p = p
        self.dim = dim

    def __call__(self, sample):
        inputs0, targets = sample
        inputs = F.normalize(inputs0, self.p, self.dim )
        return inputs, targets


class StdNormalize:
    # Performs z-score normalization of inputs over specified dimension
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, sample):
        inputs0, targets = sample
        mean = torch.mean(inputs0, dim=self.dim)
        std = torch.std(inputs0, dim=self.dim)
        # اگر یک کانال در یک نمونه کاملا ثابت باشد std=0 می‌شود و تقسیم NaN/inf می‌دهد؛
        # این خط جلوی آن را می‌گیرد.
        std = std.clamp_min(1e-8)
        return (inputs0-mean.unsqueeze(self.dim))/std.unsqueeze(self.dim), targets
