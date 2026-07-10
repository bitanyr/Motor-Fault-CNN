#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# این یک اسکریپت تشخیصی جداست، جای motor-fault-cnn.py رو نمی‌گیره.
# تفاوتش با نسخه‌ی اصلی: به‌جای split تصادفی، برای هر کلاس، ۸۰٪ اول (به
# ترتیبی که تو آرایه هستند) train و ۲۰٪ آخر test می‌شه. اگر ترتیب نمونه‌ها
# تو هر کلاس تقریبا ترتیب زمانی ضبط باشه (خیلی معموله)، این یک تست
# سخت‌گیرانه‌تره: می‌بینیم آیا مدل به یک نشست/بازه‌ی زمانی خاص وابسته
# شده یا واقعا امضای فیزیکی عیب رو یاد گرفته.
#
# فقط این رو اجرا کنید و همون خروجی متنی (لاگ epoch ها) رو برام کپی کنید،
# نیازی به فرستادن فایل نیست.

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset

import pfdataset as pfd

torch.manual_seed(0)

df_path = 'data_all.npy'
pre_disturbance, post_disturbance = 300, 300
features = (6, 7)
image_size = 30
image_width = (pre_disturbance + post_disturbance) // image_size
batch_size = 100
num_epochs = 40

# ترتیب دقیقا همونیه که pfdataset.MFataset استفاده می‌کنه:
class_order = [0, 1, 2, 3, 4, 5]

# فقط برای اندازه‌ی هر کلاس، مستقیم از فایل می‌خونیم (بدون اسلایس/ترنسفرم)
raw = np.load(df_path, allow_pickle='TRUE').item()
class_sizes = [raw[k, 0].shape[0] for k in class_order]
print("اندازه‌ی هر کلاس به ترتیبی که در دیتاست به هم می‌چسبند:", class_sizes)

class _Compose:
    def __init__(self, tlist): self.tlist = tlist
    def __call__(self, sample):
        for t in self.tlist: sample = t(sample)
        return sample

composed_pipeline = _Compose([pfd.ToTensor(), pfd.LpNormalize(dim=0), pfd.ToImage(size=image_size)])

dataset = pfd.MFataset(df_path, pre_disturbance, post_disturbance, features, transform=composed_pipeline)
assert len(dataset) == sum(class_sizes), "چیزی با اندازه‌ها جور درنمیاد، لطفا به من بگو"

# اندیس‌های train/test رو دستی، به‌ترتیب زمانی درون هر کلاس می‌سازیم
train_idx, test_idx = [], []
offset = 0
for n_k in class_sizes:
    n_train_k = round(0.8 * n_k)
    train_idx.extend(range(offset, offset + n_train_k))
    test_idx.extend(range(offset + n_train_k, offset + n_k))
    offset += n_k

print(f"تعداد train={len(train_idx)}  تعداد test={len(test_idx)}")
print("همپوشانی train/test (باید 0 باشه):", len(set(train_idx) & set(test_idx)))

train_ds = Subset(dataset, train_idx)
test_ds = Subset(dataset, test_idx)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)


class CNNnet(nn.Module):
    def __init__(self, in_channels, input_h, input_w, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 16, 2, 1, 0), nn.BatchNorm2d(16), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 2, 1, 0), nn.BatchNorm2d(32), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 2, 1, 0), nn.BatchNorm2d(64), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 2, 1, 0), nn.BatchNorm2d(64), nn.ReLU())
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_h, input_w)
            out = self.conv4(self.conv3(self.conv2(self.conv1(dummy))))
            flat_dim = out.view(1, -1).shape[1]
        self.mlp1 = nn.Linear(flat_dim, 64)
        self.mlp2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x); x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


model = CNNnet(len(features), image_size, image_width, 6)
model.apply(weights_init)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(num_epochs):
    model.train()
    tr_loss, tr_correct, tr_n = 0, 0, 0
    for images, labels in train_loader:
        labels = labels.type(torch.LongTensor)
        outputs = model(images.float())
        pred = torch.argmax(outputs, 1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tr_loss += loss.item() * images.size(0); tr_n += images.size(0)
        tr_correct += (pred == labels).sum().item()

    model.eval()
    te_correct, te_n, te_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.type(torch.LongTensor)
            outputs = model(images.float())
            pred = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)
            te_loss += loss.item() * images.size(0); te_n += images.size(0)
            te_correct += (pred == labels).sum().item()

    print(f"epoch{epoch}: train_loss={tr_loss/tr_n:.4f} train_acc={tr_correct/tr_n:.4f}  "
          f"test_loss={te_loss/te_n:.4f} test_acc={te_correct/te_n:.4f}")
