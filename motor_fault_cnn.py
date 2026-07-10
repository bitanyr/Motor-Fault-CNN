#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author : Zhenglong Sun
# Data : 2022-1-7 16:12
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, accuracy_score
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import pandas as pd

import pfdataset as pfd  # این فایل باید حتما با اسم pfdataset.py کنار این اسکریپت باشد

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_classes = 6
num_epochs = 60          # قبلا 200 بود؛ در عمل با Adam مدل تا epoch 3-4 به دقت تست ~100%
                          # می‌رسه و بعدش تغییری نمی‌کنه (early stopping پایین‌تر هم این رو تضمین می‌کنه)
batch_size = 100
learning_rate = 0.0001
early_stop_patience = 8   # اگر دقت تست این‌قدر epoch متوالی >=99.9% بود، زودتر متوقف می‌شه

# parameters for dataset processing
df_path = r'data_all.npy'

pre_disturbance = 300
post_disturbance = 300
features = (6, 7)        # Rotor_Current, Speed
image_size = 30          # پارامتر size برای ToImage؛ فقط همین یک‌جا تعریف می‌شود

# (pre_disturbance+post_disturbance) باید بر image_size بخش‌پذیر باشد،
# وگرنه ToImage در همان اولین batch با ValueError متوقف می‌شود.
assert (pre_disturbance + post_disturbance) % image_size == 0, (
    "pre_disturbance + post_disturbance باید بر image_size بخش‌پذیر باشد "
    f"(الان {pre_disturbance + post_disturbance} % {image_size} = "
    f"{(pre_disturbance + post_disturbance) % image_size})"
)

composed = transforms.Compose([pfd.ToTensor(), pfd.LpNormalize(dim=0), pfd.ToImage(size=image_size)])
# اگر خواستید نرمال‌سازی min-max شبیه نسخه‌ی TensorFlow/baseline_cnn.py را امتحان کنید (به‌جای LpNormalize):
# composed = transforms.Compose([pfd.ToTensor(), pfd.StdNormalize(dim=0), pfd.ToImage(size=image_size)])

# create dataset
dataset = pfd.MFataset(df_path, pre_disturbance, post_disturbance, features, transform=composed)

# split the train and test dataset
train_dataset, test_dataset = random_split(dataset, [round(0.8 * dataset.__len__()), round(0.2 * dataset.__len__())],
                                           generator=torch.Generator().manual_seed(7))

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self, in_channels, input_h, input_w, num_classes):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=16,
                            kernel_size=2,
                            stride=1,
                            padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 2, 1, 0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 2, 1, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 1, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        # قبلا اینجا torch.nn.Linear(26*16*64, 64) با عدد ثابت نوشته شده بود که
        # فقط برای in_channels=2, size=30, pre=post=300 درست از آب درمی‌آمد.
        # با یک forward pass آزمایشی (بدون گرادیان) اندازه‌ی واقعی خروجی conv4
        # را حساب می‌کنیم تا با تغییر features/pre_disturbance/post_disturbance/
        # image_size دیگر به RuntimeError: mat1 and mat2 shapes cannot be
        # multiplied برنخورید.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_h, input_w)
            out = self.conv4(self.conv3(self.conv2(self.conv1(dummy))))
            flat_dim = out.view(1, -1).shape[1]

        self.mlp1 = torch.nn.Linear(flat_dim, 64)
        # 输出矩阵大小为x、输入矩阵大小为n、卷积核大小为f、步长为s、padding 填充为p x=（n-f+2p）/s +1
        self.mlp2 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x


image_width = (pre_disturbance + post_disturbance) // image_size
model = CNNnet(in_channels=len(features), input_h=image_size, input_w=image_width,
               num_classes=num_classes).to(device)


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


model.apply(weights_init)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

# قبلا اینجا SGD با lr=0.0001 و بدون momentum بود (خط Adam کامنت شده بود).
# lr=0.0001 برای Adam معقول است، ولی برای SGD ساده (بدون momentum) خیلی کوچک
# است و همگرایی را به‌شدت کند می‌کند؛ همین می‌تواند دلیل اصلی این باشد که
# مدل به نتایج نزدیک Adam که در گزارش گرفته بودید نمی‌رسید.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# اگر می‌خواهید SGD را امتحان کنید، حتما momentum بدهید و lr را بزرگ‌تر بگیرید، مثلا:
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1)
# نکته: با gamma=1 این scheduler فعلا هیچ افتی در lr ایجاد نمی‌کند (1 برابر یعنی بدون تغییر).
# اگر می‌خواهید واقعا افت نرخ یادگیری داشته باشید gamma را مثلا 0.9 بگذارید.

# for drawing
train_loss_all = []
train_acc_all = []
test_loss_all = []
test_acc_all = []

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    model.train()
    train_loss = 0
    corrects = 0
    train_num = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)

        # Forward pass
        outputs = model.forward(images.float())
        pre_lab = torch.argmax(outputs, 1)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        corrects += torch.sum(pre_lab == labels.data)
        train_num += images.size(0)

    # قبلا scheduler.step() داخل حلقه‌ی batch (هر iteration) صدا زده می‌شد.
    # با step_size=10 منظور اسکریپت "هر 10 epoch" بود نه "هر 10 batch"،
    # پس این خط باید بیرون از حلقه‌ی batch و یک‌بار در هر epoch اجرا شود.
    scheduler.step()

    train_loss_all.append(train_loss / train_num)
    train_acc_all.append(corrects.double().item() / train_num)
    print("Epoch{}, Train Loss: {:.4f} Train Acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))

    model.eval()
    corrects, test_num, test_loss = 0, 0, 0

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        outputs = model.forward(images.float())
        pre_lab = torch.argmax(outputs, 1)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        corrects += torch.sum(pre_lab == labels)
        test_num += images.size(0)

    test_loss_all.append(test_loss / test_num)
    test_acc_all.append(corrects.double().item() / test_num)

    print("Epoch{} Test Loss: {:.4f} Test Acc: {:.4f}".format(epoch, test_loss_all[-1], test_acc_all[-1]))

    # Early stopping: این پروژه در عمل (با Adam) خیلی زود به دقت تست ~100%
    # می‌رسد و می‌ماند؛ به‌جای اجرای بی‌فایده‌ی epoch های باقی‌مانده، وقتی
    # دقت تست چند epoch متوالی >=99.9% بود متوقف می‌شویم.
    if epoch == 0:
        high_acc_streak = 0
    if test_acc_all[-1] >= 0.999:
        high_acc_streak += 1
    else:
        high_acc_streak = 0
    if high_acc_streak >= early_stop_patience:
        print(f"Early stopping: test accuracy >= 99.9% for {early_stop_patience} consecutive epochs.")
        break

plt.figure(figsize=[14, 5])
plt.subplot(1, 2, 1)
plt.plot(train_loss_all, "ro-", label="Train Loss")
plt.plot(test_loss_all, "bs-", label="Val Loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_acc_all, "ro-", label="Train Acc")
plt.plot(test_acc_all, "bs-", label="Test Acc")
plt.xlabel("epoch")
plt.ylabel("Acc")
plt.legend()

plt.show()

# 最后输出模型的精度
predict_labels = []
true_labels = []

for step, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    labels = labels.type(torch.LongTensor)
    labels = labels.to(device)
    outputs = model.forward(images.float())
    pre_lab = torch.argmax(outputs, 1)
    predict_labels += pre_lab.flatten().tolist()
    true_labels += labels.flatten().tolist()

print(classification_report(predict_labels, true_labels, digits=4))
print("Accuracy of the network：", accuracy_score(predict_labels, true_labels))
figure_data = pd.concat(
    [pd.DataFrame({'train_loss_all': train_loss_all}), pd.DataFrame({'test_loss_all': test_loss_all}),
     pd.DataFrame({'train_acc_all': train_acc_all}), pd.DataFrame({'test_acc_all': test_acc_all}),
     pd.DataFrame({'predict_labels': predict_labels}), pd.DataFrame({'true_labels': true_labels})]
    , axis=1)

# قبلا اینجا یک مسیر مطلق ویندوزی هارد-کد شده بود
# (C:\Users\Warrior\Desktop\...) که روی هر سیستم دیگری، یا حتی همان
# سیستم اگر آن پوشه‌ها را نساخته باشید، با FileNotFoundError کرش می‌کند.
# این نسخه یک پوشه‌ی نسبی می‌سازد و همان‌جا ذخیره می‌کند.
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(
    results_dir,
    f'Two_features_CNN_figure_data_rotor_current_speed_{pre_disturbance}-{post_disturbance}.csv'
)
figure_data.to_csv(out_path)
print("results saved to:", out_path)
