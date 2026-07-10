#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
# این اسکریپت رو کنار data_all.npy بذارید و اجرا کنید. یک عکس به اسم
# samples_preview.png می‌سازه که برای هر ۶ کلاس، ۵ نمونه‌ی واقعی از
# جریان روتور (رنگ آبی) و سرعت (رنگ نارنجی) رو نشون می‌ده، با یک خط
# عمودی قرمز روی نقطه‌ی وقوع عیب (index=400).
#
# بعد از اجرا، همون فایل samples_preview.png رو برام آپلود کنید (دقیقا
# مثل عکسی که از MATLAB فرستادید) تا مستقیم نگاهش کنم.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df_path = 'data_all.npy'
n_show = 5          # چند نمونه از هر کلاس نشون داده بشه
pre, post = 300, 300
features = (6, 7)   # Rotor_Current, Speed

label_names = {
    0: 'Rotor_Current_Faild (قطع)',
    1: 'Disconnect_Phase (قطع فاز)',
    2: 'Rotor_Current_Failed_R (رئوستا)',
    3: 'Short_between_two_phases',
    4: 'Short_phases_Ln_G',
    5: 'No_failed (بدون عیب)',
}

data = np.load(df_path, allow_pickle='TRUE').item()

fig, axes = plt.subplots(6, 1, figsize=(10, 16), sharex=True)

for k in range(6):
    arr = data[k, 0][:, (400 - pre):(400 + post), features]  # shape: [n_k, pre+post, 2]
    n_k = arr.shape[0]
    idxs = np.random.RandomState(0).choice(n_k, size=min(n_show, n_k), replace=False)

    ax = axes[k]
    for i in idxs:
        ax.plot(arr[i, :, 0], color='tab:blue', alpha=0.6, linewidth=1)
        ax.plot(arr[i, :, 1], color='tab:orange', alpha=0.6, linewidth=1)
    ax.axvline(pre, color='red', linestyle='--', linewidth=1)
    ax.set_title(f"label={k}  ({label_names[k]})   n_total={n_k}", fontsize=10)
    ax.set_ylabel("value")

axes[0].legend(['Rotor_Current', 'Speed', 'لحظه‌ی عیب (t=400)'], loc='upper right', fontsize=8)
axes[-1].set_xlabel("time index (نسبت به شروع پنجره)")
plt.tight_layout()
plt.savefig('samples_preview.png', dpi=120)
print("ذخیره شد: samples_preview.png -- همین فایل رو برام آپلود کنید")
