import matplotlib.pyplot as plt
import re

import numpy as np
import seaborn as sns
import pandas as pd

cpu_usage = []
mem_usage = []
disk_read = []
disk_write = []

pattern = r"""
    ^
    (\d+\.\d+),\s*               # 时间戳
    CPU:\s*(\d+\.\d+)%,\s*      # CPU使用率
    MEM:\s*(\d+\.\d+)%,\s*      # 内存使用率
    RSS:\s*(\d+\.\d+)MB,\s*     # RSS内存
    Absolute\s+Used\s+MEM:\s*(\d+\.\d+)MB,\s*  # 绝对使用内存
    Read:\s*(\d+\.\d+)MB,\s*    # 读取量
    Write:\s*(\d+\.\d+)MB       # 写入量
    $
"""
with open('resource_log_2.txt', 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        match = re.match(pattern, line, re.VERBOSE)
        if match:
            timestamp, cpu, mem, rss, abs_used_mem, read, write = match.groups()
            cpu_usage.append(float(cpu)*40)
            mem_usage.append(float(abs_used_mem))
            disk_read.append(float(read))
            disk_write.append(float(write))
        else:
            print(f"Line did not match: {line}")

# 绘制CPU使用率
plt.figure(figsize=(8, 3))
window_size = 15
df = pd.DataFrame(cpu_usage, columns=['value'])
# 计算滑动平均
df['cpu_usage'] = df['value'].rolling(window=window_size).mean()
sns.lineplot(data=df, x=df.index, y='cpu_usage')
plt.xlabel('Time')
plt.tight_layout()
plt.show()

# 绘制内存使用率
plt.figure(figsize=(8, 3))
sns.lineplot(x=list(range(len(mem_usage))), y=mem_usage)
plt.xlabel('Time')
plt.ylabel('Memory Usage (MB)')
plt.tight_layout()
plt.show()

# 绘制磁盘读取量
plt.figure(figsize=(8, 3))
disk_read = np.diff(disk_read)  # 计算差分
print(disk_read)
sns.lineplot(x=list(range(len(disk_read))), y=disk_read)
plt.show()
# 绘制磁盘写入量
plt.figure(figsize=(8, 3))
disk_write = np.diff(disk_write)  # 计算差分
sns.lineplot(x=list(range(len(disk_write))), y=disk_write)
plt.show()
