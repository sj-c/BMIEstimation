import re
import torch
import numpy as np

file_name = "0_F_22_157480_7493346.jpg"

ret = re.match(r"\d+?_([FMfm])_(\d+?)_(\d+?)_(\d+).+", file_name)
sex = 0 if (ret.group(1) == 'F' or ret.group(1) == 'f') else 1
height = int(ret.group(3)) / 100000
weight = int(ret.group(4)) / 100000
BMI = torch.from_numpy(np.asarray((int(ret.group(4)) / 100000) / (int(ret.group(3)) / 100000) ** 2))

print(f"height: {height}")
print(f"weight: {weight}")
print(sex)