import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

d = loadmat("./dataset/Data/Normalized/p00/day01.mat", squeeze_me=True, struct_as_record=False)

index = 1
scale = 30
img = d["data"].right.image[index]
gaze = d["data"].right.gaze[index]

h, w = img.shape
cx, cy = w / 2, h / 2

plt.imshow(img, cmap="gray")
plt.arrow(cx, cy, gaze[0] * scale, gaze[1] * scale, color="red", head_width=3)
plt.title(f"gaze: {np.round(gaze, 2)}")
plt.show()