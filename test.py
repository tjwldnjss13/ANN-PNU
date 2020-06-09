from PIL import Image
import numpy as np

fn = './digit data/{}.{}.png'.format(2, 2)
img = Image.open(fn)
img.show()
img = np.array(img)
print(img.shape)

for i in range(16, 1, -1):
    for j in range(16):
        if i < 14:
            img[j, i] = img[j, i - 2]
        elif i < 15:
            img[j, i] = 255

img = Image.fromarray(img)
img.show()