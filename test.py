from filters import *
from constrained_network import *
from PIL import Image
import cv2 as cv

g = gabor(1, np.pi / 4, 4 * np.pi, 0, 1)
plt.imshow(g)
plt.show()

img = Image.open('digit data/8.0.png').convert('L')

cn = ConstrainedNet()
img = cn.preprocessed_image(img, g)
print(img.shape)
plt.imshow(img)
plt.show()

kernel = cv.getGaborKernel((5, 5), 1, np.pi / 4, .5, 0, ktype=cv.CV_32F)





