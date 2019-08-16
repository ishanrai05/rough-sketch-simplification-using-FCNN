import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from constants import height, width
from predict import predict
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])


def preprocess(img_path):
    img = Image.open(img_path)
    img = img.convert('L')
    img = img.resize((height, width))
    img = np.asarray(img)
    return img

def samples(input_image, target_image, model):
    _,figs = plt.subplots(1,3)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.rcParams['figure.figsize'] = [15, 15]
    figs[0].set_title('Input')
    figs[0].imshow(Image.open(input_image),cmap = 'gray')

    figs[1].set_title('Prediction')
    figs[1].imshow(predict(model, input_image),cmap='gray')

    figs[2].set_title('Target')
    figs[2].imshow(Image.open(target_image),cmap = 'gray')

    plt.show()

# this function is used during training process, to calculation the loss and accuracy
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count