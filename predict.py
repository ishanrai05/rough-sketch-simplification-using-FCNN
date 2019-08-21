import numpy as np
import cv2

import torch
from torch.autograd import Variable


def predict(model, img_path, device):
    from utils import preprocess, transform

    model.eval()
    with torch.no_grad():
        in_shape = np.asarray(cv2.imread(img_path)).shape
        img = preprocess(img_path)
        fin_shape = np.asarray(img).shape
        img = transform(img)
        img = Variable(img).to(device)
        img = img.view(1, 1, fin_shape[0],fin_shape[1])
        output = model(img)
        img = ((255*output.cpu().clone().detach().numpy()).squeeze().squeeze())
        img = cv2.resize(img, (in_shape[1],in_shape[0]),interpolation = cv2.INTER_AREA)
        return img
