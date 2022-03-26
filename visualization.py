import torch
import torch.nn as nn

from PIL import Image
import cv2 as cv
from matplotlib import cm
import numpy as np



class EigenCAM():
    def __init__(self, model, device):
        layerName = self.get_layer_name(model)
        for name, layer in model.named_children():
            if name == layerName:
                layer.register_forward_hook(self.forward_hook)
                break

        model = model.to(device)
        self.model = model
        self.device = device
        self.feature_maps = {}

    def get_heatmap(self, img, img_tensor):
        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():    
            output = self.model(img_tensor)   
            feature_maps = self.feature_maps["output"]
            
            u, s, v = torch.linalg.svd(feature_maps, full_matrices=False)
            s = s[..., None]
            cam = u[..., :, :1] @ s[..., :1, :] @ v[..., :1, :]
            cam = cam.sum(1)
            cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
            cam = cam.to(dtype=torch.uint8, device="cpu")
            cam = cam.numpy().transpose(1,2,0)
            cam = cv.resize(cam, img.size[:2], interpolation=4)
            cam = np.uint8(255 * cm.get_cmap("jet")(cam.squeeze()))

            if not isinstance(img, np.ndarray):
                img = np.asarray(img)
            img_size = img.shape[:2][::-1] # w, h

            overlay = np.uint8(0.6*img + 0.4 * cam[:,:,:3])
            overlay = Image.fromarray(overlay)
            if overlay.size != img_size:
                overlay = overlay.resize(img_size, Image.BILINEAR)

        return output, overlay

    def get_layer_name(self, model):
        for n, m in model.named_children():
            # AdaptiveAvgPool2d or nn.AvgPool2d
            if isinstance(m, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                name = tmp
            tmp = n
        return name

    def forward_hook(self, module, x, y):
        #self.feature_maps["input"] = x
        self.feature_maps["output"] = y


