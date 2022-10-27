import torch
import torch.nn as nn

import cv2
import numpy as np




class CAM:
    '''
    Base Class
    '''
    def __init__(self, model, device, preprocess, layer_name=None):  
        if layer_name is None:
            self.layer_name = self.get_layer_name(model) 
        else:
            self.layer_name = layer_name
            
        self.model = model.to(device)
        self.device = device
        self.prep = preprocess
        self.feature = {}

        self.register_hook()
        
    def get_heatmap(self, img):
        pass
                                         
    def get_layer_name(self, model):
        layer_name = None

        for name, module in model.named_modules():
            if hasattr(module, 'inplace'):
                module.inplace = False

            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                layer_name = last_name
            last_name = name
        
        if layer_name is None:
            raise ValueError('Defaultly use the last layer before global average ' 
                             'pooling to plot heatmap. However, There is no such '
                             'layer in this model.\n'
                             'So you need to specify the layer to plot heatmap.\n'
                             'Arg "layer_name" is the layer you should specify.\n'
                             'Generally, the layer is deeper, the interpretaton ' 
                             'is better.')

        return layer_name

    def forward_hook(self, module, x, y):
        self.feature['output'] = y

    def register_hook(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(self.forward_hook)
                break
        else:
            raise ValueError(f'There is no layer named "{self.layer_name}" in the model')

    def check(self, feature):
        if feature.ndim != 4 or feature.shape[2] * feature.shape[3] == 1:
            raise ValueError(f'Got invalid shape of feature map: {feature.shape}, '
                              'please specify another layer to plot heatmap.') 



class EigenCAM(CAM):
    def __init__(self, model, device, preprocess, layer_name=None):
        super().__init__(model, device, preprocess, layer_name)

    def get_heatmap(self, img):
        with torch.no_grad():
            tensor = self.prep(img)[None, ...].to(self.device)
            output = self.model(tensor)
            feature = self.feature['output']
            self.check(feature)
            
            _, _, vT = torch.linalg.svd(feature)
            v1 = vT[:, :, 0, :][..., None, :]
            
            cam = feature @ v1.repeat(1, 1, v1.shape[3], 1)
            cam = cam.sum(1)
            cam -= cam.min()
            cam = cam / cam.max() * 255
            cam = cam.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
            cam = cv2.resize(cam, img.size)
            cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)        

            if not isinstance(img, np.ndarray):
                img = np.asarray(img)
            img_size = img.shape[:2][::-1] # w, h

            overlay = np.uint8(0.6 * img + 0.4 * cam)

            if overlay.size != img_size:
                overlay = cv2.resize(overlay, img_size)

        return output, overlay
    
