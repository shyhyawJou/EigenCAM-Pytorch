from PIL import Image
import argparse

import torch
from torchvision import transforms as T
from torchvision.models import mobilenet_v2

from cam import EigenCAM





def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='pytorch model path (a file incusive of weight and architecture)')
    parser.add_argument('-d', default='cpu', help='device ("cpu" or "cuda")')
    parser.add_argument('-img', help='img path')
    parser.add_argument('-layer', help='layer name to plot heatmap')
    return parser.parse_args()
    



def main():
    arg = get_arg()
    
    preprocess = T.Compose([T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225])])
    
    device = arg.d
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('There is no cuda !!!')
    
    if arg.m is None:
        model = mobilenet_v2(True).eval()
    else:
        model = torch.load(arg.m).eval()
    
    cam_obj = EigenCAM(model, arg.d, preprocess, arg.layer)
    
    print('\ndevice:', arg.d)
    print('layer Name to plot heatmap:', cam_obj.layer_name)
    print('img:', arg.img)
    
    img = Image.open(arg.img).convert('RGB')
    # output is torch Tensor, overlay is ndarray
    output, overlay = cam_obj.get_heatmap(img)
    print('\nPredict label:', output.max(1)[1].item())
    
    Image.fromarray(overlay).show()





if __name__ == "__main__":
    main()