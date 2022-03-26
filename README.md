# Overview
The implementation of [EigenCAM](https://arxiv.org/abs/2008.00299) for getting the attention map of CNN

# Usage
My code is very easy to use

### step 1: create the EigenCAM object and model
if `layer_for_cam` is None, my code will use the last layer in front of last global average pooling layer. Default is None.  
if there is no global average pooling layer in your model, remember to specify the `layer_for_cam`(string).
  
```
model = your_pytorch_model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
eigencam = EigenCAM(model, device, layer_for_cam=None)
```
### step 2: get the heatmap
```
preprocess = your_preprocess
img = Image.open(img_path)  
img_tensor = preprocess(img).unsqueeze_(0).to(device)  
outputs, overlay = eigencam.get_heatmap(img, img_tensor)
overlay.show() # show the heatmap
```

# Complete Example
```
from PIL import Image

import torch
from torchvision import transfoms as T

from visualization import EigenCAM


class_name = ['Class A', 'Class B']
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  
preprocess = T.Compose([
                        T.ToTensor()
                       ])  

eigencam = EigenCAM(model, device) # create the GradCAM object  

img = Image.open(img_path)  
img_tensor = preprocess(img).unsqueeze_(0).to(device)  
outputs, overlay = eigencam.get_heatmap(img, img_tensor)
_, pred_label = outputs.max(1)
pred_class = class_name[pred_label.item()]
probability = F.softmax(outputs, 1).squeeze()[pred_label]

print("Result:", pred_class)
print("Probability:", probability)
overlay.show() # show the heatmap
```

# Reference
Original paper:  
[Eigen-CAM: Class Activation Map using Principal Components](https://arxiv.org/abs/2008.00299)  
