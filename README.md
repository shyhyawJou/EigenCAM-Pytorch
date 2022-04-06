# Overview
The implementation of [EigenCAM](https://arxiv.org/abs/2008.00299) for getting the attention map of CNN

# Usage
My code is very easy to use

### step 1: create the EigenCAM object and model

```
model = your_pytorch_model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
eigencam = EigenCAM(model, device)
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

# create the EigenCAM object 
eigencam = EigenCAM(model, device)  

img = Image.open(img_path)  
img_tensor = preprocess(img).unsqueeze_(0).to(device)  
outputs, overlay = eigencam.get_heatmap(img, img_tensor)
_, pred_label = outputs.max(1)
pred_class = class_name[pred_label.item()]
conf = F.softmax(outputs, 1).squeeze()[pred_label]

print("Result:", pred_class)
print("Confidence:", conf)

# show the heatmap
overlay.show() 
```

# Reference
Original paper:  
[Eigen-CAM: Class Activation Map using Principal Components](https://arxiv.org/abs/2008.00299)  
