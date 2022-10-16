# Overview
The implementation of [Eigen-CAM](https://arxiv.org/abs/2008.00299)

# Example
![](assets/n01669191_46.JPEG)
![](assets/heatmap.jpg)

# Usage
- The example image is generate from mobilenetv2:  
```
python show.py -d cpu -img assets/n01669191_46.JPEG -layer features.18.0
```

- for custom model  
model path is a file including weight and architecture.
```
python show.py -d cpu -img assets/n01669191_46.JPEG -layer {layer name} -m {your model path}
```
- Get predict label  
  Very easy, you can refer to `show.py`.
  
# Note
- Remenber to check whether the image preprocess is the same as yours, if not, you should alert the preprocess in the `show.py` or the result will be wrong.
- If you have cuda, you can just replace the "cpu" to "cuda".
- If you don't specify any layer, my code will use the last layer before global average pooling  to plot heatmap.

