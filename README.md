# Airbus Ship Detection Challenge
## Find ships on satellite images segmentation with a convolution neural network (U-net)


<p float="left">
  <img src="/data/27e4f5a1c.jpg" width="250" />
  <img src="/data/0b7359c38.jpg" width="250" />
</p>

This repository contains the implementation of a convolutional neural network used to segment,  that detects  ships in satellite images . This is a binary classification task: the neural network predicts if each pixel has ships  image is either or not.
The neural network structure is derived from the *U-Net* architecture, described in this
<a href="http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/" target="blank">Article</a>


## You can install dependency:
```pip
    pip install -r requirements.txt
```
* Modules:
   - keras
   - sklearn
   - pandas
   - numpy
   - matplotlib
   - scikit-image
   - tensorflow
###  Struct all directory
    .
    ├── train                                      # Notebooks of train with difference losess and metrics
    ├────U-net_dice_comb_binar_cross.ipynb         # train combined `dice_loss` with `bynary_cross` loss 
    ├────U-net_iou_coef.ipynb                      # train `iou_loss`  
    ├────U-net_dice_coef.ipynb                     # train `dice_loss`
    ├── submission                                 # Notebooks with difference losess submission
    ├────dice_with_binary_cross_loss.ipynb         # `dice_loss` with `bynary_cross`
    ├────dice_coef_loss.ipynb                      # `dice_loss` 
    ├────iou_coef_loss.ipynb                       # `iou_coef_loss` 
    ├──model
    ├────U-net_dice_coef                           # trained model `dice_loss` 
    ├────U-net_dice_comb_binary_crossentropy       # trained model `dice_loss` with `bynary_cross`
    ├────U-net_iou_coef                            # trained model `iou_coef_loss`
    ├────callbacks.py                              # Clallbacks for training model
    ├────losses.py                                 # Losess functions
    ├────metrics.py                                # Metrics
    ├────u_net.py                                  # Model U-net
    |──data
    ├────balanced_and_split_data.ipynb             # Notebook with balance and split data
    ├────understand_data.ipynb                     # Notebook with analysis data
    ├──utils.py                                    # Help tools for preprocess data and plot data.
    ├──params.py                                   # Constants variables.
    ├──train.py                                    # Train model.
    └── README.md

## Train data
You can find discption and data source: [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/data)

# Implementation details

### U-Net layers
Convolutional layers implement
```python
def fire(x, filters, kernel_size, dropout):
    y1 = Conv2D(filters, kernel_size, activation='relu', padding='same')(x)
    if dropout is not None:
        y1= Dropout(dropout)(y1)
    y2 = Conv2D(filters, kernel_size, activation='relu', padding='same')(y1)
    y3 = BatchNormalization(momentum=momentum)(y2)
    return y3

def fire_module(filters, kernel_size, dropout=0.1):
    return lambda x: fire(x, filters, kernel_size, dropout)
```
Used with `MaxPooling2D` also` Dropout`, but in the case when moving down convolution.
```python
    down = fire_module(8, (3, 3))(input)
    down = MaxPooling2D((2, 2))(down)
```
With each subsequent layer doubling until being squeezed into matrix size 8x8 and deep 256 in center layer.

The up-convolution layers use `Conv2DTranspose` which doubles the size of the input followed by concatenate([down, up]) which combines
the learned weights of the contraction path layer to the expansion layer.
```python
def fire_up(x, filters, kernel_size, concat_layer):
    y1 = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
    y2 = concatenate([y1, concat_layer])
    y3 = fire_module(filters, kernel_size)(y2)
    y3 = fire_module(filters, kernel_size, dropout=None)(y2)
    return y3

def up_fire_module(filters, kernel_size, concat_layer):
    return lambda x: fire_up(x, filters, kernel_size, concat_layer)
```
Implement with pass in layer for need concatenate.
```python
 # get in param layer for concat.
up6 = up_fire_module(128, (3, 3), down5)(down6)
```

### Loss Function
For experiments i to used three a popular solutions for segmentation images.
1. <strong>Dice coefficient</strong>
This is  Dice coefficient  which shows a measure of similarity - in this case,
showing a measure of the area of correctly marked segments (the ratio of the intersection area to the area of association).
```python
def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
```

2. <strong>Dice coefficient combination with binary crossentropy</strong>

```python
def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
```
3. <strong>IoU, also known as the Jaccard Index</strong>
IoU is the area of overlap between the predicted segmentation and the true input,
divided by the area of union between the predicted segmentation and the true input.
[Difference between IoU and Dice coefficient](https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou/276144#276144)
```python
def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    return K.mean((intersection + smooth) / (union + smooth), axis=0)

def iou_coef_loss(y_true, y_pred):
    return 1.-iou_coef(y_true, y_pred)
```

# References
[Understanding the map evaluation metric for object detection](https://medium.com/@timothycarlen/understanding-the-map-evaluation-metric-for-object-detection-a07fe6962cf3)<br />

[Help to understand loss func for segmentation](https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/)<br />

[Custom loss functions in keras](https://medium.com/@j.ali.hab/on-custom-loss-functions-in-keras-3af88cf59e48)<br />