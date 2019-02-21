# Image-Classifier

## TODO
* Improve model architecture for better generalization

## Usage
1. Download CIFAR-10 batches [here](https://www.cs.toronto.edu/~kriz/cifar.html) (python version)
2. Extract to a directory
3. In train.py specify path to extracted directory
```
xTrain, yTrain, xTest, yTest = getCifar10Data('<path-to-batches>')
```
```
pip3 install -r requirements.txt
python3 train.py
python3 classify.py --image <image-path>
```

Defaulted to use Cifar10 dataset, but can be applied to a local dataset.

![image-classifier](https://user-images.githubusercontent.com/36581610/52970211-063d7700-3381-11e9-96fd-9d517f11267b.PNG)
