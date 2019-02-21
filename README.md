# Image-Classifier

## TODO
* Improve model architecture for better generalization

## Instalation
Git clone the repository and ```cd``` into the directory
```
git clone https://github.com/greerviau/Image-Classifier.git && cd Image-Classifier
```
Download CIFAR-10 batches [here](https://www.cs.toronto.edu/~kriz/cifar.html) (python version) and extract
In train.py specify path to extracted directory
```
xTrain, yTrain, xTest, yTest = getCifar10Data('<path-to-batches>')
```

## Usage
```
pip3 install -r requirements.txt
python3 train.py
python3 classify.py --image <image-path>
```

Defaulted to use Cifar10 dataset, but can be applied to a local dataset.

![image-classifier](https://user-images.githubusercontent.com/36581610/52970211-063d7700-3381-11e9-96fd-9d517f11267b.PNG)
