# PEL：Leaf Cultivar Identification via Prototype-enhanced Learning

## Dataset
You can download the datasets from the links below:

+ [SoyLocal, SoyGlobal, SoyGene, SoyAge, and Cotton](https://pan.baidu.com/s/1bPJYmFGWJg2eTr5Ipfw6uA). Access code: iccv
+ [SoyCultivar200](https://drive.google.com/file/d/1XsWZPYYrDsCwAy5r4t3I1F_lOOrGGhgf/view)
+ [Sweet cherry](https://github.com/WeizhenLiuBioinform/mfcis)


## Run the experiments.
Run train.py to train the model, e.g., train on the Cotton dataset with resnet50.

    $ python train.py -a resnet50 --feat_dim 2048 --epoch 120 --dataset cotton80 --save True --proto_path *.np
  
Run train.py to train the model, e.g., train on the Cotton dataset with DenseNet121.

    $ python train.py -a DenseNet121 --feat_dim 1024 --epoch 120 --dataset cotton80 --save True --proto_path *.np
