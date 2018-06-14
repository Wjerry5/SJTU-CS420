## Digit Classification on MNIST

[Zhenghui Wang](zhenghuiwang.net) & [Ruijie Wang](https://github.com/Wjerry5)

### CNN
* for training and single model testing,
```shell
python mnist_cnn_train.py --version model_version
```

* for model ensemble,
```shell
python mnist_cnn_test.py --model-dir model_dir --batch-size batch_size --use-ensemble True
```

### CapsNet

* for training and test:

```shell
python mnist_cnn_train.py
```

* for image reconstruction:
```shell
python capsulenet.py -t -w model/trained_model.h5 --digit digit
```

### Deep Forest

- for training and test a cascaded deep forest (CA) 
```shell
python mnist.py --model ca
```
- for training and test a cascaded deep forest with mutli-grained scanning (GC)
```shell
python mnist.py --model gc
```

### Domain Adaptation

- for training the source only model
```shell
python source_only.py
```
- for training MMD
```shell
python mmd.py
```

- for training DANN
```shell
python dann.py
```

- for training CORAL
```shell
python coral.py
```

- for training WDGRL

```shell
python wd.py
```


### Reference

* [CapsNet-Keras](https://github.com/XifengGuo/CapsNet-Keras)
* [CapsNet-Tensorflow](https://github.com/naturomics/CapsNet-Tensorflow)

