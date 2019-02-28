# 前言

在机器学习的场景中，训练数据经常会特别大，训练可能要持续好几天甚至上周。如果中途机器断电或是发生意外不得不中断训练过程，那就得不偿失。

使用keras 高阶API，可以很简单的保存训练现场，可以让我们很容易的恢复到上次落下的地方继续训练。

# 思路

存在两个巨大的问题：

1. 继续训练会有这样一个影响，就是我们的学习率如果是不固定的，比如前100 epoch 学习率0.1，后100 epoch 学习率要0.01，这样的话，epoch这个数据比需要被记录下来。
2. 如果设`save_best_only=True`，会一遍遍覆盖旧的`.h5`文件，当重新加载的时候，`self.best`是正负无穷（正负取决于`mointor`是`val_loss`还是`val_acc`等）

要做到从上次落下的地方继续训练，首先需要明确我们保存模型的方法是什么！

1. 保存全部训练数据(`save_weights_only=False`) or 只保存weights(`save_weights_only=True`)
2. 保存最棒的版本(`save_best_only=True`) or 保存最新的版本(`save_best_only=False`)
3. 新`.h5`文件覆盖老的文件 or 每一个文件都使用`epoch`区别开来

> 使用`'./path/to/somename-{epoch:04d}.h5'.`作为文件名即可使得每次存储的文件名都有个`epoch`数作为后缀。
>
> 原因在于，在`keras/callbacks.py`中源码是如此定义的：
>
> ```python
> def on_epoch_end(self, epoch, logs=None):
>  logs = logs or {}
>  。。。。。。
>      filepath = self.filepath.format(epoch=epoch + 1, **logs)
> ```

其次，我们必须知道**继续训练**的充要条件是什么：

1. 知道在中断时，执行到哪一个`epoch`，
2. 知道在中断时，最高的`val_acc`或最小的`val_loss`是多少

针对上面提出来的7点问题，我们下文将探讨如何设计一个能满足上述要求的方法。

# 导入相关依赖

```python
from keras.callbacks import ModelCheckpoint
import h5py
import numpy as np
import keras
```

上述引入使用的是`keras`自身API，如果您需要使用`tensorflow.keras`请导入如下依赖：

```python
from tensorflow.keras.callbacks import ModelCheckpoint
import h5py
import numpy as np
from tensorflow import keras
```



其中`ModelCheckpoint`就是我们今天的猪脚。

有关于更多`callback`的内容，请参阅官方文档：<https://keras-cn.readthedocs.io/en/latest/other/callbacks/>



# 准备训练数据

为了快速演示，我们使用mnist数据集

```python
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0
```

> 如果下载数据集的过程中出现问题，请选择代理 或 在`keras api`和`tensorflow.keras`之间切换，前者将数据托管在amazon上，后者在google上。请酌情选择。

# 构建一个简单的神经网络

```python
# Returns a short sequential model
def create_model():
  model = keras.models.Sequential([
    keras.layers.Dense(512, activation="relu", input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation="softmax")
  ])

  model.compile(optimizer=keras.optimizers.Adam(),
                loss=keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

  return model

# Create a basic model instance
model = create_model()
model.summary()
```

```console
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 512)               401920    
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 10)                5130      
=================================================================
Total params: 407,050
Trainable params: 407,050
Non-trainable params: 0
```

```python
model = create_model()
```

# 准备输出

创建h5模型输出的目录

```python
import os
if not os.path.exists('./results/'):
    os.mkdir('./results/')
```

从`ModelCheckpoint`继承一个子类用于拓展

```python
class MetaCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, training_args=None, meta=None):

        super(MetaCheckpoint, self).__init__(filepath,
                                             monitor=monitor,
                                             verbose=verbose,
                                             save_best_only=save_best_only,
                                             save_weights_only=save_weights_only,
                                             mode=mode,
                                             period=period)

        self.filepath = filepath
        self.new_file_override = True
        self.meta = meta or {'epochs': [], self.monitor: []}

        if training_args:
            self.meta['training_args'] = training_args

    def on_train_begin(self, logs={}):
        if self.save_best_only:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.best = max(self.meta[self.monitor], default=-np.Inf)
            else:
                self.best = min(self.meta[self.monitor], default=np.Inf)

        super(MetaCheckpoint, self).on_train_begin(logs)

    def on_epoch_end(self, epoch, logs={}):
        # 只有在‘只保存’最优版本且生成新的.h5文件的情况下
        if self.save_best_only:
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                self.new_file_override = True
            else:
                self.new_file_override = False

        super(MetaCheckpoint, self).on_epoch_end(epoch, logs)

        # Get statistics
        self.meta['epochs'].append(epoch)
        for k, v in logs.items():
            # Get default gets the value or sets (and gets) the default value
            self.meta.setdefault(k, []).append(v)

        # Save to file
        filepath = self.filepath.format(epoch=epoch, **logs)

        if self.new_file_override and self.epochs_since_last_save == 0:
            # 只有在‘只保存’最优版本且生成新的.h5文件的情况下 才会继续添加meta
            with h5py.File(filepath, 'r+') as f:
                meta_group = f.create_group('meta')
                meta_group.attrs['training_args'] = yaml.dump(
                    self.meta.get('training_args', '{}'))
                meta_group.create_dataset('epochs', data=np.array(self.meta['epochs']))
                for k in logs:
                    meta_group.create_dataset(k, data=np.array(self.meta[k]))

```

其中`meta`是`.h5`文件里面的`group`，意思就是指训练过程中的数据信息，相当于一个数据子集，和`data`，`label`等并列。

下面我将讲解`MetaCheckpoint`的关键代码

## 讲解MetaCheckpoint的关键代码

在类初始化的时候，初始化`epochs`，和`monitor`(一般即`val_acc`或`val_loss`)

```python
self.meta = meta or {'epochs': [], self.monitor: []}
```

`on_train_begin`当训练开始的时候，如果是`save_best_only`状态的话

```python
self.best = max(self.meta[self.monitor], default=-np.Inf)
```

把最优值赋值为`meta`里面保存的最优值

如果越小越优的话：

```python
self.best = min(self.meta[self.monitor], default=np.Inf)
```

反过来即可。

`on_epoch_end`在一个epoch训练结束之后：

```python
self.meta['epochs'].append(epoch)
```

把结果附加到`meta`里面去

然后视情况而定，是否要保存到`.h5`文件

1. `self.new_file_override`说明经过判断，认为此次训练产生了更好的`val_acc`或`val_loss`（取决于你的设定），`best`值被更新。
2. `self.epochs_since_last_save == 0`说明模型文件已经存储至你指定的路径。

如果满足以上两点要求，我们就可以往模型文件里面附加本次训练的`meta`数据。

# 调用

## 第一次训练

创建`MetaCheckpoint`实例

```python
checkpoint = MetaCheckpoint('./results/pointnet.h5', monitor='val_acc',
                            save_weights_only=True, save_best_only=True,
                            verbose=1)
```

开始训练：

```python
model.fit(train_images, train_labels, epochs = 10,
          validation_data = (test_images,test_labels),
          callbacks = [checkpoint]) ## 调用callbacks
```

```console
Train on 1000 samples, validate on 1000 samples
Epoch 1/10
1000/1000 [==============================] - 0s 454us/step - loss: 1.1586 - acc: 0.6590 - val_loss: 0.7095 - val_acc: 0.7770

Epoch 00001: val_acc improved from -inf to 0.77700, saving model to ./results/pointnet.h5
Epoch 2/10
1000/1000 [==============================] - 0s 321us/step - loss: 0.4256 - acc: 0.8780 - val_loss: 0.5295 - val_acc: 0.8320

Epoch 00002: val_acc improved from 0.77700 to 0.83200, saving model to ./results/pointnet.h5
Epoch 3/10
1000/1000 [==============================] - 0s 305us/step - loss: 0.2859 - acc: 0.9240 - val_loss: 0.4457 - val_acc: 0.8620

Epoch 00003: val_acc improved from 0.83200 to 0.86200, saving model to ./results/pointnet.h5
Epoch 4/10
1000/1000 [==============================] - 0s 319us/step - loss: 0.2093 - acc: 0.9570 - val_loss: 0.4540 - val_acc: 0.8540

Epoch 00004: val_acc did not improve from 0.86200
Epoch 5/10
1000/1000 [==============================] - 0s 295us/step - loss: 0.1518 - acc: 0.9670 - val_loss: 0.4261 - val_acc: 0.8650

Epoch 00005: val_acc improved from 0.86200 to 0.86500, saving model to ./results/pointnet.h5
Epoch 6/10
1000/1000 [==============================] - 0s 268us/step - loss: 0.1101 - acc: 0.9850 - val_loss: 0.4211 - val_acc: 0.8570

Epoch 00006: val_acc did not improve from 0.86500
Epoch 7/10
1000/1000 [==============================] - 0s 350us/step - loss: 0.0838 - acc: 0.9900 - val_loss: 0.4040 - val_acc: 0.8700

Epoch 00007: val_acc improved from 0.86500 to 0.87000, saving model to ./results/pointnet.h5
Epoch 8/10
1000/1000 [==============================] - 0s 261us/step - loss: 0.0680 - acc: 0.9920 - val_loss: 0.4097 - val_acc: 0.8660

Epoch 00008: val_acc did not improve from 0.87000
Epoch 9/10
1000/1000 [==============================] - 0s 272us/step - loss: 0.0530 - acc: 0.9960 - val_loss: 0.4001 - val_acc: 0.8750

Epoch 00009: val_acc improved from 0.87000 to 0.87500, saving model to ./results/pointnet.h5
Epoch 10/10
1000/1000 [==============================] - 0s 306us/step - loss: 0.0392 - acc: 0.9980 - val_loss: 0.3981 - val_acc: 0.8670

Epoch 00010: val_acc did not improve from 0.87500
```

## 查看保存的模型

定义一个函数来加载`meta`数据：

```python
import yaml
def load_meta(model_fname):
    ''' Load meta configuration
    '''
    meta = {}

    with h5py.File(model_fname, 'r') as f:
        meta_group = f['meta']

        meta['training_args'] = yaml.load(
            meta_group.attrs['training_args'])
        for k in meta_group.keys():
            meta[k] = list(meta_group[k])

    return meta
```

调用之：

```python
last_meta = load_meta("./results/pointnet.h5")
last_meta
```

```console
{'acc': [0.659, 0.878, 0.924, 0.957, 0.967, 0.985, 0.99, 0.992, 0.996],
 'epochs': [0, 1, 2, 3, 4, 5, 6, 7, 8],
 'loss': [1.158559440612793,
  0.4256118061542511,
  0.28586792707443237,
  0.2092902910709381,
  0.1517823133468628,
  0.11005254900455474,
  0.08378619635850192,
  0.06799231326580048,
  0.05295254367589951],
 'training_args': '{}',
 'val_acc': [0.777, 0.832, 0.862, 0.854, 0.865, 0.857, 0.87, 0.866, 0.875],
 'val_loss': [0.7094822826385498,
  0.5294894614219665,
  0.44566395616531373,
  0.454044997215271,
  0.426121289730072,
  0.42110520076751706,
  0.4039662191867828,
  0.4096675853729248,
  0.40012847566604615]}
```

可以轻易看出`epochs`只记录到第八次，因为我们`save_best_only`.

## 中断实验

手动删除`.h5`模型文件

然后回到上一次`fit`的时候

```python
model.fit(train_images, train_labels, epochs = 10,
          validation_data = (test_images,test_labels),
          callbacks = [checkpoint])
```

运行过程中强行结束：

```console
Train on 1000 samples, validate on 1000 samples
Epoch 1/10
1000/1000 [==============================] - 0s 445us/step - loss: 1.1578 - acc: 0.6580 - val_loss: 0.7134 - val_acc: 0.7900

Epoch 00001: val_acc improved from -inf to 0.79000, saving model to ./results/pointnet.h5
Epoch 2/10
 448/1000 [============>.................] - ETA: 0s - loss: 0.4623 - acc: 0.8862

KeyboardInterruptTraceback (most recent call last)
<ipython-input-23-e27c5787d098> in <module>
      1 model.fit(train_images, train_labels,  epochs = 10,
      2           validation_data = (test_images,test_labels),
----> 3           callbacks = [checkpoint])
。。。more ouput。。。
```

再执行

```python
last_meta = load_meta("./results/pointnet.h5")
last_meta
```

```console
{'acc': [0.658],
 'epochs': [0],
 'loss': [1.1578046548366547],
 'training_args': '{}',
 'val_acc': [0.79],
 'val_loss': [0.7134194440841675]}
```

果然，就记录到第1次输出的时候。

创建一个函数来加载关键数据：

```python
def get_last_status(model):
    last_epoch = -1
    last_meta = {}
    if os.path.exists("./results/pointnet.h5"):
        model.load_weights("./results/pointnet.h5")
        last_meta = load_meta("./results/pointnet.h5")
        last_epoch = last_meta.get('epochs')[-1]
    return last_epoch, last_meta
```

```python
last_epoch, last_meta = get_last_status(model)
last_epoch
```

```console
0
```

这个输出结果是正确的，我们只要从，第`last_epoch+1`次继续训练就好。`val_acc`或`val_loss`等都会被妥善在`on_train_begin`时处理好。

再次构建一个带有`meta`属性的回调函数：

```python
checkpoint = MetaCheckpoint('./results/pointnet.h5', monitor='val_acc',
                            save_weights_only=True, save_best_only=True,
                            verbose=1, meta=last_meta)
```

重新训练：（指定起始epoch）

```python
model.fit(train_images, train_labels,  epochs = 10,
          validation_data = (test_images,test_labels),
          callbacks = [checkpoint],
          initial_epoch=last_epoch+1)
```

```console
Train on 1000 samples, validate on 1000 samples
Epoch 2/10
1000/1000 [==============================] - 0s 308us/step - loss: 0.4211 - acc: 0.8840 - val_loss: 0.5157 - val_acc: 0.8450

Epoch 00002: val_acc improved from 0.79000 to 0.84500, saving model to ./results/pointnet.h5
Epoch 3/10
1000/1000 [==============================] - 0s 274us/step - loss: 0.2882 - acc: 0.9220 - val_loss: 0.5238 - val_acc: 0.8240

Epoch 00003: val_acc did not improve from 0.84500
Epoch 4/10
1000/1000 [==============================] - 0s 277us/step - loss: 0.2098 - acc: 0.9500 - val_loss: 0.4674 - val_acc: 0.8470

Epoch 00004: val_acc improved from 0.84500 to 0.84700, saving model to ./results/pointnet.h5
Epoch 5/10
1000/1000 [==============================] - 0s 275us/step - loss: 0.1503 - acc: 0.9680 - val_loss: 0.4215 - val_acc: 0.8640

Epoch 00005: val_acc improved from 0.84700 to 0.86400, saving model to ./results/pointnet.h5
Epoch 6/10
1000/1000 [==============================] - 0s 279us/step - loss: 0.1081 - acc: 0.9830 - val_loss: 0.4051 - val_acc: 0.8680

Epoch 00006: val_acc improved from 0.86400 to 0.86800, saving model to ./results/pointnet.h5
Epoch 7/10
1000/1000 [==============================] - 0s 272us/step - loss: 0.0761 - acc: 0.9920 - val_loss: 0.4186 - val_acc: 0.8650

Epoch 00007: val_acc did not improve from 0.86800
Epoch 8/10
1000/1000 [==============================] - 0s 269us/step - loss: 0.0558 - acc: 0.9970 - val_loss: 0.4088 - val_acc: 0.8700

Epoch 00008: val_acc improved from 0.86800 to 0.87000, saving model to ./results/pointnet.h5
Epoch 9/10
1000/1000 [==============================] - 0s 269us/step - loss: 0.0486 - acc: 0.9990 - val_loss: 0.3960 - val_acc: 0.8730

Epoch 00009: val_acc improved from 0.87000 to 0.87300, saving model to ./results/pointnet.h5
Epoch 10/10
1000/1000 [==============================] - 0s 266us/step - loss: 0.0354 - acc: 1.0000 - val_loss: 0.4058 - val_acc: 0.8700

Epoch 00010: val_acc did not improve from 0.87300
```

可以看见，确实是从第2个epoch开始训练的。

这！就是我们想要的效果。

# 博客地址

<https://tellyouwhat.cn/p/machine-learning-tensorflow-keras-how-to-gracefully-continue-training-from-where-it-drops/>