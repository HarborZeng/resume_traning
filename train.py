from keras.callbacks import Callback, ModelCheckpoint
import yaml
import h5py
import numpy as np
import keras

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

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
model = create_model()

import os
if not os.path.exists('./results/'):
    os.mkdir('./results/')

class MetaCheckpoint(ModelCheckpoint):
    """
    Checkpoints some training information with the model. This should enable
    resuming training and having training information on every checkpoint.
    Thanks to Roberto Estevao @robertomest - robertomest@poli.ufrj.br
    """

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
        super(MetaCheckpoint, self).on_epoch_end(epoch, logs)

        # Get statistics
        self.meta['epochs'].append(epoch)
        for k, v in logs.items():
            # Get default gets the value or sets (and gets) the default value
            self.meta.setdefault(k, []).append(v)

        # Save to file
        filepath = self.filepath.format(epoch=epoch, **logs)

        if logs.get(self.monitor) == self.best and self.epochs_since_last_save == 0:
            with h5py.File(filepath, 'r+') as f:
                meta_group = f.create_group('meta')
                meta_group.attrs['training_args'] = yaml.dump(
                    self.meta.get('training_args', '{}'))
                meta_group.create_dataset('epochs', data=np.array(self.meta['epochs']))
                for k in logs:
                    meta_group.create_dataset(k, data=np.array(self.meta[k]))


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

def get_last_status(model):
    last_epoch = -1
    last_meta = {}
    if os.path.exists("./results/pointnet.h5"):
        model.load_weights("./results/pointnet.h5")
        last_meta = load_meta("./results/pointnet.h5")
        last_epoch = last_meta.get('epochs')[-1]
    return last_epoch, last_meta

last_epoch, last_meta = get_last_status(model)

checkpoint = MetaCheckpoint('./results/pointnet.h5', monitor='val_acc',
                            save_weights_only=True, save_best_only=True,
                            verbose=1, meta=last_meta)

model.fit(train_images, train_labels,  epochs = 10,
          validation_data = (test_images,test_labels),
          callbacks = [checkpoint],
          initial_epoch=last_epoch+1)