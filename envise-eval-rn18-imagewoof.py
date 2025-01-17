import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
from rn18_model import ResNet18
import numpy as np
from keras.callbacks import ModelCheckpoint
import logging
from pudb import set_trace

# idiom.ml imports
from idiom.ml.tf import (
    setup_for_evaluation,
    setup_for_tuning,
    setup_for_export
)
from idiom.ml.tf.recipe import IdiomRecipe

logger = tf.get_logger()
logger.setLevel(logging.INFO)
logger.info(f'TF version:{tf.__version__}, cuda version:{tf.sysconfig.get_build_info()["cuda_version"]}')

root_folder = '/home/auro/pyt-rn18-notebook'
val_folder = os.path.join(root_folder, "imagewoof2-320/val/")
image_size = (320, 320, 3)
target_size = (224, 224)
nb_classes = 10

imagegen = ImageDataGenerator(featurewise_center=True,
                              featurewise_std_normalization=True,
                              zoom_range=[0.8, 1.2],
                              rotation_range=20,
                              horizontal_flip=True,)
imagegen.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3)) # ordering: [R, G, B]
imagegen.std = 64.

val = imagegen.flow_from_directory(val_folder, class_mode="categorical",
                                   shuffle=False, batch_size=128, target_size=target_size)

imported_model = ResNet18(nb_classes)
# imported_model.build(input_shape = (1, 224, 224, 3))  
imported_model.build(input_shape = (1, 320, 320, 3))  
imported_model.summary()

# Perform quantized inference
# First Conv2D layer should be im2col
# determine the first layer name
first_conv_layer_name = None
for layer in imported_model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        first_conv_layer_name = layer.name
        break
else:  # no break
    raise RuntimeError('cannot determine first conv layer name')
recipe = IdiomRecipe(layer_names=[first_conv_layer_name])
recipe.update_capability(
    first_conv_layer_name, 'conv_algorithm', None, 'im2col'
)

set_trace()
# ERROR
quant_model = setup_for_evaluation(imported_model, finetuning_method="ept", recipe=recipe)

model = imported_model

model_path = os.path.join('checkpoint', 'rn18-best-epoch-39-acc-0.653.hdf5')
if os.path.exists(model_path):
    print(f'loading file:{model_path}')
    load_status = model.load_weights(model_path)
else:
    print(f'file {model-path} does not exist.')


    

# these are mechanics to make eval work (revisit)
sgd_optimizer = tf.keras.optimizers.SGD(
  learning_rate=0.1,
  momentum=0.9,
  nesterov=False,
  name='SGD',
)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd_optimizer,
              metrics=['accuracy'])


loss, accuracy = model.evaluate(val)
