
<code>
__________________________________________________________________________________________________
245/245 [==============================] - 12s 38ms/step - loss: 1.3797 - accuracy: 0.7395
2022-08-04 14:54:09.071717: I tensorflow/compiler/xla/service/service.cc:171] XLA service 0x562feb4012c0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2022-08-04 14:54:09.071759: I tensorflow/compiler/xla/service/service.cc:179]   StreamExecutor device (0): A100-PCIE-40GB, Compute Capability 8.0
2022-08-04 14:54:09.203300: I tensorflow/compiler/jit/xla_compilation_cache.cc:351] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
WARNING:tensorflow:5 out of the last 9 calls to <function NonIdiomLayerWrapper._forward_jit at 0x7f26290bf160> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
WARNING:tensorflow:6 out of the last 11 calls to <function NonIdiomLayerWrapper._forward_jit at 0x7f2628fbd3a0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
INFO:tensorflow:Show Envise eval  accuracy
245/245 [==============================] - 126s 195ms/step - loss: 1.4404 - accuracy: 0.7362
INFO:tensorflow:Done setting up for fine-tuning. Starting fine-tuning...
loading file:/home/auro/idiom-ml-tf/rn18/checkpoint/rn18-best-epoch-43-acc-0.7.hdf5
Found 9024 images belonging to 10 classes.
564/564 [==============================] - 205s 106ms/step - loss: 0.9830 - accuracy: 0.8525
INFO:tensorflow:Evaluating on validation data...
245/245 [==============================] - 81s 66ms/step - loss: 1.2382 - accuracy: 0.8028

</code>