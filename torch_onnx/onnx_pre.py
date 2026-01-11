

import onnx
import torch
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
import torch
import torchvision
from torchvision import transforms
from PIL import Image




model = models.densenet121(weights=DenseNet121_Weights.DEFAULT).eval()


softmax_node = onnx.helper.make_node('Softmax',
                  inputs = ['logits'],
                    outputs = ['soft_output'])


argmax_node = onnx.helper.make_node('ArgMax',
    inputs = ['soft_output'],
    outputs = ['final_out'],
    axis = 1,
    keepdims = 0)


# Extend onnx graph for required outputs
model = onnx.load('/model_folder/first.onnx')
model.graph.node.extend([softmax_node, argmax_node])

model.graph.output.extend([
    onnx.helper.make_tensor_value_info('soft_output',onnx.TensorProto.FLOAT, [None, None]),
    onnx.helper.make_tensor_value_info('final_out', onnx.TensorProto.INT64, [None])
])

del model.graph.output[0]
del model.graph.output[0]

onnx.save_model(model, '/model_folder/first.onnx')