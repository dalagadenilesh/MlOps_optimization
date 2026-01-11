from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
from PIL import Image
import io
import os
import torch
# import torch_tensorrt as torchtrt
import torchvision.transforms as transforms
import asyncio
#from azure.storage.blob import BlobServiceClient
from concurrent.futures import ThreadPoolExecutor
import base64
from pydantic import BaseModel, Field, Base64Bytes, field_validator
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function
import glob 
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import onnx 
import onnxruntime as rt 
import numpy as np
import time
from typing import Optional
from fastapi.responses import JSONResponse


app = FastAPI()

def check_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
device = check_device()

sess_options = rt.SessionOptions()
sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.inter_op_num_threads = 1
sess_options.intra_op_num_threads = 8
sess_options.enable_profiling = False

sess_options.enable_mem_pattern = True
sess_options.enable_mem_reuse = True


sess_options.add_session_config_entry("sess_options.use_env_alloc", "1")
sess_options.add_session_config_entry("sess_options.disable_prepacking", "0")

sess_options.add_session_config_entry("session.intra_op.allow_spinning", "0")
sess_options.add_session_config_entry("session.set_denormal_as_zero", "1")


sess = rt.InferenceSession("/home/nileshdalagade/tutorial/modified.onnx", sess_options = sess_options, providers = ['CPUExecutionProvider'])
io_binding = sess.io_binding()


# ------------------------
# 2️⃣ Preprocessing setup
# ------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225]),
])

# ------------------------
# 3️⃣ Single image inference
# ------------------------
class Metadata(BaseModel):
    file_name: Optional[str] = Field(None, description = 'Image identity file name for user reference', example = 'tutorial/n01496331_12286.JPEG')

class ImageRequest(BaseModel):
    image_b64: Base64Bytes = Field(... , description = "base64.b64encode Image data.")
    metadata: Optional[Metadata] = None

    @field_validator('image_b64', mode = 'after')
    @classmethod
    def validate_base64_image(cls, value) -> str:
        image_stream = io.BytesIO(value)
        img = Image.open(image_stream)
        img.verify()
        img.format in ['JPEG', 'PNG']
        assert img.mode == 'RGB'
        return value

class BatchRequest(BaseModel):
    images: List[ImageRequest]

class PredictIdentity(BaseModel):
    file_name: str = Field(..., description = 'Image identifier. this will be same as file_name provided in request', example = 'img1.jpg or file_name in request')
    predicted_class: int = Field(..., description = 'Model prediction class')

class BatchResponse(BaseModel):
    prediction: List[PredictIdentity]
    batchsize: int = Field(..., description = 'batch size of predicted images. this will be useful for batch prediction')


@app.post("/predict", response_model = PredictIdentity)
async def predict(request: ImageRequest):
    try:
        pil_image = Image.open(io.BytesIO(request.image_b64)).convert('RGB')
        tensor_input = preprocess(pil_image)
        tensor_input = tensor_input.unsqueeze(0).to(device, dtype = torch.float32, non_blocking = True)
        
        io_binding.bind_input(
            name = sess.get_inputs()[0].name,
            device_type = device,
            device_id = 0,
            element_type = np.float32,
            shape = tuple(tensor_input.shape),
            buffer_ptr = tensor_input.data_ptr()
            )
        out = torch.empty(1, dtype = torch.int64, device = device)

        # case 1: this method requires io_binding.copy_outputs_to_cpu()
        # io_binding.bind_output(name = sess.get_outputs()[0].name, device_type= 'cpu')

        io_binding.bind_output(
            name = sess.get_outputs()[0].name,
            device_type = 'cpu',
            buffer_ptr = out.data_ptr(),
            shape = tuple(out.shape),
            element_type = np.int64)
        
        sess.run_with_iobinding(io_binding)
        # case 1: Y = io_binding.copy_outputs_to_cpu()[0]
        return {'file_name': request.metadata.file_name if request.metadata else 'img1', 'predicted_class': out.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code = 404, detail = f"{e}")



@app.post("/predict_batch", response_model = BatchResponse)
async def predict_batch(request: BatchRequest):
    try:
        
        lst = [Image.open(io.BytesIO(i.image_b64)).convert('RGB') for i in request.images]
        tensors = [preprocess(i) for i in lst]
    
        results = torch.stack(tuple(tensors), dim = 0).to(device, dtype = torch.float32, non_blocking = True)
        io_binding.bind_input(
            name = sess.get_inputs()[0].name,
            device_type = device,
            device_id = 0,
            element_type = np.float32,
            shape = tuple(results.shape),
            buffer_ptr = results.data_ptr())
        
        out = torch.empty(results.shape[0], dtype = torch.int64, device = device)

        io_binding.bind_output(
            name = sess.get_outputs()[0].name,
            device_type = 'cpu',
            buffer_ptr = out.data_ptr(),
            shape = tuple(out.shape),
            element_type = np.int64)

        sess.run_with_iobinding(io_binding)
        pred = out.tolist()
        
        return {'prediction':[{'file_name': item.metadata.file_name if item.metadata else f'img{num}', 'predicted_class': pred[num]} for num, item in enumerate(request.images)], "batchsize": len(request.images)}

    except Exception as e:
        raise HTTPException(status_code = 404, detail = f"{e}")