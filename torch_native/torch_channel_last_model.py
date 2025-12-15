from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import List
from PIL import Image
import io
import os
import torch
# import torch_tensorrt as torchtrt
import torchvision.transforms as transforms
import asyncio
from concurrent.futures import ThreadPoolExecutor
import base64
from pydantic import BaseModel, Field, field_validator, Base64Bytes
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function
import glob 
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
from fastapi.responses import JSONResponse
from typing import Optional

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(weights=DenseNet121_Weights.DEFAULT).to(device = device)
model = models.to(memory_format = torch.channels_last)


# ------------------------
# 2️⃣ Preprocessing setup
# ------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225]),
])


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




# ------------------------
# 3️⃣ Single image inference
# ------------------------

@app.post("/predict", response_model = PredictIdentity)
async def predict(request: ImageRequest):
    try:
        pil_image = Image.open(io.BytesIO(request.image_b64)).convert('RGB')
        tensor_input = preprocess(pil_image)
        tensor = tensor_input.unsqueeze(0).to(device, dtype = torch.float32, memory_format = torch.channels_last, non_blocking = True)

        with torch.no_grad():
            output = model(tensor).to('cpu', non_blocking = True)
        pred = torch.argmax(torch.softmax(output, dim = -1)).item()
        return {'file_name': request.metadata.file_name if request.metadata else 'img1', 'predicted_class': pred}
    except Exception as e:
        raise HTTPException(status_code = 404, detail = f"{e}")


# ------------------------
# 3️⃣ Single image inference
# ------------------------



@app.post("/predict_batch", response_model = BatchResponse)
async def predict_batch(request: BatchRequest):
    try:
        lst = [Image.open(io.BytesIO(i.image_b64)).convert('RGB') for i in request.images]
        tensors = [preprocess(i) for i in lst]
        results = torch.stack(tuple(tensors), dim = 0).to(device, dtype = torch.float32, memory_format = torch.channels_last, non_blocking = True)
    
        with profile(
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory = True,
            record_shapes = False,
            on_trace_ready = torch.profiler.tensorboard_trace_handler("/profiler/"),
            with_stack = True
        ) as prof:
            with record_function("model_inference"):
                with torch.no_grad():
                    output = model(results)
        
        pred = torch.argmax(torch.softmax(output.to('cpu', non_blocking = True), dim = -1), dim = -1).tolist()
        return {'prediction':[{'file_name': item.metadata.file_name if item.metadata else f'img{num}', 'predicted_class': pred[num]} for num, item in enumerate(request.images)], "batchsize": len(request.images)}
    
    except Exception as e:
        raise HTTPException(status_code = 404, detail = f"{e}")


