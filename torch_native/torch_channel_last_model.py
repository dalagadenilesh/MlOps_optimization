from fastapi import FastAPI, UploadFile, File, Form
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
from pydantic import BaseModel
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function
import glob 
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.models as models
from torchvision.models import DenseNet121_Weights


app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(weights=DenseNet121_Weights.DEFAULT).to(device = device)
model = models.to(memory_format = torch.channels_last)


# ------------------------
# 2️⃣ Preprocessing setup
# ------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


executor = ThreadPoolExecutor(max_workers=4)  # for CPU preprocessing

async def preprocess_image(content):
    img = Image.open(io.BytesIO(content)).convert("RGB")
    loop = asyncio.get_running_loop()
    tensor = await loop.run_in_executor(executor, lambda: preprocess(img))
    return tensor



class ImageRequest(BaseModel):
    image_b64: str
    metadata: dict

class BatchRequest(BaseModel):
    images: List[ImageRequest]

async def decode_base64(b64_str: str) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, base64.b64decode, b64_str)


async def process_one(img: ImageRequest):
    img_byte = await decode_base64(img.image_b64)
    img_data = Image.open(io.BytesIO(img_byte)).convert("RGB")
    return img_data



# ------------------------
# 3️⃣ Single image inference
# ------------------------

@app.post("/predict")
async def predict(request: ImageRequest):
    img = await process_one(request)
    loop = asyncio.get_running_loop()
    tensor = await loop.run_in_executor(executor, lambda: preprocess(img))
    tensor = tensor.unsqueeze(0).to(device, dtype = torch.float32, memory_format = torch.channels_last, non_blocking = True)
    
    with torch.no_grad():
        output = model(tensor).to('cpu', non_blocking = True)
    pred = torch.argmax(torch.softmax(output, dim = -1)).item()
    return {'file': request.metadata['file_name'], 'predicted_class': pred}



# ------------------------
# 3️⃣ Single image inference
# ------------------------


@app.post("/predict_batch")
async def predict_batch(request: BatchRequest):
    metadata = [i.metadata for i in request.images]
    results = await asyncio.gather(*(process_one(item) for item in request.images))
    
    async def transform_image(img):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, lambda: preprocess(img))
    
    tensors = await asyncio.gather(*(transform_image(item) for item in results))
    results = torch.stack(tensors).to(device, dtype = torch.float32, memory_format = torch.channels_last, non_blocking = True)
    
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

    return {"batch_size": len(metadata), 'predictions':[{'file': item['file_name'], 'predicted_class': pred[num]} for num, item in enumerate(metadata)]}

