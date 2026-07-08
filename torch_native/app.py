from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from typing import List
from PIL import Image
import io
import os
import glob 
import torch
import uuid
# import torch_tensorrt as torchtrt
import torchvision.transforms as transforms
import asyncio
from concurrent.futures import ThreadPoolExecutor
import base64
from pydantic import BaseModel, Field, field_validator, Base64Bytes
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
from fastapi.responses import JSONResponse
from typing import Optional
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import logging
import uvicorn
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('app.log', mode='a', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

MAX_QUEUE_SIZE = 200
MAX_BATCH_SIZE = 8
MAX_WAIT_MS = 1000
MAX_BATCH_WORKER = 3
SEMAPHORE_TASKS = 2
REQUEST_TIMEOUT_SECONDS = 10

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

_queue = asyncio.Queue(maxsize = MAX_QUEUE_SIZE)
_semaphore = asyncio.Semaphore(value = SEMAPHORE_TASKS)
executor = ThreadPoolExecutor(max_workers = 2)
batch_worker_tasks: list[asyncio.Task] = []

def log_task_exception(task: asyncio.Task):
    try:
        if not task.cancelled():
            task.result()
    except Exception as e:
        print('background task err', e)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global batch_worker_tasks

    for worker_id in range(MAX_BATCH_WORKER):
        task = asyncio.create_task(batch_worker(worker_id))
        task.add_done_callback(log_task_exception)
        batch_worker_tasks.append(task)

    yield

    for task in batch_worker_tasks:
        task.cancel()

    for task in batch_worker_tasks:
        try:
            await task
        except asyncio.CancelledError:
            pass

app = FastAPI(lifespan = lifespan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(weights=DenseNet121_Weights.DEFAULT).to(device = device, non_blocking = True)
model = model.to(memory_format = torch.channels_last)

class ImageRequest(BaseModel):
    image_b64: Base64Bytes = Field(... , description = "base64.b64encode Image data.")

    # @field_validator('image_b64', mode = 'after')
    # @classmethod
    # def validate_base64_image(cls, value) -> str:
    #     image_stream = io.BytesIO(value)
    #     img = Image.open(image_stream)
    #     img.verify()
    #     img.format in ['JPEG', 'PNG']
    #     assert img.mode == 'RGB'
    #     return value

def preprocess(lst):
    out = []
    TORCH_TRANSFORM = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225]),
        ])
    for by in lst:
        img = Image.open(io.BytesIO(by)).convert('RGB')
        out.append(TORCH_TRANSFORM(img).unsqueeze(0))
    return torch.cat(out, dim = 0)

def model_prediction(input):
    with torch.no_grad():
        output =  model(input).to(device = 'cpu', non_blocking = True)
    pred = torch.argmax(torch.softmax(output, dim = -1), dim = -1).tolist()
    return pred

async def collect_batch(worker_id):
    loop = asyncio.get_event_loop()
    batch_items = []
    item = await _queue.get()
    batch_items.append(item)

    st = loop.time()

    while len(batch_items) < MAX_BATCH_SIZE:
        remaining_time = (MAX_WAIT_MS/1000) - (loop.time() - st)
        if float(remaining_time) <= 0.0:
            break
        try:
            item = await asyncio.wait_for(_queue.get(), timeout = remaining_time)
            batch_items.append(item)
        except asyncio.TimeoutError:
            break
    
    return batch_items

async def batch_worker(worker_id):
    loop = asyncio.get_running_loop()
    while True:
        batch_items = await collect_batch(worker_id = worker_id)
        batch_inputs = [i.input_data for i in batch_items]
        batch_inputs = await loop.run_in_executor(executor, preprocess, batch_inputs)
        
        try:
            async with _semaphore:
                predictions = await loop.run_in_executor(executor, model_prediction, batch_inputs)
            
            for item, prediction in zip(batch_items, predictions):
                if not item.future.cancelled():
                    item.future.set_result(prediction)

        except Exception as e:
            for item in batch_items:
                if not item.future.cancelled():
                    item.future.set_exception(str(e))

        finally:
            if 'batch_items' in locals():
                for _ in batch_items:
                    _queue.task_done()

@dataclass
class Queueitem:
    input_data: Base64Bytes
    future: asyncio.Future

@app.post("/predict")
async def predict(request: ImageRequest):
    try:

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await _queue.put(Queueitem(input_data = request.image_b64, future = future))

        try:
            prediction = await asyncio.wait_for(future, timeout = REQUEST_TIMEOUT_SECONDS)
            return prediction
        
        except asyncio.TimeoutError:
            future.cancel()
            raise HTTPException(status_code = 504, detail = 'Prediction Time Out')
        
    except Exception as e:
        raise HTTPException(status_code = 404, detail = f"{e}")
    

@app.middleware("http")
async def log_and_time_requests(request: Request, call_next):

    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    request.state.request_id = request_id
    response = await call_next(request)

    process_time = time.perf_counter() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = f"{process_time * 1000:.2f}"
    
    print(f"ID: {request_id} | Path: {request.url.path} | Time: {process_time*1000:.2f}ms")
    return response

if __name__=='__main__':
    uvicorn.run("app:app", host = "127.0.0.1", port = 8000, reload = True)
