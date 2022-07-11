import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from mmrotate_infer import infer, load_model
import time
app = FastAPI()

model = load_model()

@app.get('/')
async def root():
    return {'msg': 'hello world'}

@app.post('/pen_detection/')
async def receive_from_cli(img: UploadFile = File(...)):
    orig_img = Image.open(img.file)
    image_arr = np.array(orig_img)
    start = time.time()
    time_infer = infer(model, image_arr)
    return {'msg': time_infer, 'total time': time.time()-start}