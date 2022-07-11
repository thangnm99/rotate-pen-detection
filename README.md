# rotate-pen-detection
Detect the pen with oriented bounding box
# Installation
```buildoutcfg
# install dependencies: (use cu111 because colab has CUDA 11.1) no need 
!pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# install mmcv-full thus we could use CUDA operators
!pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# Install mmdetection
!pip install mmdet

# Install mmrotate
!rm -rf mmrotate
!git clone https://github.com/open-mmlab/mmrotate.git
%cd mmrotate
!pip install -e .

#Install fast api
pip install fastapi
pip install "uvicorn[standard]"
```
# Training notebook link
## Note book
https://colab.research.google.com/drive/16fuiXJ66wKha6PyPplFT6kKzI9vupIcC?authuser=1#scrollTo=4E3b4K7sixIt
## Trained model for pen detection with mmrotate
the link is in the txt file in saved_model directory

# Run code
uvicorn fast_api:app --host 0.0.0.0 --port 80