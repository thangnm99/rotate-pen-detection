import torch, torchvision
import mmrotate
import mmdet
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmcv
from mmcv.runner import load_checkpoint
import time
from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from mmrotate.utils import compat_cfg, setup_multi_processes

def load_model():
    config = 'mmrotate/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py'
    checkpoint = 'saved_model/epoch_10 (1).pth'
    device = 'cpu'
    config = mmcv.Config.fromfile(config)
    setup_multi_processes(config)

    num_classes = 1
    config.model.roi_head.bbox_head.num_classes = num_classes
    config.model.pretrained = None

    model = build_detector(config.model)
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)
    model.CLASSES = checkpoint['meta']['CLASSES']

    model.cfg = config
    model.to(device)
    model.eval()
    return model

def infer(model, image):
    '''
    Load model and inference an image.
    :param image: could be path or ndarray
    :return: time for inference
    '''

    start = time.time()
    result = inference_detector(model, image)
    model.show_result(image[...,::-1], result, score_thr=0.3, out_file='image/result.jpeg')
    return time.time() - start

if __name__ == '__main__':
    path = 'image/pen_detection.jpg'
    infer(path)
