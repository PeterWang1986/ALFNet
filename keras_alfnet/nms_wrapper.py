# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# from .nms.gpu_nms import gpu_nms
# from .nms.cpu_nms import cpu_nms
from .nms.py_cpu_nms import py_cpu_nms

def nms(dets, thresh, usegpu, gpu_id):
    """Dispatch to either CPU or GPU NMS implementations."""
    """
    if dets.shape[0] == 0:
        return []
    if usegpu:
        return gpu_nms(dets, thresh, device_id=gpu_id)
    else:
        return cpu_nms(dets, thresh)
    """
    if dets.shape[0] == 0:
        return []
    else:
        return py_cpu_nms(dets, thresh)
