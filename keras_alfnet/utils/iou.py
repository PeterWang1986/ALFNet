#!/usr/bin/env python

import numpy as np

def iou(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    ious = []
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    b_xmin = boxes[:, 0]
    b_ymin = boxes[:, 1]
    b_xmax = boxes[:, 2]
    b_ymax = boxes[:, 3]
    boxes_area = (b_xmax - b_xmin) * (b_ymax - b_ymin)
    for i in range(K):
        query_box = query_boxes[i, :]
        q_xmin = query_box[0]
        q_ymin = query_box[1]
        q_xmax = query_box[2]
        q_ymax = query_box[3]
        q_area = (q_xmax - q_xmin) * (q_ymax - q_ymin)
        iw = np.minimum(b_xmax, q_xmax) - np.maximum(b_xmin, q_xmin)
        ih = np.minimum(b_ymax, q_ymax) - np.maximum(b_ymin, q_ymin)
        iw = np.array([v if v > 0 else 0.0 for v in iw])
        ih = np.array([v if v > 0 else 0.0 for v in ih])
        iarea = iw * ih
        iou = iarea / (boxes_area + q_area - iarea)
        ious.append(iou)
    print('call iou with N: %s, K: %s' % (N, K)) # test

    return np.array(ious).T


if __name__ == '__main__':
    boxes = np.array([[0, 0, 2, 2], [1, 1, 4, 5], [0, 1, 5, 7], [1, 0, 5, 5]])
    query_boxes = np.array([[0, 1, 3, 3], [2, 2, 4 ,5], [4, 5, 2, 3]])
    print(boxes)
    print(query_boxes)
    print('=' * 100)
    ious = iou(boxes.astype(np.float), query_boxes.astype(np.float))
    print('iou.shape: ', ious.shape)
    print(ious)
