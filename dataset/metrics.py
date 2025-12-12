import numpy as np

def box_iou(boxA, boxB):
    if boxA is None or boxB is None:
        return 0.0

    ax1 = boxA[0] - boxA[2]/2
    ay1 = boxA[1] - boxA[3]/2
    ax2 = boxA[0] + boxA[2]/2
    ay2 = boxA[1] + boxA[3]/2

    bx1 = boxB[0] - boxB[2]/2
    by1 = boxB[1] - boxB[3]/2
    bx2 = boxB[0] + boxB[2]/2
    by2 = boxB[1] + boxB[3]/2

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = (ax2-ax1)*(ay2-ay1)
    areaB = (bx2-bx1)*(by2-by1)

    if areaA == 0 or areaB == 0:
        return 0.0

    iou = inter_area / (areaA + areaB - inter_area)
    return float(iou)


def center_error(boxA, boxB):
    """L2 distance between centers."""
    if boxA is None or boxB is None:
        return 99999
    dx = boxA[0] - boxB[0]
    dy = boxA[1] - boxB[1]
    return float(np.sqrt(dx*dx + dy*dy))
