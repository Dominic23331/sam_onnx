import time
import argparse

import cv2
import numpy as np

from sam.predictor import SamPredictor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, help="Path to either a single input image or folder of images.")
    parser.add_argument("--vit-model", type=str, default="model/vit_b.onnx", help="vit model path.")
    parser.add_argument("--decoder-model", type=str, default="model/sam_vit_b.onnx", help="decoder model path.")
    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup epoch. if set 0, model won`t warmup.")
    parser.add_argument("--output", type=str, default="", help="Path to the directory where masks will be output.")
    args = parser.parse_args()
    return args


args = get_args()

predictor = SamPredictor(args.vit_model,
                         args.decoder_model,
                         args.device,
                         args.warmup)

img = cv2.imread(args.img)

start = time.time()
predictor.register_image(img)
end = time.time()
print("The encoder waist time: {:.3f}".format(end - start))

points = []
boxes = []
box_point = []
im0 = img.copy()
get_first_box_point = False


def change_box(box):
    x1 = min(box[0], box[2])
    y1 = min(box[1], box[3])
    x2 = max(box[0], box[2])
    y2 = max(box[1], box[3])
    return [x1, y1, x2, y2]


def draw_circle(event, x, y, flags, param):
    global img, get_first_box_point, box_point
    if event == cv2.EVENT_LBUTTONDOWN and not get_first_box_point:
        print("Add point:", (x, y))
        points.append([x, y])

        mask = predictor.get_mask(points,
                                  [1 for i in range(len(points))],
                                  boxes if boxes != [] else None)

        mask = mask["masks"][0][0][:, :, None]
        cover = np.ones_like(img) * 255
        cover = cover * mask
        cover = np.uint8(cover)
        img = cv2.addWeighted(im0, 0.6, cover, 0.4, 0)

        for b in boxes:
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
            cv2.circle(img, (b[0], b[1]), 5, (0, 0, 255), -1)
        for p in points:
            cv2.circle(img, (p[0], p[1]), 5, (255, 0, 0), -1)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if not get_first_box_point:
            box_point.append(x)
            box_point.append(y)
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            get_first_box_point = True
        else:
            box_point.append(x)
            box_point.append(y)
            box_point = change_box(box_point)
            boxes.append(box_point)
            print("add box:", box_point)

            mask = predictor.get_mask(points if points != [] else None,
                                      [1 for i in range(len(points))] if points != [] else None,
                                      boxes)

            mask = mask["masks"][0][0][:, :, None]
            cover = np.ones_like(img) * 255
            cover = cover * mask
            cover = np.uint8(cover)
            img = cv2.addWeighted(im0, 0.6, cover, 0.4, 0)

            get_first_box_point = False
            box_point = []

        for b in boxes:
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
            cv2.circle(img, (b[0], b[1]), 5, (0, 0, 255), -1)
            cv2.circle(img, (b[2], b[3]), 5, (0, 0, 255), -1)
        for p in points:
            cv2.circle(img, (p[0], p[1]), 5, (255, 0, 0), -1)

    cv2.imshow("image", img)


cv2.namedWindow("image")
cv2.setMouseCallback('image', draw_circle)
key = cv2.waitKey()
if key & 0xff == ord("s"):
    if args.output != "":
        cv2.imwrite(args.output, img)
cv2.destroyAllWindows()
