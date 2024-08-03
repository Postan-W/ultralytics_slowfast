import cv2,torch
import numpy as np
from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image, )
from torchvision.transforms._functional_video import normalize

class MyVideoCapture:

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.idx = -1
        self.end = False
        self.stack = []

    def read(self):
        self.idx += 1
        ret, img = self.cap.read()
        if ret:
            self.stack.append(img)
        else:
            self.end = True
        return ret, img

    def to_tensor(self, img):
        img = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img.unsqueeze(0)

    def get_video_clip(self):
        assert len(self.stack) > 0, "clip length must large than 0 !"
        self.stack = [self.to_tensor(img) for img in self.stack]
        clip = torch.cat(self.stack).permute(-1, 0, 1, 2)
        del self.stack
        self.stack = []
        return clip

    def release(self):
        self.cap.release()

def plot_one_box(x, img, color=[100, 100, 100], text_info="None",
                 velocity=None, thickness=1, fontsize=0.5, fontthickness=1):
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize, fontthickness + 2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1] * 1.45)), color, -1)
    cv2.putText(img, text_info, (c1[0], c1[1] + t_size[1] + 2),
                cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255, 255, 255], fontthickness)
    return img

def ava_inference_transform(
        clip,
        boxes,
        num_frames=32,  # if using slowfast_r50_detection, change this to 32, 4 for slow
        crop_size=640,
        data_mean=[0.45, 0.45, 0.45],
        data_std=[0.225, 0.225, 0.225],
        slow_fast_alpha=4,  # if using slowfast_r50_detection, change this to 4, None for slow
):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy()
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    clip, boxes = short_side_scale_with_boxes(clip, size=crop_size, boxes=boxes, )
    clip = normalize(clip,
                     np.array(data_mean, dtype=np.float32),
                     np.array(data_std, dtype=np.float32), )
    boxes = clip_boxes_to_image(boxes, clip.shape[2], clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip
        slow_pathway = torch.index_select(clip, 1,
                                          torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long())
        clip = [slow_pathway, fast_pathway]

    return clip, torch.from_numpy(boxes), roi_boxes

def ava_inference_transform_slow_r50(
    clip,
    boxes,
    num_frames = 4, #if using slowfast_r50_detection, change this to 32
    crop_size = 256,
    data_mean = [0.45, 0.45, 0.45],
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = None, #if using slowfast_r50_detection, change this to 4
):

    boxes = np.array(boxes)
    ori_boxes = boxes.copy()

    # Image [0, 255] -> [0, 1].
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0

    height, width = clip.shape[2], clip.shape[3]
    # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
    # range of [0, width] for x and [0,height] for y
    boxes = clip_boxes_to_image(boxes, height, width)

    # Resize short side to crop_size. Non-local and STRG uses 256.
    clip, boxes = short_side_scale_with_boxes(
        clip,
        size=crop_size,
        boxes=boxes,
    )

    # Normalize images by mean and std.
    clip = normalize(
        clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),
    )

    boxes = clip_boxes_to_image(
        boxes, clip.shape[2],  clip.shape[3]
    )

    # Incase of slowfast, generate both pathways
    if slow_fast_alpha is not None:
        fast_pathway = clip
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            clip,
            1,
            torch.linspace(
                0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha
            ).long(),
        )
        clip = [slow_pathway, fast_pathway]

    return clip,torch.from_numpy(boxes), ori_boxes

def yolopreds_filter(result, id_to_ava_labels,max_conf={}):
    result.final_boxes = []
    result.final_boxes_textes = []
    result.final_boxes_colors = []
    boxes = result.boxes.data.cpu().numpy().tolist()
    if len(boxes):
        for box in boxes:
            if len(box) == 7:  # 有追踪id
                if box[4] in id_to_ava_labels.keys():
                    if box[-1] == 0:
                        if (16 in id_to_ava_labels[box[4]]["action_index"]) and (
                                58 in id_to_ava_labels[box[4]]["action_index"]):
                            # if (9 in id_to_ava_labels[box[4]]["action_index"]) or (19 in id_to_ava_labels[box[4]]["action_index"]) or (6 in id_to_ava_labels[box[4]]["action_index"]):
                            text = "climb" + " conf:" + str(round(box[5], 2))
                            for i, name in enumerate(id_to_ava_labels[box[4]]["action_name"]):
                                text += " " + name + ":" + str(id_to_ava_labels[box[4]]["action_prob"][i])

                            color = [255, 0, 0]
                            result.final_boxes.append(box)
                            result.final_boxes_textes.append(text)
                            result.final_boxes_colors.append(color)
                            if box[4] in max_conf.keys():
                                if box[5] > max_conf[box[4]]:
                                    max_conf[box[4]] = box[5]
                            else:
                                max_conf[box[4]] = box[5]

                    elif box[-1] == 1:
                        if ((4 in id_to_ava_labels[box[4]]["action_index"]) and
                            id_to_ava_labels[box[4]]["action_prob"][
                                id_to_ava_labels[box[4]]["action_index"].index(4)] > 0.2) or (
                                (7 in id_to_ava_labels[box[4]]["action_index"]) and
                                id_to_ava_labels[box[4]]["action_prob"][
                                    id_to_ava_labels[box[4]]["action_index"].index(7)] > 0.2):
                            text = "fall" + " conf:" + str(round(box[5], 2))
                            for i, name in enumerate(id_to_ava_labels[box[4]]["action_name"]):
                                text += " " + name + ":" + str(id_to_ava_labels[box[4]]["action_prob"][i])

                            color = [0, 0, 255]
                            result.final_boxes.append(box)
                            result.final_boxes_textes.append(text)
                            result.final_boxes_colors.append(color)
                            if box[4] in max_conf.keys():
                                if box[5] > max_conf[box[4]]:
                                    max_conf[box[4]] = box[5]
                            else:
                                max_conf[box[4]] = box[5]

                else:
                    if box[5] > 0.94:
                        if box[-1] == 0:
                            color = [255, 0, 0]
                            text = "climb conf:{}".format(box[5])
                            result.final_boxes.append(box)
                            result.final_boxes_textes.append(text)
                            result.final_boxes_colors.append(color)
                        elif box[-1] == 1:
                            color = [0, 0, 255]
                            text = "fall conf:{}".format(box[5])
                            result.final_boxes.append(box)
                            result.final_boxes_textes.append(text)
                            result.final_boxes_colors.append(color)



def save_yolopreds_tovideo_yolov8_version(result, id_to_ava_labels, output_video):
        im = result.orig_img
        boxes = result.boxes.data.cpu().numpy().tolist()
        if len(boxes):
            for box in boxes:
                if len(box) == 7:  # 有追踪id
                    if box[4] in id_to_ava_labels.keys():
                        if box[-1] == 0:
                            if (16 in id_to_ava_labels[box[4]]["action_index"]) and (58 in id_to_ava_labels[box[4]]["action_index"]):
                                # if (9 in id_to_ava_labels[box[4]]["action_index"]) or (19 in id_to_ava_labels[box[4]]["action_index"]) or (6 in id_to_ava_labels[box[4]]["action_index"]):
                                    text = "climb" + " conf:" + str(round(box[5],2))
                                    for i, name in enumerate(id_to_ava_labels[box[4]]["action_name"]):
                                        text += " " + name + ":" + str(id_to_ava_labels[box[4]]["action_prob"][i])

                                    color = [255, 0, 0]
                                    im = plot_one_box(box, im, color, text)
                            # else:
                            #     text = "yolo targets filtered by slowfast"
                            #     color = [0, 255, 0]
                            #     im = plot_one_box(box, im, color, text)
                        elif box[-1] == 1:
                            if ((4 in id_to_ava_labels[box[4]]["action_index"]) and
                                id_to_ava_labels[box[4]]["action_prob"][
                                    id_to_ava_labels[box[4]]["action_index"].index(4)] > 0.2) or (
                                    (7 in id_to_ava_labels[box[4]]["action_index"]) and
                                    id_to_ava_labels[box[4]]["action_prob"][
                                        id_to_ava_labels[box[4]]["action_index"].index(7)] > 0.2):
                                text = "fall" + " conf:" + str(round(box[5],2))
                                for i, name in enumerate(id_to_ava_labels[box[4]]["action_name"]):
                                    text += " " + name + ":" + str(id_to_ava_labels[box[4]]["action_prob"][i])

                                color = [0, 0, 255]
                                im = plot_one_box(box, im, color, text)
                            # else:
                            #     text = "yolo targets filtered by slowfast"
                            #     color = [0, 255, 0]
                            #     im = plot_one_box(box, im, color, text)

        im = im.astype(np.uint8)
        output_video.write(im)


def save_yolopreds_tovideo_yolov8_version_origin(result, id_to_ava_labels, output_video):
    im = result.orig_img
    boxes = result.boxes.data.cpu().numpy().tolist()
    if len(boxes):
        for box in boxes:
            if len(box) == 7:  # 有追踪id
                if box[4] in id_to_ava_labels.keys():
                    ava_label = id_to_ava_labels[box[4]]
                else:
                    ava_label = 'Unknow'
                text = '{} {}'.format(int(box[4]), ava_label)
                color = [15, 76, 243]
                im = plot_one_box(box, im, color, text)
            else:  # 此时的box[4]是conf了
                ava_label = 'Unknow'
                text = 'conf:{} {}'.format(box[4], ava_label)
                color = [15, 76, 243]
                im = plot_one_box(box, im, color, text)

    im = im.astype(np.uint8)
    output_video.write(im)

