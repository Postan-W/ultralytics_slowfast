#author: wmingzhu
#date: 2024/7/16
import random,warnings, argparse
warnings.filterwarnings("ignore", category=UserWarning)
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from ultralytics import YOLO
from utils.myutil import *
import glob
import os
def main(config):
    device = "cuda"
    model = YOLO("./weights/yolov8l.engine")
    video_model = slowfast_r50_detection(True).eval().to(device)
    ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("utils/ava_action_list.pbtxt")
    print(config.input)
    cap = cv2.VideoCapture(config.input)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    slowfast_stack = []
    ret, frame = cap.read()
    height, width, _ = frame.shape
    slowfast_stack.append(frame)
    output_video = cv2.VideoWriter(config.output,cv2.VideoWriter_fourcc(*'X264'), 30, (width, height))
    processed_count = 0
    id_to_ava_labels = {}

    while ret:
        #待做，这里使用关键点检测模型。关键点缺失较多的直接pass。因为测试发现了一个上半身的人也分配了动作
        result = model.track(source=frame,verbose=False,persist=True,tracker="./track_config/botsort.yaml",classes=[0],conf=0.7,iou=0.7)[0]
        boxes = result.boxes.data.cpu().numpy()
        if len(slowfast_stack) == 15:
            slowfast_stack = [torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0) for frame in slowfast_stack]
            clip = torch.cat(slowfast_stack).permute(-1, 0, 1, 2)
            slowfast_stack = []
            if boxes.shape[0]:
                # 低于一定置信度的box，追踪算法不为其分配id，所以这里做一下筛选。筛选后要判断一下是否为空
                boxes_with_id = np.array([box for box in boxes.tolist() if len(box) == 7])#[x1,x2,y1,y2,trackid,conf,cls]
                if not len(boxes_with_id):
                    print("所有的box都没有id")
                    continue

                inputs, inp_boxes, _ = ava_inference_transform(clip,boxes_with_id[:, 0:4])
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1),inp_boxes],dim=1).float()#转成和inputs一样的类型
                if isinstance(inputs, list):
                    inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                else:
                    inputs = inputs.unsqueeze(0).to(device)
                with torch.no_grad():
                    slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                    slowfaster_preds = slowfaster_preds.cpu()

                result_labels = []
                #为每个id选取概率最大的5个动作
                for id_pres in slowfaster_preds:
                    result_labels.append(np.argsort(-id_pres).tolist()[:5])
                # print(result_labels)#[[11, 16, 78, 10, 47]]
                for i,(tid, avalabel) in enumerate(zip(boxes_with_id[:, 4].tolist(), result_labels)):
                    action_text = ""
                    if (7 in avalabel) or (4 in avalabel):
                        action_text = "fall"
                    else:
                        action_text = "not fall"
                    id_to_ava_labels[tid] = action_text

        processed_count += 1
        print("{}/{},{}%".format(processed_count, total_frames, round((processed_count / total_frames) * 100, 2)))
        ret, frame = cap.read()
        slowfast_stack.append(frame)
        #这意味着第一帧给每个id分配的动作类型，默认也分配给剩下n-1帧的相同id(n是clip包含的帧数)
        save_yolopreds_tovideo_yolov8_version(result, id_to_ava_labels,output_video)


    cap.release()
    output_video.release()
    print('saved video to:',config.output)


if __name__ == "__main__":
    videos = glob.glob("./videos/input/*")
    print(videos)
    for video in videos:
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', type=str, default=video,
                            help='test imgs folder or video or camera')
        output = os.path.splitext(os.path.split(video)[1])[0]+".mp4"
        parser.add_argument('--output', type=str, default="./videos/output/{}".format(output),
                            help='folder to save result imgs, can not use input folder')

        config = parser.parse_args()
        main(config)
