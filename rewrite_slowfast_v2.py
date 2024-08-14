#author: wmingzhu
#date: 2024/7/16
import warnings, argparse
warnings.filterwarnings("ignore", category=UserWarning)
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection,slow_r50_detection
from ultralytics import YOLO
from utils.myutil import *
import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
def main(config):
    device = "cuda"
    model = YOLO("./weights/climb_fall_20240812.engine")
    video_model = slow_r50_detection(True).eval().to(device)
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
    results = []
    stack_length = 15
    f = open("results.txt","a",encoding='utf-8')
    while ret:
        result = model.track(source=frame,verbose=False,imgsz=1280,persist=True,tracker="./track_config/botsort.yaml",classes=[0,1],conf=0.6,iou=0.7)[0]
        results.append(result)
        boxes = result.boxes.data.cpu().numpy()
        if len(slowfast_stack) == stack_length:
            slowfast_stack = [torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0) for frame in slowfast_stack]
            clip = torch.cat(slowfast_stack).permute(-1, 0, 1, 2)
            slowfast_stack = []
            if boxes.shape[0]:
                # 低于一定置信度的box，追踪算法不为其分配id，所以这里做一下筛选。筛选后要判断一下是否为空
                boxes_with_id = np.array([box for box in boxes.tolist() if len(box) == 7])#[x1,x2,y1,y2,trackid,conf,cls]
                if not len(boxes_with_id):
                    print("所有的box都没有id")
                    for r in results:  #每n帧共用一个动作类型
                        save_yolopreds_tovideo_yolov8_version(r, id_to_ava_labels, output_video)
                    results = []
                    continue

                inputs, inp_boxes, _ = ava_inference_transform_slow_r50(clip,boxes_with_id[:, 0:4])
                inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0],1),inp_boxes],dim=1).float()#转成和inputs一样的类型
                if isinstance(inputs, list):
                    inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
                else:
                    inputs = inputs.unsqueeze(0).to(device)
                with torch.no_grad():
                    slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                    slowfaster_preds = slowfaster_preds.cpu()#请注意slowfast的结果并没有经过归一化
                    slowfaster_preds = slowfaster_preds / slowfaster_preds.sum(axis=1,keepdims=True)#不会出现分母为0报错的情况

                result_labels = []
                for id_pres in slowfaster_preds:
                    result_labels.append(np.argsort(-id_pres).tolist()[:15])
                for i,(tid, avalabel) in enumerate(zip(boxes_with_id[:, 4].tolist(), result_labels)):
                    id_to_ava_labels[tid] = {"action_index":avalabel,"action_name":[ava_labelnames[action_index + 1].split(" ")[0] for action_index in avalabel],"action_prob":[round(slowfaster_preds[i][action_index].item(), 3) for action_index in avalabel]}

            max_conf = {}
            for r in results:#每n帧共用一个动作类型
                yolopreds_filter(r, id_to_ava_labels, max_conf)

            flag = False #标志该帧有没有目标对象，有的话就重复写几帧，便于查看
            for r in results:
                flag = False
                for i,box in enumerate(r.final_boxes):
                    if box[4] in max_conf.keys():
                        if (box[5] == max_conf[box[4]]):
                            plot_one_box(box, r.orig_img, r.final_boxes_colors[i], r.final_boxes_textes[i])
                            f.write(r.final_boxes_textes[i] + "\n")
                            flag = True
                    else:
                        plot_one_box(box, r.orig_img, r.final_boxes_colors[i], r.final_boxes_textes[i])
                        flag = True
                if flag:
                    for i in range(15):
                        output_video.write(r.orig_img)
                else:
                    output_video.write(r.orig_img)

            results = []
            id_to_ava_labels = {}#每n帧共用一个动作类型，不保留到下一批

        processed_count += 1
        print("{}/{},{}%".format(processed_count, total_frames, round((processed_count / total_frames) * 100, 2)))
        ret, frame = cap.read()
        slowfast_stack.append(frame)

    cap.release()
    output_video.release()
    print('saved video to:',config.output)
    f.close()

if __name__ == "__main__":
    videos = glob.glob("C:/Users/wmingdru/Desktop/loubao/*")
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
