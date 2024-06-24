
import random,warnings, argparse
warnings.filterwarnings("ignore", category=UserWarning)
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from deep_sort.deep_sort import DeepSort
from ultralytics import YOLO
from utils.myutil import *

def main(config):
    device = "cuda"
    model = YOLO("./weights/yolov8m.engine")
    video_model = slowfast_r50_detection(True).eval().to(device)
    ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("utils/ava_action_list.pbtxt")


    vide_save_path = config.output
    video = cv2.VideoCapture(config.input)
    width, height = int(video.get(3)), int(video.get(4))
    video.release()
    outputvideo = cv2.VideoWriter(vide_save_path,cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    cap = MyVideoCapture(config.input)
    id_to_ava_labels = {}
    while not cap.end:
        ret, img = cap.read()
        if not ret:
            continue
        result = model.track(source=img,verbose=False,persist=True,tracker="./track_config/botsort.yaml",classes=[0],conf=0.3,iou=0.7)[0]
        boxes = result.boxes.data.cpu().numpy()
        if len(cap.stack) == 30:
            print(f"processing {cap.idx // 30}th second clips")
            clip = cap.get_video_clip()
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


                #注意追踪的id是用浮点数表示的
                # for tid, avalabel in zip(boxes_with_id[:,4].tolist(),
                #                          np.argmax(slowfaster_preds, axis=1).tolist()):
                #     id_to_ava_labels[tid] = ava_labelnames[avalabel + 1]

                # #我这里选取1:bend/bow 7:jump/leap 11:sit 20:climb这四个动作
                # for tid, avalabel in zip(boxes_with_id[:, 4].tolist(),slowfaster_preds.tolist()):
                #
                #     name1,name2,name3,name4 = ava_labelnames[1].split(" ")[0],ava_labelnames[7].split(" ")[0],ava_labelnames[11].split(" ")[0],ava_labelnames[20].split(" ")[0]
                #     prob1,prob2,prob3,prob4 = round(avalabel[0],2),round(avalabel[6],2),round(avalabel[10],2),round(avalabel[19],2)
                #     action_text = "{}:{} {}:{} {}:{} {}:{}".format(name1,prob1,name2,prob2,name3,prob3,name4,prob4)
                #     id_to_ava_labels[tid] = action_text

                result_labels = []
                #为每个id选取概率最大的5个动作
                for id_pres in slowfaster_preds:
                    result_labels.append(np.argsort(-id_pres).tolist()[:5])
                for i,(tid, avalabel) in enumerate(zip(boxes_with_id[:, 4].tolist(), result_labels)):
                    action_text = ""
                    for action_index in avalabel:
                        current_action = " " + ava_labelnames[action_index + 1].split(" ")[0] + ":" + str(round(slowfaster_preds[i][action_index].item(),2))
                        action_text += current_action

                    id_to_ava_labels[tid] = action_text

        #这意味着第一帧给每个id分配的动作类型，默认也分配给剩下n-1帧的相同id(n是clip包含的帧数)
        save_yolopreds_tovideo_yolov8_version(result, id_to_ava_labels,outputvideo)


    cap.release()
    outputvideo.release()
    print('saved video to:', vide_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="./videos/allscenes_merged.mp4",
                        help='test imgs folder or video or camera')
    parser.add_argument('--output', type=str, default="./videos/output/output_allscenes.mp4",
                        help='folder to save result imgs, can not use input folder')

    config = parser.parse_args()
    main(config)
