import cv2
import numpy as np
from utils import *
import sys 
import os
import argparse
import torch
import torch.utils.data as data
import torch.nn as nn
from losses.utils import _gather_feat, _transpose_and_gather_feat
from utils.image import get_affine_transform, affine_transform
from resNet_GC_obj_att_for_demo import get_pose_net, save_model, load_model
'''
#For cv rectange documentation
cv.Rectangle(img, pt1, pt2, color, thickness=1, lineType=8, shift=0) → None¶
Parameters: 
img – Image.
pt1 – Vertex of the rectangle.
pt2 – Vertex of the rectangle opposite to pt1 .
rec – Alternative specification of the drawn rectangle.
color – Rectangle color or brightness (grayscale image).
thickness – Thickness of lines that make up the rectangle. Negative values, like CV_FILLED , mean that the function has to draw a filled rectangle.
lineType – Type of the line. See the line() description.
shift – Number of fractional bits in the point coordinates.
'''
def parse_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorboard_dir", type=str, default= r"D:\00_master_2\no_reg_version\tensorboard\res101_obj_v4_self")
    parser.add_argument('--test', action="store_true")

    parser.add_argument('--pth_dir', type=str, default= r"D:\00_master_2\no_reg_version\res_101\pth_file")
    parser.add_argument("--pth_save_name", type=str, default="res101_obj_v4_self")
    parser.add_argument("--pth_test_name", type=str, default="large_hour_objatt_v2_cls6_V4_self_ep640_2.00.pt")


    parser.add_argument('--root_dir', type=str, default= r"D:\dataset\COCO\coco_2017\knife_scissors_2017")
    parser.add_argument('--train_img_path', type=str, default= r"D:\dataset\COCO\coco_2017\knife_scissors_2017\knife_2017_train")
    parser.add_argument('--train_csv_path', type=str, default= r"D:\dataset\COCO\coco_2017\knife_scissors_2017\knife_2017_train.csv")
    parser.add_argument('--val_img_path', type=str, default= r"D:\dataset\COCO\coco_2017\knife_scissors_2017\knife_2017_val")
    parser.add_argument('--val_csv_path', type=str, default= r"D:\dataset\COCO\coco_2017\knife_scissors_2017\knife_2017_val.csv")
    parser.add_argument('--test_img_path', type=str, default= r"D:\dataset\rgb_view_1080p\data_img\0040")
    parser.add_argument('--test_csv_path', type=str, default= r"D:\dataset\rgb_view_1080p\csv_file\for_01_39_51.csv")

     # 所有的 train 和 val 放在同一個資料夾, csv 用 train_csv_path
    parser.add_argument("--combine_img_path", type=str, default= r"D:\dataset\rgb_view_1080p\train_set_img")
    parser.add_argument("--combine_csv_path", type=str, default= r"D:\dataset\rgb_view_1080p\csv_file\for_01_39_51.csv")

    # 訓練前要調整的參數
    # class_dict_coco = {'knife': 0,
    #                     'person' : 1,
    #                 }
    # class_dict_coco = {'person': 0,
    #                 }
    class_dict_coco = {'knife': 0,
    }

    class_dict_self = {'left_hand': 0,
                        "right_hand": 0,
                        "scalpel": 1,
                        # "other_instruments": 2,
                        "scalpel_tip": 2,
                        }
    parser.add_argument('--cls', type=dict, default=class_dict_self)
    parser.add_argument('--num_epochs', type=int, default=160)
    parser.add_argument("--num_class", type=int, default=6)
    parser.add_argument("--resize_wh", type=int, default=512)
    parser.add_argument("--down_ratio", type=int, default=4)
    parser.add_argument('--lr', type=float, default=2.5e-4)  # 預設 0.00025   0.00025/ 128(原batch) * 2(新batch) 若hm_loss卡在9.208，則要將lr調低
    parser.add_argument('--lr_step', type=str, default='100,140')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=16)
    ##################################################################################################

   

    # parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'pascal'])
    # parser.add_argument('--arch', type=str, default='large_hourglass')
    # parser.add_argument('--split_ratio', type=float, default=1.0)

    parser.add_argument("--val_intervals", type=int, default=1)   # 訓練每幾個epoch要validation一次

    parser.add_argument("--IOU_thresh", type=float, default=0.5)    # 預設0.5
    parser.add_argument("--score_thresh", type=float, default=0.3)  # 預設0.3
    
    cfg = parser.parse_args()
    print(cfg.cls)
    cfg.lr_step = [int(i) for i in cfg.lr_step.split(',')]
    print(cfg.lr_step)
    return cfg
def img_preprocessing(img):
    resize_wh = 512
    # resize 要輸入的image
    img = cv2.resize(img, (resize_wh, resize_wh))
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)   //給 unity AR眼鏡用
    img = img.astype('float32')
    cv2.normalize(img, img, 1.0, 0, cv2.NORM_MINMAX)
    img = img.transpose(2, 0, 1)

    # c 是計算原始圖像中心點
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    # s 是得到原始圖像最長的一條邊
    s = 512

    output_wh = resize_wh // 4

    ret = {'input': img}
    meta = {'c': c, 's': s}
    # ret.update(meta)
    ret['meta'] = meta
    return ret

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    # print("scores:",scores)
    
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
      
    return detections

def calculate_IOU(detection, bboxes):
    # print(detection.get_device())
    # print(bboxes.get_device())
    box_x_topleft = max(detection[0], bboxes[0])
    box_y_topleft = max(detection[1], bboxes[1])
    box_x_botright = min(detection[2], bboxes[2])
    box_y_botright = min(detection[3], bboxes[3])
    overlap_w = box_x_botright - box_x_topleft + 1
    overlap_h = box_y_botright - box_y_topleft + 1
    overlap_area = overlap_w * overlap_h

    detection_w = detection[2] - detection[0] + 1
    detection_h = detection[3] - detection[1] + 1
    detection_area = detection_w * detection_h

    bboxes_w = bboxes[2] - bboxes[0] + 1
    bboxes_h = bboxes[3] - bboxes[1] + 1
    bboxes_area = bboxes_w * bboxes_h
    
    union_area = bboxes_area + detection_area - overlap_area
    # if (union_area<0):
    #     print(detection)
    #     print(bboxes)
    #     print(xxx)
    IOU = overlap_area / union_area
    return IOU


def centerNet_nms(detections, num_classes, IOU_thresh = 0.7, score_thresh = 0.7):
    detections = detections[0]   # [B, k, 6] => [k, 6]
    results = []                 # 建一個2維 list 來去儲存每個 class 的 bounding box結果，計算該class有幾個物件
    class_list = []
    for i in range(num_classes):
        results.append([])
        class_list.append(0)

    for i in range(len(detections)):
        cls = int(detections[i][-1])
        can_add = True
        for t in range(4):
            # 若是detection裡的點有小於0 或是 xmax < xmin 或是 ymax < ymin， 就不加進去results裡面
            if(detections[i][t]<0) or ((detections[i][0] >= detections[i][2]) or (detections[i][1] >= detections[i][3])):
                detections[i][t] = 0
                can_add = False
            # if(detections[i][t]<0):
            #     detections[i][t] = 0
                # can_add = False
        if (len(results[cls]) == 0) and (detections[i][4] > score_thresh) and can_add == True:    # 若該class的results是空的，就存進第一個 k 的 bounding box 和score 並且該類別數量+1
            results[cls].append(detections[i][0:4])
            class_list[cls] = class_list[cls] + 1
        elif (len(results[cls]) != 0):
            total = len(results[cls])
            for j in range(total):
                IOU = calculate_IOU(detections[i][0:4], results[cls][j])
                # if (IOU < IOU_thresh and IOU >0 and detections[i][4] > score_thresh):
                if (IOU < IOU_thresh and detections[i][4] > score_thresh):
                    pass
                else:
                    can_add = False
            if (can_add):
                results[cls].append(detections[i][0:4])
                class_list[cls] = class_list[cls] + 1       

    return results, class_list


# def draw_bounding_box(cfg, img, results, class_list, num_class, dataset_class, batch):
#     # img_path = os.path.join(img_path, file_name)
#     # print(file_name)
#     # img = cv2.imread(img_path)
#     # cv2.namedWindow(file_name, cv2.WINDOW_AUTOSIZE)
#     # img = np.array(img)
#     # print(np.shape(img))
#     shape = img.shape  #(h, w)

#     # for i in range(num_class):
#     for i in range(2,3): #只畫刀尖
#         for name, value in dataset_class.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
#             if value == i:
#                 cls_name = name
#         for j in range(class_list[i]):
#             # c 是計算原始圖像中心點
#             c = np.array([cfg.resize_wh / 2., cfg.resize_wh / 2.], dtype=np.float32)
#             # s 是得到原始圖像最長的一條邊
#             s = max(cfg.resize_wh, cfg.resize_wh) * 1.0
#             trans = get_affine_transform(c, s, 0, (cfg.resize_wh/cfg.down_ratio, cfg.resize_wh/cfg.down_ratio), inv=1)
#             # detech = results[i][j].detach().numpy()
#             detech = results[i][j].cpu().detach().numpy()
#             bbox_x_topleft, bbox_y_topleft = affine_transform(detech[:2], trans)
#             bbox_x_botright, bbox_y_botright = affine_transform(detech[2:], trans)
#             # print(results[i][j])
#             # bbox_x_topleft = int(bbox_x_topleft * shape[1] / (cfg.resize_wh))
#             # bbox_y_topleft = int(bbox_y_topleft * shape[0] / (cfg.resize_wh))
#             # bbox_x_botright = int(bbox_x_botright * shape[1] / (cfg.resize_wh))
#             # bbox_y_botright = int(bbox_y_botright * shape[0] / (cfg.resize_wh))
#             bbox_x_topleft = int(bbox_x_topleft * shape[1] / (cfg.resize_wh))
#             bbox_y_topleft = int(bbox_y_topleft * shape[0] / (cfg.resize_wh))
#             bbox_x_botright = int(bbox_x_botright * shape[1] / (cfg.resize_wh))
#             bbox_y_botright = int(bbox_y_botright * shape[0] / (cfg.resize_wh))
#             # print(bbox_x_topleft, bbox_y_topleft, bbox_x_botright, bbox_y_botright)
#             ################################################################################################
#             #設定bounding box的框的顏色粗細等
#             bbox_color = (0, 255, 0)
#             bbox_thickness = 1
#             bbox_lineType = 4
            
            
#             cv2.rectangle(img, (bbox_x_topleft, bbox_y_topleft), (bbox_x_botright, bbox_y_botright), bbox_color, bbox_thickness, bbox_lineType)
#             ###################################################################################################################################################
#             #文字框大小
#             text_size = cv2.getTextSize(cls_name, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
            
#             #文字框右上角座標
#             textbottom = np.array([bbox_x_topleft, bbox_y_topleft]) + np.array([text_size[0], -text_size[1]])
#             cv2.rectangle(img, (bbox_x_topleft, bbox_y_topleft), tuple(textbottom), bbox_color, -1)

#             cv2.putText(img, cls_name, (bbox_x_topleft, bbox_y_topleft), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)
#             # cv2.putText(img, class_name, (int(round(float(row["bbox_x_topleft"]))), int(round(float(row["bbox_y_topright"]))) - int(text_size[1]/2 + 4)), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)

#             ####################################################################################################################################################
    
#     # cv2.imshow(file_name, img)
#     # cv2.imwrite(os.path.join("./res_101/gc_obj_att_img/0040",  file_name), img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     return img
def draw_bounding_box(cfg, img, results, class_list, num_class, dataset_class, batch):
    # img_path = os.path.join(img_path, file_name)
    # print(file_name)
    # img = cv2.imread(img_path)
    # cv2.namedWindow(file_name, cv2.WINDOW_AUTOSIZE)
    # img = np.array(img)
    # print(np.shape(img))
    shape = img.shape  #(h, w)
    draw_list = [2, 4]
    # for i in range(num_class):
    for i in draw_list: #只畫刀尖
        for name, value in dataset_class.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if value == i:
                cls_name = name
        # for j in range(class_list[i]):
        if (class_list[i]>0):
            # c 是計算原始圖像中心點
            c = np.array([cfg.resize_wh / 2., cfg.resize_wh / 2.], dtype=np.float32)
            # s 是得到原始圖像最長的一條邊
            s = max(cfg.resize_wh, cfg.resize_wh) * 1.0
            trans = get_affine_transform(c, s, 0, (cfg.resize_wh/cfg.down_ratio, cfg.resize_wh/cfg.down_ratio), inv=1)
            # detech = results[i][j].detach().numpy()
            detech = results[i][0].cpu().detach().numpy()
            bbox_x_topleft, bbox_y_topleft = affine_transform(detech[:2], trans)
            bbox_x_botright, bbox_y_botright = affine_transform(detech[2:], trans)
            # print(results[i][j])
            # bbox_x_topleft = int(bbox_x_topleft * shape[1] / (cfg.resize_wh))
            # bbox_y_topleft = int(bbox_y_topleft * shape[0] / (cfg.resize_wh))
            # bbox_x_botright = int(bbox_x_botright * shape[1] / (cfg.resize_wh))
            # bbox_y_botright = int(bbox_y_botright * shape[0] / (cfg.resize_wh))
            bbox_x_topleft = int(bbox_x_topleft * shape[1] / (cfg.resize_wh))
            bbox_y_topleft = int(bbox_y_topleft * shape[0] / (cfg.resize_wh))
            bbox_x_botright = int(bbox_x_botright * shape[1] / (cfg.resize_wh))
            bbox_y_botright = int(bbox_y_botright * shape[0] / (cfg.resize_wh))
            # print(bbox_x_topleft, bbox_y_topleft, bbox_x_botright, bbox_y_botright)
            ################################################################################################
            #設定bounding box的框的顏色粗細等
            bbox_color = (0, 255, 0)
            bbox_thickness = 1
            bbox_lineType = 4
            
            
            cv2.rectangle(img, (bbox_x_topleft, bbox_y_topleft), (bbox_x_botright, bbox_y_botright), bbox_color, bbox_thickness, bbox_lineType)
            ###################################################################################################################################################
            #文字框大小
            text_size = cv2.getTextSize(cls_name, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
            
            #文字框右上角座標
            textbottom = np.array([bbox_x_topleft, bbox_y_topleft]) + np.array([text_size[0], -text_size[1]])
            cv2.rectangle(img, (bbox_x_topleft, bbox_y_topleft), tuple(textbottom), bbox_color, -1)

            cv2.putText(img, cls_name, (bbox_x_topleft, bbox_y_topleft), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)
            # cv2.putText(img, class_name, (int(round(float(row["bbox_x_topleft"]))), int(round(float(row["bbox_y_topright"]))) - int(text_size[1]/2 + 4)), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)

            ####################################################################################################################################################
    
    # cv2.imshow(file_name, img)
    # cv2.imwrite(os.path.join("./res_101/gc_obj_att_img/0040",  file_name), img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img

#Function for getting video feed via webcam
def getVideoFeed(cfg, class_dict):
    dataset_class = class_dict

    torch.manual_seed(317)
    torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_pose_net(cfg.num_class, 101)  #網路架構 resNet
    model = model.to(device)
    model = load_model("res101_obj_v3_self_final_ep160_0.34.pt", model, device)

    model.eval()
    cam = cv2.VideoCapture(0); #0 enables the webcam device
    print ("Press Esc key to exit..")
    while True:
        x,img_frame = cam.read()
        if not x:
            print('Unable to read from camera feed')
            sys.exit(0)

        batch = img_preprocessing(img_frame)
        
        input = torch.tensor(batch["input"]).to(device)
        outputs = model(input.unsqueeze(0))
        hm = outputs[0]['hm'].sigmoid_()
        wh = outputs[0]['wh']
        reg = None
        det = ctdet_decode(hm, wh, reg)

        results, class_list = centerNet_nms(det, cfg.num_class, cfg.IOU_thresh, cfg.score_thresh)
        
        # 將 bounding box一張圖一張圖畫出來
        img_frame = draw_bounding_box(cfg, img_frame, results, class_list, cfg.num_class, dataset_class, batch)
        

        cv2.imshow('knife_detection',img_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: #27 for Esc Key
            break

    cv2.destroyAllWindows()
    cam.release()

#Function for getting video feed via webcam
def unity_AR_Feed_v2(cfg, class_dict, rec_img):
    dataset_class = class_dict

    torch.manual_seed(317)
    torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_pose_net(cfg.num_class, 101)  #網路架構 resNet
    model = model.to(device)
    model = load_model("res101_obj_v3_self_final_ep160_0.34.pt", model, device)

    model.eval()
    cam = cv2.VideoCapture(0); #0 enables the webcam device
    print ("Press Esc key to exit..")
    while True:
        # x,img_frame = cam.read()
        # if not x:
        #     print('Unable to read from camera feed')
        #     sys.exit(0)

        batch = img_preprocessing(rec_img)
        
        input = torch.tensor(batch["input"]).to(device)
        outputs = model(input.unsqueeze(0))
        hm = outputs[0]['hm'].sigmoid_()
        wh = outputs[0]['wh']
        reg = None
        det = ctdet_decode(hm, wh, reg)

        results, class_list = centerNet_nms(det, cfg.num_class, cfg.IOU_thresh, cfg.score_thresh)
        
        # 將 bounding box一張圖一張圖畫出來
        img_frame = draw_bounding_box(cfg, img_frame, results, class_list, cfg.num_class, dataset_class, batch)
        

        cv2.imshow('knife_detection',img_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: #27 for Esc Key
            break

    cv2.destroyAllWindows()
    cam.release()




if __name__ == '__main__':
    class_dict = class_dict_self = {'left_hand': 0,
                        "right_hand": 0,
                        "scalpel": 1,
                        # "other_instruments": 2,
                        "scalpel_tip": 2,
                        }

    cfg = parse_setting()
    getVideoFeed(cfg, class_dict)
    # if (cfg.test):
    #     test(cfg, cfg.cls)
    # else:
    #     main(cfg, cfg.cls)