import os
import glob
import json
import pandas as pd
import numpy as np
import csv
import torch
import time
from torch.autograd import Variable
import cv2
from torch.nn import functional as F
import gc
import math
# from opts import parse_opts_online
# from model import generate_model
# from mean import get_mean, get_std
# from spatial_transforms import *
# from temporal_transforms import *
# from target_transforms import ClassLabel
# from dataset import get_online_data
# import datasets.egogesture_online as ego_on
# from utils import  AverageMeter, LevenshteinDistance, Queue
from Demo import parse_setting, getVideoFeed, ctdet_decode, centerNet_nms, draw_bounding_box
from losses.utils import _gather_feat, _transpose_and_gather_feat
from utils.image import get_affine_transform, affine_transform
# from resNet_GC_obj_att_for_demo import get_pose_net, save_model, load_model
from large_hourglasses_obj_att_v2 import save_model, load_model
# from hourglasses_objatt_simpleAdd import get_large_hourglass_net
# from hourglasses_CSPv3_dcn_v4_simpleAddv3 import get_large_hourglass_net
# from hourglasses_CSP_v5_simpleAddv1_v1 import get_large_hourglass_net
from hourglasses_CSP_v11_simpleAddv1_v1 import get_large_hourglass_net


import pdb
import datetime

from ast import arg
from asyncio.proactor_events import _ProactorWritePipeTransport
from asyncio.windows_events import NULL
from base64 import decode, encode
from email.mime import image
from glob import glob
from os import system
import socket
from tkinter import Image
from tkinter.tix import IMAGE

import keyboard
import threading
import sys
import io
import PIL.Image as Image
import argparse
import queue



knife_cfg = parse_setting()

torch.manual_seed(317)
# torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training
num_gpus = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# model = get_pose_net(knife_cfg.num_class, 101)  #網路架構 resNet
model = get_large_hourglass_net(knife_cfg.num_class) 
model = model.to(device)
# model = load_model("hour_CSPv5_simpleAddv1_v1_ep496_mAP74.85.pt", model, device)
model = load_model("hourglasses_CSP_v11_simpleAddv1_v1_ep498_mAP75.34.pt", model, device)
# model = load_model("large_hour_objatt_v2_cls6_V2_self_ep640_2.00.pt", model, device)
model.eval()
output_encode = "0,0"

client_rgb = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_tof = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


num = 1  # 存到第幾張圖片
num_tof = 1
time_bg = time.time()  # 算FPS，從接收到資料開始算
time_bg_tof = time.time()  # 算FPS，從接收到資料開始算
time_end = time.time()  # 算FPS，到存取一張完整圖片結束
time_end_tof = time.time()  # 算FPS，到存取一張完整圖片結束
fpsavg = 0.0  # FPS的平均
fpsavg_tof = 0.0  # FPS的平均
count = 0
count_tof = 0
# img_numpy = []

def img_length(size, rec_data):
    # print(rec_data[:size+1])
    try:
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@?????????????1')
        rec_data[size:size+1].decode('ascii')
        if rec_data[size:size+3].decode('ascii') != 'end':

            return rec_data[size:size+1].decode('ascii')+(img_length(size=size+1, rec_data=rec_data))
        else:
            return ''
    except:
        print('except')
        return ''


def seperate(rec_data, size, bytearr, client):
    global num, time_bg, time_end, fpsavg, img_numpy

    if size == 0:
        # print('2')
        size = int(img_length(size=size, rec_data=rec_data))
        rec_data = rec_data[len(str(size))+3:]


    if len(rec_data) < size:
        # print('3')
        bytearr.extend(bytearray(rec_data))
        size -= len(rec_data)

    elif len(rec_data) == size:
        # print('4')
        bytearr.extend(bytearray(rec_data))
        size -= len(rec_data)

        img_buffer_numpy = np.frombuffer(bytes(bytearr), dtype=np.uint8)
        img_numpy = cv2.flip(cv2.imdecode(img_buffer_numpy, cv2.IMREAD_COLOR), -1)

        time_end = time.time()
        '''fpsavg += 1/(time_end-time_bg)'''
        time_bg = time.time()
        num += 1
        bytearr = bytearray()


    elif len(rec_data) > size:
        bytearr.extend(bytearray(rec_data[:size]))
        img_buffer_numpy = np.frombuffer(bytes(bytearr), dtype=np.uint8)
        img_numpy = cv2.flip(cv2.imdecode(img_buffer_numpy, cv2.IMREAD_COLOR), -1)

        time_end = time.time()
        '''fpsavg += 1/(time_end-time_bg)'''
        # print(f'FPS = {fpsavg/num:.6f} ')
        time_bg = time.time()
        num += 1
        bytearr = bytearray()
        # print('next')
        temp = bytearray()
        temp.extend(rec_data[size:])
        rec_data = client.recv(400000)
        temp.extend(rec_data)
        rec_data = bytes(temp)
        # print('recv length = '+str(len(rec_data)))
        size = 0
        size, bytearr = seperate(rec_data, size, bytearr, client)
        # print('size = '+str(size))

    return size, bytearr


def job(ip, port):
    global time_bg
    try:
        if port == 5566:
            client = client_rgb
            type = 'RGB'
        else:
            # client = client_tof
            type = 'ToF'
        client.connect((ip, port))
        time_bg = time.time()
        size = 0
        bytearr = bytearray()
        temp = bytearray()
        rec_data = client.recv(400000)
        # print(len(rec_data))
        temp.extend(rec_data)
        rec_data = client.recv(400000)
        # print(len(rec_data))
        temp.extend(rec_data)
        rec_data = bytes(temp)
        size = int(img_length(size, rec_data))
        rec_data = rec_data[len(str(size))+3:]
        
        receive = threading.Thread(target=receiving, args=(client, rec_data, size, bytearr))
        send = threading.Thread(target=sending, args=(client, ))
        receive.daemon = True
        send.daemon = True
        receive.start()
        send.start()

    except:
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@byebyebyebyebyebyebyebyebye9999')
        client_rgb.close()
        # client_tof.close()

def job2(ip, port):
    global time_bg
    # count_result = 1
    try:
        if port == 5566:
            client = client_rgb
            type = 'RGB'
        else:
            client = client_tof
            type = 'ToF'
        client.connect((ip, port))
        time_bg = time.time()
        size = 0
        bytearr = bytearray()
        temp = bytearray()
        rec_data = client.recv(400000)
        # print(len(rec_data))
        print('@@@@@@@@@@@@@@@@@@@@@19')
        temp.extend(rec_data)
        rec_data = client.recv(400000)
        print('@@@@@@@@@@@@@@@@@@@@@29')
        # print(len(rec_data))
        temp.extend(rec_data)
        rec_data = bytes(temp)
        size = int(img_length(size, rec_data))
        rec_data = rec_data[len(str(size))+3:]

        return client, rec_data, size, bytearr
        
    except:
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@byebyebyebyebyebyebyebyebye9999')
        if port == 5566:
            client_rgb.close()
        else:
            client_tof.close()
        



def receiving(client, rec_data, size, bytearr):
    while(True):
        size, bytearr = seperate(rec_data, size, bytearr, client)
        rec_data = client.recv(400000)
        # print(len(bytearr))

def receiving_depth(client, rec_data, size, bytearr):
    while(True):
        size, bytearr = seperate(rec_data, size, bytearr, client)
        rec_data = client.recv(400000)
        # print(len(bytearr))
        print("rec_data:  ", rec_data)

def sending(client):
    # count_result = 2
    while(True):
        try:
            # if(len(predicted) == count_result):
            # data = idx_to_class[predicted[-1]] + "\n"
            data = output_encode
            print(data)
            # print("data")
            # # 對資料進行編碼格式轉換
            data = data.encode('utf-8')
            # # 傳送data
            client.sendall(data)
            # print(idx_to_class[predicted[-1]])
            time.sleep(0.5)
            # '''data = "111"
            # data = data.encode('utf-8')
            # client.sendall(data)'''
            pass
        except:
            # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@byebyebyebyebyebyebyebyebye8888')
            pass


    # try:
    #     while(len(predicted) == count_result):


def seperate_tof(rec_data, size, bytearr, client, type):
    global num_tof, time_bg_tof, time_end_tof, fpsavg_tof
    if size == 0:
        print('2')
        size = int(img_length(size=size, rec_data=rec_data))
        rec_data = rec_data[len(str(size))+3:]
        print('2end')

    if len(rec_data) < size:
        print('3')
        bytearr.extend(bytearray(rec_data))
        size -= len(rec_data)
        print('bytearr length = '+str(len(bytearr)))
        print('size'+str(size))
        print('3end')

    elif len(rec_data) == size:
        print('4')
        bytearr.extend(bytearray(rec_data))
        size -= len(rec_data)
        print('size'+str(size))
        print('bytearr length = '+str(len(bytearr)))
        print('unsave')
        time_end_tof = time.time()
        fpsavg_tof += 1/(time_end_tof-time_bg_tof)
        print(f'FPS = {fpsavg_tof/num_tof:.6f} ')
        time_bg_tof = time.time()
        num_tof += 1
        bytearr = bytearray()
        print('save')
        print('4end')

    elif len(rec_data) > size:
        print('5')
        print('rec_data length = '+str(len(rec_data)))
        print('size ='+str(size))
        print(rec_data[size:size+10])
        bytearr.extend(bytearray(rec_data[:size]))
        print('bytearr length = '+str(len(bytearr)))
        print('num_tof = '+str(num_tof))
        print('unsave')
        print('save')
        time_end_tof = time.time()
        fpsavg_tof += 1/(time_end_tof-time_bg_tof)
        print(f'FPS = {fpsavg_tof/num_tof:.6f} ')
        time_bg_tof = time.time()
        num_tof += 1
        bytearr = bytearray()
        print('next')
        temp = bytearray()
        temp.extend(rec_data[size:])
        rec_data = client.recv(400000)
        temp.extend(rec_data)
        rec_data = bytes(temp)
        print('recv length = '+str(len(rec_data)))
        size = 0
        size, bytearr = seperate_tof(rec_data, size, bytearr, client, type)
        print('size = '+str(size))

    return size, bytearr


def job_tof(ip, port):
    global time_bg_tof
    try:
        if port == 5566:
            client = client_rgb
            type = 'RGB'
        else:
            # client = client_tof
            type = 'ToF'
        client.connect((ip, port))
        time_bg_tof = time.time()
        size = 0
        bytearr = bytearray()
        temp = bytearray()
        rec_data = client.recv(400000)
        print(len(rec_data))
        temp.extend(rec_data)
        rec_data = client.recv(400000)
        print(len(rec_data))
        temp.extend(rec_data)
        rec_data = bytes(temp)
        size = int(img_length(size, rec_data))
        rec_data = rec_data[len(str(size))+3:]
        print('size = '+str(size))
        print('sizelen = '+str(len(str(size))))
        print('recv length = '+str(len(rec_data)))
        while(True):
            size, bytearr = seperate_tof(rec_data, size, bytearr, client, type)
            print('\n\n')
            print('size = '+str(size))
            print('recv length = '+str(len(rec_data)))
            rec_data = client.recv(400000)
            # 接受控制檯的輸入
            # data = input()
            data = 'holymoly'
            # 對資料進行編碼格式轉換
            data = data.encode('utf-8')
            # 傳送data
            client.sendall(data)
    except:
        print('server is not found')
        # client.sendall('hello'.encode('ascii'))
        client_rgb.close()
        # client_tof.close()

#-------------------------------------------IP 輸入-------------------------------------------------------------------------
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--ip', type=str,
                        default='192.168.0.177', help='device ip')
    parser.add_argument('--port', type=int, default=5566,
                        help='port 5566 is RGB and port 5577 is ToF')
    return parser.parse_args()

# opt = parse_opts_online()
# data1 = ego_on.load_annotation_data(opt.annotation_path)
# class_to_idx = ego_on.get_class_labels(data1)
# idx_to_class = {}
# print('tttttttttt', type(idx_to_class))
# predicted = [83]
class_dict = {'left_hand': 0,
                "right_hand": 0,
                "scalpel": 1,
                # "other_instruments": 2,
                "scalpel_tip": 2,
                "tweezer" : 3,
                "liver" : 4,
                "other_instruments" : 5
                }
# knife_cfg = parse_setting()

# torch.manual_seed(317)
# # torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training
# num_gpus = torch.cuda.device_count()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# # model = get_pose_net(knife_cfg.num_class, 101)  #網路架構 resNet
# model = get_large_hourglass_net(knife_cfg.num_class) 
# model = model.to(device)
# model = load_model("large_hour_objatt_v2_cls6_V3_self_ep640_2.00.pt", model, device)
# model.eval()
# output_encode = "0,0"


args = parse_args()
ip = args.ip
print('ip = ', ip)
threads = []
queue = queue.Queue()
'''t_rgb = threading.Thread(target=job, args=(ip, 5566))
print('rgb begin')
# t_tof = threading.Thread(target=job_tof, args=(ip, 5577))
print('tof begin')
t_rgb.daemon = True
# t_tof.daemon = True
t_rgb.start()
# t_tof.start()'''


# 接收深度資訊
# client_tof, rec_data_tof, size_tof, bytearr_tof = job2(ip, 5577)
# receive_tof = threading.Thread(target=receiving_depth, args=(client_tof, rec_data_tof, size_tof, bytearr_tof))

client, rec_data, size, bytearr = job2(ip, 5566)
receive = threading.Thread(target=receiving, args=(client, rec_data, size, bytearr))
send = threading.Thread(target=sending, args=(client, ))
receive.daemon = True
# receive_tof.daemon = True
send.daemon = True
# receive_tof.start()
receive.start()
time.sleep(2.5)
send.start()
def weighting_func(x):
    return (1 / (1 + np.exp(-0.2 * (x - 9))))

def img_preprocessing(img):
    resize_wh = 512
    # resize 要輸入的image
    img = cv2.resize(img, (resize_wh, resize_wh))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
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

def unity_AR_Feed_v2(cfg, class_dict, rec_img):
    dataset_class = class_dict

    
    # cam = cv2.VideoCapture(0); #0 enables the webcam device
    print ("Press Esc key to exit..")
   
        # x,img_frame = cam.read()
        # if not x:
        #     print('Unable to read from camera feed')
        #     sys.exit(0)

    batch = img_preprocessing(rec_img)
    print("sssssssss456456")
    input = torch.tensor(batch["input"]).to(device)
    outputs = model(input.unsqueeze(0))
    hm = outputs[0]['hm'].sigmoid_()
    wh = outputs[0]['wh']
    reg = None
    det = ctdet_decode(hm, wh, reg)
    print("sssssssss123123")
    # results, class_list = centerNet_nms(det, cfg.num_class, cfg.IOU_thresh, cfg.score_thresh)
    # print("so long")
    # 將 bounding box一張圖一張圖畫出來
    # rec_img = draw_bounding_box(cfg, rec_img, results, class_list, cfg.num_class, dataset_class, batch)
    

    
    
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     print("You pressed q")
    #     data = 'end'
    #     # 對資料進行編碼格式轉換
    #     data = data.encode('utf-8')
    #     # 傳送data
    #     client_rgb.sendall(data)
    #     # client_tof.sendall(data)
    #     client_rgb.close()
    #     # client_tof.close()
    #     sys.exit()
    #     break
    
    # cam.release()
    return rec_img
# frame = img_numpy
# print(type(img_numpy))
# print(len(img_numpy))
# print(img_numpy)

def output_only_knife(cfg, result_from_nms, class_list):
    shape = (1000, 1500) #AR眼鏡的螢幕大小
    # c 是計算原始圖像中心點
    c = np.array([cfg.resize_wh / 2., cfg.resize_wh / 2.], dtype=np.float32)
    # s 是得到原始圖像最長的一條邊
    s = max(cfg.resize_wh, cfg.resize_wh) * 1.0
    trans = get_affine_transform(c, s, 0, (cfg.resize_wh/cfg.down_ratio, cfg.resize_wh/cfg.down_ratio), inv=1)

    if (class_list[2]>0):
    #for j in range(class_list[2]):  #只回傳一個刀尖結果
        # # c 是計算原始圖像中心點
        # c = np.array([cfg.resize_wh / 2., cfg.resize_wh / 2.], dtype=np.float32)
        # # s 是得到原始圖像最長的一條邊
        # s = max(cfg.resize_wh, cfg.resize_wh) * 1.0
        # trans = get_affine_transform(c, s, 0, (cfg.resize_wh/cfg.down_ratio, cfg.resize_wh/cfg.down_ratio), inv=1)
        # detech = results[i][j].detach().numpy()
        detech = result_from_nms[2][0].cpu().detach().numpy()
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
        center_x = int(round((bbox_x_topleft + bbox_x_botright)/2))
        center_y = int(round((bbox_y_topleft + bbox_y_botright)/2))
    else:
        bbox_x_topleft = 0
        bbox_y_topleft = 0
        bbox_x_botright = 0
        bbox_y_botright = 0
        center_x = 0
        center_y = 0

    # 回傳肝臟座標
    if (class_list[4]>0):
    #for j in range(class_list[2]):  #只回傳一個刀尖結果
        
        # detech = results[i][j].detach().numpy()
        detech = result_from_nms[4][0].cpu().detach().numpy()
        liver_x_topleft, liver_y_topleft = affine_transform(detech[:2], trans)
        liver_x_botright, liver_y_botright = affine_transform(detech[2:], trans)
        # print(results[i][j])
        # bbox_x_topleft = int(bbox_x_topleft * shape[1] / (cfg.resize_wh))
        # bbox_y_topleft = int(bbox_y_topleft * shape[0] / (cfg.resize_wh))
        # bbox_x_botright = int(bbox_x_botright * shape[1] / (cfg.resize_wh))
        # bbox_y_botright = int(bbox_y_botright * shape[0] / (cfg.resize_wh))
        liver_x_topleft = int(liver_x_topleft * shape[1] / (cfg.resize_wh))
        liver_y_topleft = int(liver_y_topleft * shape[0] / (cfg.resize_wh))
        liver_x_botright = int(liver_x_botright * shape[1] / (cfg.resize_wh))
        liver_y_botright = int(liver_y_botright * shape[0] / (cfg.resize_wh))
        liver_center_x = int(round((liver_x_topleft + liver_x_botright)/2))
        liver_center_y = int(round((liver_y_topleft + liver_y_botright)/2))
    else:
        liver_x_topleft = 0
        liver_y_topleft = 0
        liver_x_botright = 0
        liver_y_botright = 0
        liver_center_x = 0
        liver_center_y = 0
    

    if (center_x !=0 and center_y !=0 and liver_center_x != 0 and liver_center_y != 0):
        d_x = (center_x - liver_center_x)*73/1920
        d_y = (center_y - liver_center_y)*53/1080
        distance = math.sqrt(d_x * d_x + d_y * d_y)
        # distance = math.sqrt((center_x - liver_center_x)*(center_x - liver_center_x) + (center_y - liver_center_y)*(center_y - liver_center_y))
        distance_str = str(int(distance)) + " cm"
    else:
        distance_str = " "

    if (class_list[4]>0 and class_list[2]>0):
        if (distance != " "):
            if (distance < 5):
                warning = "1"
            else:
                warning = "0"
        # if ((center_x > liver_x_topleft and center_x <=liver_x_botright) and (center_y > liver_y_topleft and center_y <= liver_y_botright)):
        #     warning = "1"
        # else:
        #     warning = "0"
    else:
        warning = "0"

    tof_center_x = center_x * 224/1920
    tof_center_y = center_y * 172/1080
    tof_position = int(tof_center_y) * 224 + int(tof_center_x)

    return str(center_x)+","+str(center_y) +","+ str(liver_center_x)+","+str(liver_center_y) + "," + warning + "," + distance_str + "," + str(tof_position)


while (True):
    if len(img_numpy) == 0:
        pass
    else:
        rec_frame = img_numpy
        batch = img_preprocessing(rec_frame)
        input = torch.tensor(batch["input"]).to(device)
        input = input.unsqueeze(0)
        # print("sssssssss477777777")
        outputs = model(input)
        # print("8888888")
        hm = outputs[0]['hm'].sigmoid_()
        wh = outputs[0]['wh']
        # print("6666666")
        reg = None
        
        det = ctdet_decode(hm, wh, reg)
        # print("sssssssss123123")
        results, class_list = centerNet_nms(det, knife_cfg.num_class, knife_cfg.IOU_thresh, knife_cfg.score_thresh)
        rec_img = draw_bounding_box(knife_cfg, rec_frame, results, class_list, knife_cfg.num_class, class_dict, batch)
        output_encode = output_only_knife(knife_cfg, results, class_list)
        cv2.namedWindow("knife_detection", 0)
        cv2.imshow('knife_detection',rec_img)
        # key = cv2.waitKey(1) & 0xFF
        # if key == 27: #27 for Esc Key
        #     break
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("You pressed q")
            data = 'end'
            # 對資料進行編碼格式轉換
            data = data.encode('utf-8')
            # 傳送data
            client_rgb.sendall(data)
            # client_tof.sendall(data)
            client_rgb.close()
            # client_tof.close()
            sys.exit()
            break
cv2.destroyAllWindows()

# while(True):
#     # if len(img_numpy) == 0:
#     #     pass
#     # else:
#     print("111")
#     # frame = img_numpy 
    
#     # frame = cv2.resize(frame, (512, 512))
#     # # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)\
#     # cv2.namedWindow('knife_detection', 0)
#     # cv2.imshow('knife_detection',frame)
#     # key = cv2.waitKey(1) & 0xFF
#     # if key == 27: #27 for Esc Key
#     #     break
# cv2.destroyAllWindows()




# def load_models(opt):
#     opt.resume_path = opt.resume_path_det
#     opt.pretrain_path = opt.pretrain_path_det
#     opt.sample_duration = opt.sample_duration_det
#     opt.model = opt.model_det
#     opt.model_depth = opt.model_depth_det
#     opt.width_mult = opt.width_mult_det
#     opt.modality = opt.modality_det
#     opt.resnet_shortcut = opt.resnet_shortcut_det
#     opt.n_classes = opt.n_classes_det
#     opt.n_finetune_classes = opt.n_finetune_classes_det

#     if opt.root_path != '':
#         opt.video_path = os.path.join(opt.root_path, opt.video_path)
#         opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
#         opt.result_path = os.path.join(opt.root_path, opt.result_path)
#         if opt.resume_path:
#             opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
#         if opt.pretrain_path:
#             opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

#     opt.scales = [opt.initial_scale]
#     for i in range(1, opt.n_scales):
#         opt.scales.append(opt.scales[-1] * opt.scale_step)
#     opt.arch = '{}'.format(opt.model)
#     opt.mean = get_mean(opt.norm_value)
#     opt.std = get_std(opt.norm_value)

#     print(opt)
#     with open(os.path.join(opt.result_path, 'opts_det.json'), 'w') as opt_file:
#         json.dump(vars(opt), opt_file)

#     torch.manual_seed(opt.manual_seed)

#     detector, parameters = generate_model(opt)
#     detector = detector.cuda()
#     if opt.resume_path:
#         opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
#         print('loading checkpoint {}'.format(opt.resume_path))
#         checkpoint = torch.load(opt.resume_path)

#         detector.load_state_dict(checkpoint['state_dict'])

#     print('Model 1 \n', detector)
#     pytorch_total_params = sum(p.numel() for p in detector.parameters() if
#                                p.requires_grad)
#     print("Total number of trainable parameters: ", pytorch_total_params)

#     opt.resume_path = opt.resume_path_clf
#     opt.pretrain_path = opt.pretrain_path_clf
#     opt.sample_duration = opt.sample_duration_clf
#     opt.model = opt.model_clf
#     opt.model_depth = opt.model_depth_clf
#     opt.width_mult = opt.width_mult_clf
#     opt.modality = opt.modality_clf
#     opt.resnet_shortcut = opt.resnet_shortcut_clf
#     opt.n_classes = opt.n_classes_clf
#     opt.n_finetune_classes = opt.n_finetune_classes_clf
#     if opt.root_path != '':
#         opt.video_path = os.path.join(opt.root_path, opt.video_path)
#         opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
#         opt.result_path = os.path.join(opt.root_path, opt.result_path)
#         if opt.resume_path:
#             opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
#         if opt.pretrain_path:
#             opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

#     opt.scales = [opt.initial_scale]
#     for i in range(1, opt.n_scales):
#         opt.scales.append(opt.scales[-1] * opt.scale_step)
#     opt.arch = '{}'.format(opt.model)
#     opt.mean = get_mean(opt.norm_value)
#     opt.std = get_std(opt.norm_value)

#     print(opt)
#     with open(os.path.join(opt.result_path, 'opts_clf.json'), 'w') as opt_file:
#         json.dump(vars(opt), opt_file)

#     torch.manual_seed(opt.manual_seed)
#     classifier, parameters = generate_model(opt)
#     classifier = classifier.cuda()
#     if opt.resume_path:
#         print('loading checkpoint {}'.format(opt.resume_path))
#         checkpoint = torch.load(opt.resume_path)

#         classifier.load_state_dict(checkpoint['state_dict'])

#     print('Model 2 \n', classifier)
#     pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
#                                p.requires_grad)
#     print("Total number of trainable parameters: ", pytorch_total_params)

#     return detector, classifier

# detector, classifier = load_models(opt)

# send.start()

# if opt.no_mean_norm and not opt.std_norm:
#     norm_method = Normalize([0, 0, 0], [1, 1, 1])
# elif not opt.std_norm:
#     norm_method = Normalize(opt.mean, [1, 1, 1])
# else:
#     norm_method = Normalize(opt.mean, opt.std)

# spatial_transform = Compose([
#     Scale(112),
#     CenterCrop(112),
#     ToTensor(opt.norm_value), norm_method
# ])


# opt.sample_duration = max(opt.sample_duration_clf, opt.sample_duration_det)
# fps = ""
# # cap = cv2.VideoCapture(opt.video)
# # cap = cv2.VideoCapture(0)

# num_frame = 0
# clip = []
# active_index = 0
# passive_count = 0
# active = False
# prev_active = False
# finished_prediction = None
# pre_predict = False
# detector.eval()
# classifier.eval()
# cum_sum = np.zeros(opt.n_classes_clf, )
# clf_selected_queue = np.zeros(opt.n_classes_clf, )
# det_selected_queue = np.zeros(opt.n_classes_det, )
# myqueue_det = Queue(opt.det_queue_size, n_classes=opt.n_classes_det)
# myqueue_clf = Queue(opt.clf_queue_size, n_classes=opt.n_classes_clf)
# results = []
# prev_best1 = opt.n_classes_clf
# spatial_transform.randomize_parameters()


# # length_predicted = 0
# probability_best1 = 0.0

# for name, label in class_to_idx.items():
#     idx_to_class[label] = name


# while(True):
#     t1 = time.time()
#     frame = img_numpy
#     if num_frame == 0:
#         cur_frame = cv2.resize(frame,(320,240))
#         cur_frame = Image.fromarray(cv2.cvtColor(cur_frame,cv2.COLOR_BGR2RGB))
#         cur_frame = cur_frame.convert('RGB')
#         for i in range(opt.sample_duration):
#             clip.append(cur_frame)
#         clip = [spatial_transform(img) for img in clip]
#     clip.pop(0)
#     _frame = cv2.resize(frame,(320,240))
#     _frame = Image.fromarray(cv2.cvtColor(_frame,cv2.COLOR_BGR2RGB))
#     _frame = _frame.convert('RGB')
#     _frame = spatial_transform(_frame)
#     clip.append(_frame)
#     im_dim = clip[0].size()[-2:]
#     try:
#         test_data = torch.cat(clip, 0).view((opt.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)
#     except Exception as e:
#         pdb.set_trace()
#         raise e
#     inputs = torch.cat([test_data],0).view(1,3,opt.sample_duration,112,112)
#     num_frame += 1


#     ground_truth_array = np.zeros(opt.n_classes_clf + 1, )
#     with torch.no_grad():
#         inputs = Variable(inputs)
#         inputs_det = inputs[:, :, -opt.sample_duration_det:, :, :]
#         outputs_det = detector(inputs_det)
#         outputs_det = F.softmax(outputs_det, dim=1)
#         outputs_det = outputs_det.cpu().numpy()[0].reshape(-1, )
#         # enqueue the probabilities to the detector queue
#         myqueue_det.enqueue(outputs_det.tolist())

#         if opt.det_strategy == 'raw':
#             det_selected_queue = outputs_det
#         elif opt.det_strategy == 'median':
#             det_selected_queue = myqueue_det.median
#         elif opt.det_strategy == 'ma':
#             det_selected_queue = myqueue_det.ma
#         elif opt.det_strategy == 'ewma':
#             det_selected_queue = myqueue_det.ewma
#         # print('@@@@@@', det_selected_queue)
        
#         prediction_det = np.argmax(det_selected_queue)
#         # print('@@@@@@', prediction_det)

#         prob_det = det_selected_queue[prediction_det]
        
#         #### State of the detector is checked here as detector act as a switch for the classifier
#         # if prediction_det == 1:
#         if prediction_det == 1:
#             inputs_clf = inputs[:, :, :, :, :]
#             inputs_clf = torch.Tensor(inputs_clf.numpy()[:,:,::1,:,:])
#             outputs_clf = classifier(inputs_clf)
#             outputs_clf = F.softmax(outputs_clf, dim=1)
#             outputs_clf = outputs_clf.cpu().numpy()[0].reshape(-1, )
#             myqueue_clf.enqueue(outputs_clf.tolist())
#             passive_count = 0

#             if opt.clf_strategy == 'raw':
#                 clf_selected_queue = outputs_clf
#             elif opt.clf_strategy == 'median':
#                 clf_selected_queue = myqueue_clf.median
#             elif opt.clf_strategy == 'ma':
#                 clf_selected_queue = myqueue_clf.ma
#             elif opt.clf_strategy == 'ewma':
#                 clf_selected_queue = myqueue_clf.ewma

#         else:
#             outputs_clf = np.zeros(opt.n_classes_clf, )
#             # Push the probabilities to queue
#             myqueue_clf.enqueue(outputs_clf.tolist())
#             passive_count += 1
    
#     if passive_count >= opt.det_counter:
#         active = False
#     else:
#         active = True

#     # one of the following line need to be commented !!!!
#     if active:
#         active_index += 1
#         cum_sum = ((cum_sum * (active_index - 1)) + (weighting_func(active_index) * clf_selected_queue)) / active_index  # Weighted Aproach
#         #cum_sum = ((cum_sum * (active_index-1)) + (1.0 * clf_selected_queue))/active_index #Not Weighting Aproach
#         best3, best2, best1 = tuple(cum_sum.argsort()[-3:][::1])
#         # print('111111111111111111111111', cum_sum.argsort()[-3:][::1])
#         if float(cum_sum[best1] - cum_sum[best2]) > opt.clf_threshold_pre:
#             finished_prediction = True
#             pre_predict = True

#     else:
#         late_index = active_index
#         active_index = 0
#     if active == False and prev_active == True:
#         finished_prediction = True
#     elif active == True and prev_active == False:
#         finished_prediction = False
#     # print(finished_prediction,pre_predict)
    
#     if finished_prediction == True:
#         if cum_sum[best1] > opt.clf_threshold_final:
#             if pre_predict == True:
#                 if best1 != prev_best1:
#                     if cum_sum[best1] > opt.clf_threshold_final:
#                         results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
#                         # print('Early Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1],
#                         #                                                                       (
#                         #                                                                                   i * opt.stride_len) + opt.sample_duration_clf))
#                         print('Early Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1], active_index))
#                         # try:

#                         #     pass
#             else:
#                 if cum_sum[best1] > opt.clf_threshold_final:
#                     if best1 == prev_best1:
#                         if cum_sum[best1] > 5:
#                             results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
#                             print('Late Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1], late_index))
#                     else:
#                         results.append(((i * opt.stride_len) + opt.sample_duration_clf, best1))
#                         print('Late Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1], late_index))

#                     #     pass

#             finished_prediction = False
#             prev_best1 = best1
#         probability_best1 = cum_sum[best1]
#         cum_sum = np.zeros(opt.n_classes_clf, )
    
#     if active == False and prev_active == True:
#         pre_predict = False

#     prev_active = active

#     if len(results) != 0:
#         predicted = [np.array(results)[-1, 1]]
#         # print('################', results)
#         print('################', predicted)
#         prev_best1 = -1
#     else:
#         predicted = [83]

#     statement = (best1 == 82) & (best2 == 20) & (best3 == 22)
#     if (len(predicted) != 0) & ~statement:
#     # if (len(predicted) != 0):
#         try:
#             top1 = "Top-1: {}".format(idx_to_class[best1])
#             cv2.putText(frame, top1, (2, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (38, 0, 255), 1, cv2.LINE_AA)
#             top2 = "Top-2: {}".format(idx_to_class[best2])
#             cv2.putText(frame, top2, (2, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (38, 0, 255), 1, cv2.LINE_AA)
#             top3 = "Top-3: {}".format(idx_to_class[best3])
#             cv2.putText(frame, top3, (2, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (38, 0, 255), 1, cv2.LINE_AA)
#         except:
#             pass 
#     cv2.namedWindow("Result", 0)
#     cv2.imshow("Result", frame)

# if cv2.waitKey(1) & 0xFF == ord('q'):
#     print("You pressed q")
#     data = 'end'
#     # 對資料進行編碼格式轉換
#     data = data.encode('utf-8')
#     # 傳送data
#     client_rgb.sendall(data)
#     # client_tof.sendall(data)
#     client_rgb.close()
#     # client_tof.close()
#     sys.exit()
#     break
# cv2.destroyAllWindows()






































