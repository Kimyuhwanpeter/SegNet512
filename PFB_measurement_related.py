# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import f1_score, recall_score

# 안녕하세요!!  지난번에 이어서 좋은 글 잘 읽었습니다.  mIoU 계산시에, true label에서는 등장하지 않고 predict
# label에서는 등장하는 label의 경우는 iou를 0으로 두고 계산을 해주나요?
# 안녕하세요.네 맞습니다.  간단히 교집합이 없으니 0이 되는 원리 입니다.  true label에는 없으나 predict에만 존재한다는 것
# 자체가 성능이 나쁜것이니 0에 수렴해야 하는 논리와 동일합니다.

# 저 위의 논리대라면, 나누
# 아!!!  predict한 이미지에 void부분을 11로 만들고!  miou를 구할 때, void클래스 빈도수는 제외하고 진행하면 되지
# 않을까?  기억해!!!!!  지금생각났음!!!!!!!!!!!!!
# 왜냐면!  어차피 predict와 label 위치에 같은 값이 동일하게 있다면 confusion metrice 할때 중앙성분만있음!
# 그렇기에 cm을 구한 뒤 대각성분을 추출한 뒤 맨 뒤에있는 void라벨을 제거하면 됨!  오키!!  내일 다시한번더
# 생각천천히해봐!!!기억해 꼭해!!!!!!!!!!!
class Measurement:
    def __init__(self, predict, label, shape, total_classes):
        self.predict = predict
        self.label = label
        self.total_classes = total_classes
        self.shape = shape

    def MIOU(self):

        self.predict = np.reshape(self.predict, self.shape)
        self.label = np.reshape(self.label, self.shape)

        predict_count = np.bincount(self.predict, minlength=self.total_classes)
        label_count = np.bincount(self.label, minlength=self.total_classes)
         
        temp = self.total_classes * np.array(self.label, dtype="int") + np.array(self.predict, dtype="int")  # Get category metrics
    
        temp_count = np.bincount(temp, minlength=self.total_classes * self.total_classes)
        cm = np.reshape(temp_count, [self.total_classes, self.total_classes])
        cm = np.diag(cm)
    
        U = label_count + predict_count - cm
        U = np.delete(U, -1)
        cm_ = np.delete(cm, -1)

        out = np.zeros((2))
        miou = np.divide(cm_, U + 1e-7)
        crop_iou = miou[0]
        weed_iou = miou[1]
        miou = np.nanmean(miou)
        

        if weed_iou == float('NaN'):
            weed_iou = 0.
        if crop_iou == float('NaN'):
            crop_iou = 0.

        return miou, crop_iou, weed_iou

    def F1_score_and_recall(self):  # recall - sensitivity

        # 1일 때의 predict와 label 가지고오기 (TP, FP)
        self.predict_positive = np.where(self.predict == 1, 1, 0)
        self.label_positive = np.where(self.label == 1, 1, 0)
        self.predict_positive = np.reshape(self.predict_positive, self.shape)
        self.label_positive = np.reshape(self.label_positive, self.shape)

        TP_func = lambda predict: predict[:] == 1
        TP_func2 = lambda predict, label: predict[:] == label[:]
        TP = np.where(TP_func(self.predict_positive) & TP_func2(self.predict_positive, self.label_positive), 1, 0)
        TP = np.sum(TP, dtype=np.int32)

        FP_func = lambda predict: predict[:] == 1
        FP_func2 = lambda predict, label: predict[:] != label[:]
        FP = np.where(FP_func(self.predict_positive) & FP_func2(self.predict_positive, self.label_positive), 1, 0)
        FP = np.sum(FP, dtype=np.int32)

        # 0일 때의 predict와 label 가져오기 (TN, FN)
        self.predict_negative = np.where(self.predict == 0, 0, 1)
        self.label_negative = np.where(self.label == 0, 0, 1)
        self.predict_negative = np.reshape(self.predict_negative, self.shape)
        self.label_negative = np.reshape(self.label_negative, self.shape)

        TN_func = lambda predict: predict[:] == 0
        TN_func2 = lambda predict, label: predict[:] == label[:]
        TN = np.where(TN_func(self.predict_negative) & TN_func2(self.predict_negative, self.label_negative), 1, 0)
        TN = np.sum(TN, dtype=np.int32)

        FN_func = lambda predict: predict[:] == 0
        FN_func2 = lambda predict, label: predict[:] != label[:]
        FN = np.where(FN_func(self.predict_negative) & FN_func2(self.predict_negative, self.label_negative), 1, 0)
        FN = np.sum(FN, dtype=np.int32)


        TP_FP = (TP + FP) + 1e-7

        TP_FN = (TP + FN) + 1e-7

        out = np.zeros((1))
        Precision = np.divide(TP, TP_FP)
        Recall = np.divide(TP, TP_FN)

        Pre_Re = (Precision + Recall) + 1e-7

        F1_score = np.divide(2. * (Precision * Recall), Pre_Re)

        return F1_score, Recall

    def TDR(self): # True detection rate

        # 1일 때의 predict와 label 가지고오기 (TP, FP)
        self.predict_positive = np.where(self.predict == 1, 1, 0)
        self.label_positive = np.where(self.label == 1, 1, 0)
        self.predict_positive = np.reshape(self.predict_positive, self.shape)
        self.label_positive = np.reshape(self.label_positive, self.shape)

        TP_func = lambda predict: predict[:] == 1
        TP_func2 = lambda predict, label: predict[:] == label[:]
        TP = np.where(TP_func(self.predict_positive) & TP_func2(self.predict_positive, self.label_positive), 1, 0)
        TP = np.sum(TP, dtype=np.int32)

        FP_func = lambda predict: predict[:] == 1
        FP_func2 = lambda predict, label: predict[:] != label[:]
        FP = np.where(FP_func(self.predict_positive) & FP_func2(self.predict_positive, self.label_positive), 1, 0)
        FP = np.sum(FP, dtype=np.int32)

        # 0일 때의 predict와 label 가져오기 (TN, FN)
        self.predict_negative = np.where(self.predict == 0, 0, 1)
        self.label_negative = np.where(self.label == 0, 0, 1)
        self.predict_negative = np.reshape(self.predict_negative, self.shape)
        self.label_negative = np.reshape(self.label_negative, self.shape)

        TN_func = lambda predict: predict[:] == 0
        TN_func2 = lambda predict, label: predict[:] == label[:]
        TN = np.where(TN_func(self.predict_negative) & TN_func2(self.predict_negative, self.label_negative), 1, 0)
        TN = np.sum(TN, dtype=np.int32)

        FN_func = lambda predict: predict[:] == 0
        FN_func2 = lambda predict, label: predict[:] != label[:]
        FN = np.where(FN_func(self.predict_negative) & FN_func2(self.predict_negative, self.label_negative), 1, 0)
        FN = np.sum(FN, dtype=np.int32)

        TP_FP = (TP + FP) + 1e-7

        out = np.zeros((1))
        TDR = np.divide(FP, TP_FP)

        TDR = 1 - TDR

        return TDR

    def show_confusion(self):
        # TP - Red [255, 0, 0] 10, TN - 하늘색 [0, 255, 255] 20, FP - 분홍색 [255, 0, 255] 30, FN - 노랑 [255, 255, 0] 40

        self.predict = np.squeeze(self.predict, -1)

        color_map = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [236, 184, 199]])
        TP_func = lambda func:func[:] == 1
        output = np.where(TP_func(self.predict) & TP_func(self.label), 10, 0)  # get TP

        TN_func = lambda func:func[:] == 0
        output = np.where(TN_func(self.predict) & TN_func(self.label), 20, output)  # get TN
        
        FP_func = lambda func:func[:] == 1
        FP_func2 = lambda func:func[:] == 0
        FP_func3 = lambda func:func[:] == 2
        output = np.where(FP_func(self.predict) & (FP_func2(self.label)|FP_func3(self.label)), 30, output)  # get FP

        FN_func = lambda func:func[:] == 0
        FN_func2 = lambda func:func[:] == 1
        FN_func3 = lambda func:func[:] == 2
        output = np.where(FN_func(self.predict) & (FN_func2(self.label)|FN_func3(self.label)), 40, output)  # get FN

        output = np.expand_dims(output, -1)
        output_3D = np.concatenate([output, output, output], -1)
        temp_output_3D = output_3D
        
        output_3D = np.where(output == np.array([0, 0, 0], dtype=np.uint8), np.array([0, 0, 0], dtype=np.uint8), output_3D).astype(np.uint8)
        output_3D = np.where(output == np.array([10, 10, 10], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), output_3D).astype(np.uint8)
        output_3D = np.where(output == np.array([20, 20, 20], dtype=np.uint8), np.array([0, 255, 255], dtype=np.uint8), output_3D).astype(np.uint8)
        output_3D = np.where(output == np.array([30, 30, 30], dtype=np.uint8), np.array([255, 0, 255], dtype=np.uint8), output_3D).astype(np.uint8)
        output_3D = np.where(output == np.array([40, 40, 40], dtype=np.uint8), np.array([255, 255, 0], dtype=np.uint8), output_3D).astype(np.uint8)


        return output_3D, temp_output_3D

#import matplotlib.pyplot as plt

#if __name__ == "__main__":

    
#    path = os.listdir("D:/[1]DB/[5]4th_paper_DB/other/CamVidtwofold_gray/CamVidtwofold_gray/train/labels")

#    b_buf = []
#    for i in range(len(path)):
#        img = tf.io.read_file("D:/[1]DB/[5]4th_paper_DB/other/CamVidtwofold_gray/CamVidtwofold_gray/train/labels/"+ path[i])
#        img = tf.image.decode_png(img, 1)
#        img = tf.image.resize(img, [513, 513], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#        img = tf.image.convert_image_dtype(img, tf.uint8)
#        img = tf.squeeze(img, -1)
#        #plt.imshow(img, cmap="gray")
#        #plt.show()
#        img = img.numpy()
#        a = np.reshape(img, [513*513, ])
#        print(np.max(a))
#        img = np.array(img, dtype=np.int32) # void클래스가 정말 12 인지 확인해봐야함
#        #img = np.where(img == 0, 255, img)

#        b = np.bincount(np.reshape(img, [img.shape[0]*img.shape[1],]))
#        b_buf.append(len(b))
#        total_classes = len(b)  # 현재 124가 가장 많은 클래스수

#        #miou = MIOU(predict=img, label=img, total_classes=total_classes, shape=[img.shape[0]*img.shape[1],])
#        miou_ = Measurement(predict=img,
#                            label=img, 
#                            shape=[513*513, ], 
#                            total_classes=12).MIOU()
#        print(miou_)

#    print(np.max(b_buf))
