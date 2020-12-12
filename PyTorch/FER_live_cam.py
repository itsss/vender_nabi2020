import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import time
from model import *

import pyaudio
import numpy as np

import socket
import time
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = '192.168.0.5'
port = 30001

total_score = 0
scene_number = 1

CHUNK = 4096 # number of data points to read at a time
RATE = 44100 # time resolution of the recording device (Hz)
sw = 0
duration = 0

def load_trained_model(model_path):
    model = Face_Emotion_CNN()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    return model


def FER_live_cam():
    TIMER = int(2)
    state = ""
    ps_state = '987654321'

    model = load_trained_model('./models/FER_trained_model.pt')
    
    emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
                    4: 'anger', 5: 'disguest', 6: 'fear'}


    val_transform = transforms.Compose([
        transforms.ToTensor()])

    cap = cv2.VideoCapture(0)
    prev = time.time()

    while TIMER >= 0:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
            X = resize_frame/256
            X = Image.fromarray((X))
            X = val_transform(X).unsqueeze(0)
            with torch.no_grad():
                model.eval()
                log_ps = model.cpu()(X)
                ps = torch.exp(log_ps)
                ps_state = ps
                top_p, top_class = ps.topk(1, dim=1)

                state = int(top_class.numpy())
                # print(int(top_class.numpy()))
                pred = emotion_dict[int(top_class.numpy())]
            cv2.putText(frame, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

        cv2.imshow('frame', frame)
        cur = time.time()
        # print("time: ", TIMER)
        if cur - prev >= 1:
            prev = cur
            TIMER = TIMER - 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return state, ps_state

def emotion_classification():
    global scene_number
    # {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disguest', 6: 'fear'}
    state, ps_state = FER_live_cam()
    if str(ps_state) == 987654321:
        state_string = str(scene_number)+",0,0,0"
    else:
        state_string = str(scene_number)+",0,0,"+str(ps_state)[9:-3]
    print(state_string)
    sock.sendto(state_string.encode(), (ip, port))
    if state == 1:
        return 1
    else:
        return -1

def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t

    avg = sum_num / len(num)
    return avg

def scene():
    print(str(scene_number)+", TTS OUTPUT")
    time.sleep(6)
    print(str(scene_number)+", TTS END")
    sw = 0

    p=pyaudio.PyAudio() # start the PyAudio class
    stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK) #uses default input device

    val = []
    value = 0
    # create a numpy array holding a single read of audio data
    while sw != 1: #to it a few times just to see
        data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
        
        for d in data:
            if abs(d) > 10000:
                print("START")
                duration = 0
                val.append(emotion_classification())
                val.append(emotion_classification())
                sw = 1
                break
        if sw == 1:
            break

    # close the stream gracefully
    stream.stop_stream()
    stream.close()
    p.terminate()

    value = cal_average(val)
    print(value)

    if value < 0:
        print(-1)
        return -1
    else:
        print(1)
        return 1

def eval():
    global total_score
    if total_score <= -2:
        print('매우부정')
        txt = str(scene_number)+",0,1"
        sock.sendto(txt.encode(), (ip, port))

    elif total_score < 0:
        print('부정')
        txt = str(scene_number)+",0,2"
        sock.sendto(txt.encode(), (ip, port))

    elif total_score >= 2:
        print('매우긍정')
        txt = str(scene_number)+",0,4"
        sock.sendto(txt.encode(), (ip, port))

    else:
        print('긍정')
        txt = str(scene_number)+",0,3"
        sock.sendto(txt.encode(), (ip, port))

if __name__ == "__main__":
    for i in range(5):
        total_score += scene()
        print(total_score)
        scene_number+=1
    eval()

# if __name__ == "__main__":
#     state = FER_live_cam()
#     print(state)



# class AvgMeter():
#     def __init__(self):
#         self.avg = 0
#         self.weight = 0.9
# 
#     def update(self, value):
#         self.avg = self.weight*self.avg + (1-self.weight)*value
#
#
# class Scene():
#     def __init__(self, scene_number):
#         self.score = AvgMeter()
#         self.scene_number = scene_number
#         self.state = False
#
#     def eval(self):
#         # calculate emotion scores
#         emotion_score = emotion_classification()
#         self.score.update(emotion_score)
#
#
#
# class Scene1(Scene):
#     def __init__(self):
#         super().__init__(scene_number=1)
#
#
#     def set_state(self, state):
#         self.state = state
#
#     def criterion(self):
#         # mask detection
#         if mask_score > 0:
#             return True
#
#
#     def eval(self):
#         raise NotImplementedError
#
#
#
#
# class Scene9(Scene):
#
#     def final(self):
#         return
#
#
#
#
# class SceneManager():
#     def __init__(self):
#         # self.scene1 = Scene1()
#         # self.scene2 = Scene2()
#         self.scenes = []
#         self.current_scene = None
#         self.current_scene_number = -1
#
#
#
#     def play(self):
#         score = self.current_scene.eval()
#         #UDP to TouchDesigner
#





