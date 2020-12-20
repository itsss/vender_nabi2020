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

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

import serial
import syslog
import random

import socket
import time
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ip = '192.168.0.8'
port = 30001

import threading


CHUNK = 4096 # number of data points to read at a time
RATE = 44100 # time resolution of the recording device (Hz)

total_score = 0
scene_number = 1
status = 9999
mask = 0
sw = 0
duration = 0
coin = 0
judgement = 0
sub_scene=0

storage = [999999,999999,999999,999999,999999,999999]

portar = '/dev/ttyACM0'
ard = serial.Serial(portar,9600,timeout=5)
time.sleep(2)

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

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

def mask_detection():
    decision = 0
    TIMER = int(2)
    # construct the argument parser and parse the arguments

    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
    weightsPath = os.path.sep.join([args["face"],
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    maskNet = load_model(args["model"])

    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    prev = time.time()
    # loop over the frames from the video stream

    while TIMER >= 0:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            # label = "Mask" if mask > withoutMask else "No Mask"

            if mask > withoutMask:
                label = "Mask"
                decision = 0
            else:
                label = "No Mask"
                decision = 1
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
#            cv2.putText(frame, label, (startX, startY - 10),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
#            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        # cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        cur = time.time()
        # if the `q` key was pressed, break from the loop
        if cur - prev >= 1:
            prev = cur
            TIMER = TIMER - 1
        if key == ord("q"):
            break

    # do a bit of cleanup
    # cv2.destroyAllWindows()
    vs.stop()
    return decision

def socket_communication(val):
    sock.sendto(val.encode(), (ip, port))

def emotion_classification():
    global scene_number
    # {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disguest', 6: 'fear'}
    state, ps_state = FER_live_cam()
    if str(ps_state) == 987654321:
        state_string = str(scene_number)+","+str(scene_number*10+sub_scene)+",0,0,0"
    else:
        state_string = str(scene_number)+","+str(scene_number*10+sub_scene)+",0,0,0"+str(ps_state)[9:-3]
    print(state_string)
    socket_communication(state_string)
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
    global sub_scene
    if int(scene_number) != 3:
        print(str(scene_number)+", TTS OUTPUT")
        time.sleep(1)
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
            if abs(d) > 5000 or scene_number == 3:
                print("START")
                duration = 0
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
        print(str(scene_number)+","+str(scene_number*10+2)+",0")
        sub_scene=2
        # socket_communication(str(scene_number)+",999999999")
        socket_communication(str(scene_number)+","+str(scene_number*10+2)+",0")
        time.sleep(1)
        return -1
    else:
        print(str(scene_number)+","+str(scene_number*10+1)+",0")
        sub_scene=1
        # socket_communication(str(scene_number)+",999999999")
        socket_communication(str(scene_number)+","+str(scene_number*10+2)+",0")
        time.sleep(1)
        return 1

def eval(judge):
    global total_score, judgement, sub_scene
    sum_storage = 0
    for i in range(1,6):
       sum_storage += storage[i]
    if(sum_storage <= 0):
        print("NO FOOD. PLEASE REFILL")
        exit()
    if total_score <= -2:
        if(judge==1): 
            judgement=4
            print('매우부정')
            for i in range(0):
                for j in range(1,6):
                    r = random.randint(0,5)
                    if(storage[r] <= 0):
                        continue
                    else: 
                        ard.flush()
                        ard.write(str.encode(str(r)))
                        time.sleep(2)
                        #ard.write([r])
                        storage[r]-=1
                        break
            # ard.flush()
            # ard.write()
        sub_scene=4
        txt = str(scene_number)+","+str(scene_number*10+sub_scene)+",0,"+str(judgement)
        socket_communication(txt)

    elif total_score < 0:
        if(judge==1): 
            print('부정')
            judgement=3
            for i in range(1):
                for j in range(1,6):
                    r = random.randint(0,5)
                    if(storage[r] <= 0):
                        continue
                    else: 
                        ard.flush()
                        ard.write(str.encode(str(r)))
                        time.sleep(2)
                        #ard.write([r])
                        storage[r]-=1
                        break
        sub_scene=3
        txt = str(scene_number)+","+str(scene_number*10+sub_scene)+",0,"+str(judgement)
        socket_communication(txt)

    elif total_score >= 2:
        if(judge==1): 
            print('매우긍정')
            judgement=1
            for i in range(2):
                for j in range(1,6):
                    r = random.randint(0,5)
                    if(storage[r] <= 0):
                        continue
                    else: 
                        ard.flush()
                        ard.write(str.encode(str(r)))
                        time.sleep(2)
                        #ard.write([r])
                        storage[r]-=1
                        break
        sub_scene=1
        txt = str(scene_number)+","+str(scene_number*10+sub_scene)+",0,"+str(judgement)
        socket_communication(txt)

    else:
        if(judge==1): 
            print('긍정')
            judgement=2
            for i in range(2):
                for j in range(1,6):
                    r = random.randint(0,5)
                    if(storage[r] <= 0):
                        continue
                    else: 
                        ard.flush()
                        ard.write(str.encode(str(r)))
                        time.sleep(2)
                        #ard.write([r])
                        storage[r]-=1
                        break
        sub_scene=2
        txt = str(scene_number)+","+str(scene_number*10+sub_scene)+",0,"+str(judgement)
        socket_communication(txt)

def coin_detect():
    global coin, status, sub_scene
    i = 0
    coin = 0

    while True:
        msg = ard.read()
        # print ("Message from arduino: ")
        mess = str(msg)[2:3]
        if(mess != "r" and mess != "n" and mess != "\\"):
            # print (mess)
            if(mess.isdigit()):
                status = int(mess)
                socket_communication(str(scene_number)+","+str(scene_number*10+sub_scene)+","+str(mask)+","+str(judgement)+","+mess)
            if(mess == "I"):
                coin = 1

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                    default="face_detector",
                    help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                    default="mask_detector.model",
                    help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

if __name__ == "__main__":
    # 기기 앞에 사람 있는지 감지 필요 (1)
    
    my_thread = threading.Thread(target=coin_detect)
    my_thread.start()
    while True:
        total_score = 0
        scene_number = 1
        status = 9999
        sw = 0
        duration = 0
        coin = 0
        mask = 1
        judgement = 0
        sub_scene=0

        sub_scene=1
        socket_communication("1,11")
        print(status)
        time.sleep(0.01)
        if(status <= 7): # ultrasonic sensor
            sub_scene=2
            socket_communication("2,21")
            print(str(scene_number)+ ", TTS WAIT")
            time.sleep(0.5)
            scene_number += 1
            # Mask Detection (2)
            sub_scene=1
            mask = mask_detection()
            if mask == 0:
                print("마스크 착용으로 진행 불가")
                sub_scene=2
                socket_communication("2,22,1")
                time.sleep(4)
            else:
                print("continue")
                socket_communication("3,31,0")
                time.sleep(0.5)
                for i in range(2): # (3,4)
                    scene_number+=1
                    sub_scene=0
                    total_score += scene()
                    #total_score += 1
                    print(total_score)
                time.sleep(4)
                scene_number+=1
                sub_scene=0
                eval(0) # 금액 제시 및 최종 판정 (5,6)
                print(str(scene_number)+ ", TTS WAIT")
                time.sleep(5)
                coin = 0
                sec = 0
                
                while True:
                    print("INSERT COIN")
                    print(coin)
                    cur = time.time()
                    time.sleep(1)
                    sec += 1
                    if(sec > 15):
                        break
                    if(coin >= 1):
                        break
                if(coin == 0):
                    scene_number = 6
                    sub_scene=5
                    socket_communication(str(scene_number)+","+str(scene_number*10+sub_scene)+","+str(mask))
                    time.sleep(1)
                else:
                    scene_number+=1
                    eval(1)
                    print(str(scene_number)+ ", TTS WAIT")
                    time.sleep(5)

                scene_number=7
                sub_scene=0
                socket_communication(str(scene_number) + "," +str(scene_number*10+sub_scene) + "," + str(mask) + "," + str(judgement))
                print("RESTART, PLEASE WAIT 10 SECONDS")
                time.sleep(10)

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





