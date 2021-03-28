from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import cv2
import dlib
from tkinter import Tk, Frame, Label, Button
from time import sleep
import pandas as pd
import datetime
import smtplib
from email.message import EmailMessage
from email.utils import formatdate
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
import mimetypes
import os

net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

data_dict={"Time":[],"Roll No":[],"Attention":[],"Transcribing":[],"Yawning":[],"Drowsing":[],"Face":[],"Test":[]}



def send_mail_with_excel(recipient_email, subject, excel_file,ROLL):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = SEMAIL
    msg['To'] = recipient_email
    msg['Date'] = formatdate(localtime = True)

    with open(excel_file, 'rb') as f:
        file_data = f.read()
    msg.add_attachment(file_data, maintype="application", subtype="xlsx", filename=excel_file)
    attachment_path = ROLL+".png"
    attachment_filename = os.path.basename(attachment_path)
    mime_type, _ = mimetypes.guess_type(attachment_path)
    mime_type, mime_subtype = mime_type.split('/', 1)
    with open(attachment_path, 'rb') as ap:
        msg.add_attachment(ap.read(), maintype=mime_type, subtype=mime_subtype,filename=attachment_filename)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(SEMAIL, PASSWORD)
        smtp.send_message(msg)


def send_mail_with_png(recipient_email, subject, ROLL):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = SEMAIL
    msg['To'] = recipient_email
    msg['Date'] = formatdate(localtime = True)

    attachment_path = ROLL+".png"
    attachment_filename = os.path.basename(attachment_path)
    mime_type, _ = mimetypes.guess_type(attachment_path)
    mime_type, mime_subtype = mime_type.split('/', 1)
    with open(attachment_path, 'rb') as ap:
        msg.add_attachment(ap.read(), maintype=mime_type, subtype=mime_subtype,filename=attachment_filename)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(SEMAIL, PASSWORD)
        smtp.send_message(msg)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50, 53):
        top_lip_pts.append(landmarks[i])
    for i in range(61, 64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:, 1])


def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:, 1])


def mouth_open(image):
    landmarks = get_landmarks(image)

    if landmarks == "error":
        return image, 0
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return lip_distance



tkWindow = Tk()
tkWindow.geometry('250x100')
tkWindow.title('Welcome')

rollLabel = Label(tkWindow,text="Roll No (eg-A749)").grid(row=0, column=0)
roll = StringVar()
rollEntry = Entry(tkWindow, textvariable=roll).grid(row=0, column=1)


passwordLabel = Label(tkWindow,text="Password").grid(row=1, column=0)
password = StringVar()
passwordEntry = Entry(tkWindow, textvariable=password, show='*').grid(row=1, column=1)


NAME=""
ROLL=""
SEMAIL=""
PASSWORD=""
TEMAIL=""
def close_window ():
    global NAME,ROLL,SEMAIL,TEMAIL,PASSWORD
    f = open("student_db.txt", "r")
    db=f.read()
    l=db.split("\n")
    dict={}
    for i in l:
        dict[i.split(" ")[0]]=i.split(" ")[1::]


    ROLL=roll.get()

    rollpass=password.get()
    #TEMAIL="rgitstudent.attendance@gmail.com"
    TEMAIL="khenir99@gmail.com"
    if(ROLL in dict and dict[ROLL][0]==rollpass):
        NAME=dict[ROLL][1]
        SEMAIL=dict[ROLL][2]
        PASSWORD=dict[ROLL][3]
        tkWindow.destroy()
    else:
        msg="Invalid Login Credentials"
        popup = Tk()
        popup.wm_title("!")
        label = ttk.Label(popup, text=msg, font=("Helvetica", 10))
        label.pack(side="top", fill="x", pady=10)
        B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
        B1.pack()
        popup.mainloop()

startButton = Button(tkWindow, text="Start", command=close_window).grid(row=5, column=0)

tkWindow.mainloop()
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 1
eye_thresh = 15
initBB = None
COUNTER = 0
drowsy = 0
alarm = "No"
TOTAL = 0
yawns = 0
yawn_status = False

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


vs = VideoStream(src=0).start()
time.sleep(1.0)


face = 0
flag = 1
f=0
while True:
    frame = vs.read()
    now = datetime.datetime.now()
    d_drows="Not Drowsing"
    d_yawn="Not Yawning"
    d_atten="Not Paying Attention"
    d_tran="Not Transcribing"
    if not face:
        d_face="Not Detected"
        d_atten="None"
        d_tran="None"
        d_drows="None"
        d_yawn="None"
        #tracker = cv2.TrackerKCF_create()
        tracker = cv2.TrackerCSRT_create()

        frame = imutils.resize(frame, width=400)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(
            frame, (299, 299)), 1.0, (299, 299), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()
        frame = imutils.resize(frame, width=500)
        (H, W) = frame.shape[:2]
        ch = None
        TOTAL = 0
        count = 0
        ear = 0.0
        yawns = 0
        drowsy = 0
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence >= 0.8:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                y = startY - 10 if startY - 10 > 10 else startY + 10
                if flag:
                    flag = 0
                    X = startX
                    Y = startY
                sx = startX + 35
                sy = startY + 65
                ex = endX - 110
                ey = endY - 180
                if sx >= 0 and sy >= 0 and ey >= 0 and ex >= 0 and sx <= 380 and ex <= 280:
                    face = 1
                    initBB = (sx, sy, ex, ey)
                    tracker.init(frame, initBB)
                else:
                    time.sleep(0.6)
    else:
        d_face="Detected"
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        d_drows="Not Drowsing"
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                drowsy += 1
                if drowsy >= eye_thresh:
                    d_drows="Drowsing"
                    if alarm == "No":
                        alarm = "Yes"
                        print("Drowsing")
                        ch = None

            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                d_drows="Not Drowsing"

                COUNTER = 0
                drowsy = 0
                alarm = "No"

        lip_distance = mouth_open(frame)

        prev_yawn_status = yawn_status

        if isinstance(lip_distance, int):
            if lip_distance > 25:
                ch = None
                d_yawn="Yawning"
                yawn_status = True

            else:
                yawn_status = False
                d_yawn="Not Yawning"


        if prev_yawn_status == True and yawn_status == False:
            yawns += 1

        (success, box) = tracker.update(frame)
        if success:
            face = 1
            (x, y, w, h) = [int(v) for v in box]
            Xch = X - x + 35
            Ych = Y - y + 65

            if Xch > 45:
                d_tran="Not Transcribing"
                d_atten="Not Paying Attention"
                if ch != "left":
                    print("left")
                cv2.rectangle(frame, (x - 35, y - 65),
                              (x + w + 50, y + h + 80), (0, 0, 255), 2)
                ch = "left"
                count = 0
            elif Xch < -45:
                d_tran="Not Transcribing"
                d_atten="Not Paying Attention"
                if ch != "right":
                    print("right")
                cv2.rectangle(frame, (x - 35, y - 65),
                              (x + w + 50, y + h + 80), (0, 0, 255), 2)
                ch = "right"
                count = 0
            elif Ych > 35:
                d_tran="Not Transcribing"
                d_atten="Not Paying Attention"
                if ch != "up":
                    print("up")
                cv2.rectangle(frame, (x - 35, y - 65),
                              (x + w + 50, y + h + 80), (0, 0, 255), 2)
                ch = "up"
                count = 0
            elif Ych < -35:
                d_tran="Transcribing"
                d_atten="Paying Attention"
                if ch != "down":
                    print("down")
                cv2.rectangle(frame, (x - 35, y - 65),
                              (x + w + 50, y + h + 80), (0, 0, 255), 2)
                ch = "down"
                count = 0

            else:
                d_tran="Not Transcribing"
                d_atten="Not Paying Attention"
                cv2.rectangle(frame, (x - 35, y - 65),(x + w + 50, y + h + 80), (255, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if abs(Xch) < 45 and abs(Ych) < 35:
                ch = "front"
                count += 1
                if count >= 30:
                    d_tran="Not Transcribing"
                    d_atten="Paying Attention"
                    cv2.rectangle(frame, (x - 35, y - 65),
                                  (x + w + 50, y + h + 80), (0, 255, 0), 2)
                    if count == 60:
                        print("stable")
                    ch = "stable"

        else:
            time.sleep(0.6)
            face = 0
    info = [
        ("Face detected", "Yes" if face else "No"),
        ("Direction", ch),
        ("Blinks", TOTAL if ch != None else ch),
        ("Drowsiness", alarm if ch != None else ch),
        ("EAR", "{:.2f}".format(ear) if ch != None else ch),
        ("Yawn", "Yes" if yawn_status else "No"),
        ("Yawn count", yawns)

    ]

    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF


    if key==ord("w"):
        class Question:
            def __init__(self, question, answers, correctLetter):
                self.question = question
                self.answers = answers
                self.correctLetter = correctLetter

            def check(self, letter, view):
                global right
                if(letter == self.correctLetter):
                    label = Label(view, text="Right!")
                    right += 1
                else:
                    label = Label(view, text="Wrong!")
                label.pack()
                view.after(1000, lambda *args: self.unpackView(view))


            def getView(self, window):
                view = Frame(window)
                Label(view, text=self.question).pack()
                Button(view, text=self.answers[0], command=lambda *args: self.check("A", view)).pack()
                Button(view, text=self.answers[1], command=lambda *args: self.check("B", view)).pack()
                Button(view, text=self.answers[2], command=lambda *args: self.check("C", view)).pack()
                Button(view, text=self.answers[3], command=lambda *args: self.check("D", view)).pack()
                return view

            def unpackView(self, view):
                view.pack_forget()
                askQuestion()

        def askQuestion():
            global questions, window, index, button, right, number_of_questions
            if(len(questions) == index + 1):
                Label(window, text="Thank you for answering the questions. " + str(right) + " of " + str(number_of_questions) + " questions answered right").pack()
                return
            button.pack_forget()
            index += 1
            questions[index].getView(window).pack()

        questions = []
        file = open("C:\\Users\\Rushabh\\Downloads\\questions.txt", "r")
        line = file.readline()
        while(line != ""):
            questionString = line
            answers = []
            for i in range (4):
                answers.append(file.readline())

            correctLetter = file.readline()
            correctLetter = correctLetter[:-1]
            questions.append(Question(questionString, answers, correctLetter))
            line = file.readline()
        file.close()
        index = -1
        right = 0
        number_of_questions = len(questions)

        window = Tk()
        window.geometry("400x250")
        button = Button(window, text="Start", command=askQuestion)
        button.pack()
        window.mainloop()
        d_test=str(right) + " of " + str(number_of_questions) + " questions answered right"

    else:
        d_test=""
    if f:
        data_dict["Test"].append(d_test)
        data_dict["Drowsing"].append(d_drows)
        data_dict["Yawning"].append(d_yawn)
        data_dict["Transcribing"].append(d_tran)
        data_dict["Face"].append(d_face)
        data_dict["Attention"].append(d_atten)
        data_dict["Time"].append(now.strftime("%H:%M:%S"))
        data_dict["Roll No"].append(ROLL)
    f=1
    if key == ord("q"):
        break
vs.stop()
total=len(data_dict["Attention"])
data = {'': ['Not Paying Attention','Yawning','Drowsing','Transcribing'],
        'Percent': [data_dict["Attention"].count("Not Paying Attention")/total*100,data_dict["Yawning"].count("Yawning")/total*100,
        data_dict["Drowsing"].count("Drowsing")/total*100,data_dict["Transcribing"].count("Transcribing")/total*100]
       }
df = pd.DataFrame(data,columns=['','Percent'])
df.plot(x ='', y='Percent', kind = 'bar')
fig=plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.savefig(ROLL+".png", dpi=100)
prev=(data_dict["Test"][0],data_dict["Drowsing"][0],data_dict["Yawning"][0],data_dict["Transcribing"][0],data_dict["Face"][0],data_dict["Attention"][0])
ptime=data_dict["Time"][0]
i=0
while i < len(data_dict["Test"]):
    new=(data_dict["Test"][i],data_dict["Drowsing"][i],data_dict["Yawning"][i],data_dict["Transcribing"][i],data_dict["Face"][i],data_dict["Attention"][i])
    ntime=data_dict["Time"][i]
    if prev==new:
        data_dict["Test"].pop(i)
        data_dict["Drowsing"].pop(i)
        data_dict["Yawning"].pop(i)
        data_dict["Transcribing"].pop(i)
        data_dict["Face"].pop(i)
        data_dict["Attention"].pop(i)
        data_dict["Time"].pop(i)
        data_dict["Roll No"].pop(i)
    else:
        prev=new
        data_dict["Time"][i]=ptime+"-"+ntime
        ptime=ntime
        i=i+1
    if i>=len(data_dict["Test"]):
        break


df = pd.DataFrame(data_dict)
df.to_csv(ROLL+".csv")
subject=ROLL+NAME
excel_file=ROLL+".csv"
send_mail_with_excel(TEMAIL, subject, excel_file,ROLL)
cv2.destroyAllWindows()
