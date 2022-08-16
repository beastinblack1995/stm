import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional
import os
import pandas as pd
from datetime import  datetime
import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    WebRtcStreamerContext,
    webrtc_streamer,
)

##
import face_recognition as face_rec
import cv2
import shutil
path = 'employee images'
employeeImg = []

employeeName = []
myList = os.listdir(path)
filename = 'click'


def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)



def findEncoding(images) :
    imgEncodings = []
    for img in images :
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings
def MarkAttendence(name):
    with open('attendence.csv', 'r+') as f:
        myDatalist =  f.readlines()
        nameList = []
        for line in myDatalist :
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            f.writelines(f'\n{name}, {timestr}')
            statment = str('welcome to seasia' + name)

for cl in myList :
    curimg = cv2.imread(f'{path}/{cl}')
    employeeImg.append(curimg)
    employeeName.append(os.path.splitext(cl)[0])

EncodeList = findEncoding(employeeImg)





##


HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def main():
    st.header("Attendence_system")
    takepic()




def emprec(img):    
    
    facesInFrame = face_rec.face_locations(img)
    if len(facesInFrame) > 0:
        encodeFacesInFrame = face_rec.face_encodings(img, facesInFrame)


        for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame) :
            matches = face_rec.compare_faces(EncodeList, encodeFace)
            facedis = face_rec.face_distance(EncodeList, encodeFace)
            print(facedis)
            #if min(facedis) < 0.5:
            matchIndex = np.argmin(facedis)

            print(matchIndex)


            name = employeeName[matchIndex].upper()
#             y1, x2, y2, x1 = faceloc
#             y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            top, right, bottom, left = faceloc
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, name,  (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            print(name)        
            MarkAttendence(name)
            return img
    
    else:
        return img
         
    

    
    
    
    
    
def takepic():
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # To read image file buffer as a PIL Image:

        
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)       
        
        
        

        # To convert PIL Image to numpy array:
        img = np.array(cv2_img)
        img = emprec(img)
        st.image(img)
    
    



if __name__ == "__main__":
    import os



    main()
