import cv2
import face_recognition
from face_recognition.api import face_locations
import numpy as np

imgrdj = face_recognition.load_image_file('facereg/images/rdj.jpg')
imgrdj = cv2.cvtColor(imgrdj,cv2.COLOR_BGR2RGB)

imgtest = face_recognition.load_image_file('facereg/images/vj.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgrdj)[0]
encoderdj = face_recognition.face_encodings(imgrdj)[0]
cv2.rectangle(imgrdj,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encoderdj],encodetest)
facedis = face_recognition.face_distance([encoderdj],encodetest)
print(results,facedis)
cv2.putText(imgtest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('RDJ',imgrdj)
cv2.imshow('RDJ TEST',imgtest)

cv2.waitKey(0)
