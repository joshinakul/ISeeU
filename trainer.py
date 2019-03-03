import os
import cv2
import numpy 
from PIL import Image

recognizer=cv2.face.LBPHFaceRecognizer_create()
path='dataSet'

def imageId(path):
	
    folderpaths= [os.path.join(path,f) for f in os.listdir(path) if f!='.DS_Store'] #only use this condition if you are a mac user
    for folderpath in folderpaths:
        imagepath = [os.path.join(folderpath,g) for g in os.listdir(folderpath)]
    face=[]
    Id=[]
    for ipath in imagepath:
        faceimg=numpy.array(Image.open(ipath).convert('L'),'uint8') #8bit black-white image
        Ids=int(ipath.split('.')[1])
        face.append(faceimg)
        Id.append(Ids)
        cv2.imshow("trainer",faceimg)
        cv2.waitKey(3)
    return numpy.array(Id),face
ID,FACE=imageId(path)
recognizer.train(FACE,ID)
recognizer.save('recognizer/trainer.yml')
cv2.destroyAllWindows()