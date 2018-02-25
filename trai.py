import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as ptl

reco = cv2.createLBPHFaceRecognizer()
detct = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
path = 'image'


def getimage(path):
    imgpath = [os.path.join(path, n) for n in os.listdir(path)]
    sampface = []
    name = []
    for impath in imgpath:
        Pilimg = Image.open(impath).convert('L')
        imagenp = np.array(Pilimg, np.uint8)
        id = int(os.path.split(impath)[-1].split('.')[1])
        faces = detct.detectMultiScale(imagenp)
        for (x, y, w, h) in faces:
            sampface.append(imagenp[y:y + h, x:x + w])
            print id
            name.append(id)
    return sampface, name


face, name1 = getimage('image')
reco.train(face, np.array(name1))
reco.save('trainer/trainerdata.yml')
cv2.destroyAllWindows()
