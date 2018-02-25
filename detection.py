import cv2
import pyttsx3

engine = pyttsx3.init()
def hi(name):
    engine.say('hi' + name + "\n\n please \n Sir\n the say the paassword")
    engine.runAndWait()

ids = {15103047: 'Vishal', 15103032: 'Chinmay', 15103031: 'Akash', 15103013: 'Ashish', 15103049: 'Rahul', 15103048: 'Prince', 15103093: 'Kovid', 15103027: 'Deepak', 15103083: 'Rohit', 151022: 'sanjana'}
det = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
recog = cv2.createLBPHFaceRecognizer()
recog.load('trainer/trainerdata.yml')
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL, 1, 1, 0, 1)
v = 0
while True:
    rect, img = cam.read()
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = det.detectMultiScale(grey, 1.3, 5)
    for (x, y, w, h) in faces:
        t = v
        k, conf = recog.predict(grey[y:y+h, x:x+h])
        v = k
        print(k, conf)
        if k != t:
            hi(ids[k])
        # cv2.cv.PutText(cv2.cv.fromarray(img), str(k), (x, y+h), font, 255)
        cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        cv2.cv.PutText(cv2.cv.fromarray(img), ids[k], (x, y + h), font, (2, 10, 255))
    cv2.imshow('win', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
      