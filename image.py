import cv2
det = cv2.CascadeClassifier('haarcascade_defafrontalface_ult.xml')
cam = cv2.VideoCapture(0)
name = raw_input('enter rollno like : 1510...\n')
sample = 0
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL, 1, 1, 0, 1)
while True:
    rect, img = cam.read()
    #print img
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = det.detectMultiScale(grey, 1.3, 5)
    for (x, y, w, h) in faces:
        sample += 1
        cv2.imwrite('image/user.' + name + '.' + str(sample) + '.jpg', grey[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        cv2.imshow('frame', img)
        cv2.waitKey(1)
    if sample > 300:
        break

cam.release()
cv2.destroyAllWindows()
