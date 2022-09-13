import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    
    #analyze face using DeepFace lib
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    
    #face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.1,4)

#   Draw rectangle around face
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
       
    #using putText() for displaying dominant emotion from analysis    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, result['dominant_emotion'], (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_4)
        
    cv2.imshow('video',frame)
    
    if cv2.waitKey(1) == 27:
        break
    

cap.release()
cv2.destroyAllWindows()    