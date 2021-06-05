import cv2


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')


webcam = cv2.VideoCapture(0)


while True :
    (successful_frame_read, frame) = webcam.read()

    if not successful_frame_read :
        break
    
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(frame_grayscale, 1.3, 5)
    #smiles = smile_detector.detectMultiScale(frame_grayscale)

    for (x,y,w,h) in faces :

        cv2.rectangle(frame, (x,y), (x+w, y+h), (100, 200, 50), 4) 
        
        face_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        the_face = frame[y:y+h, x:x+h]


        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor =1.7, minNeighbors = 20)

        for (x_, y_, w_, h_) in smiles :
        
            #drawing rectangles around smiles
            cv2.rectangle(frame, (x_,y_), (x_+w_, y_+h_), (50, 50, 200), 4) 


   

    #display
    cv2.imshow('smile-detector', frame)
    cv2.waitKey(1)

#cleaning up 
webcam.release()
cv2.destroyAllWindows() 