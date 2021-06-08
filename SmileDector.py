import cv2


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
lefteye_detector = cv2.CascadeClassifier('haarcascade_lefteye_2plits.xml')
righteye_detector = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
mouth_detector = cv2.CascadeClassifier('haarcascade_mouth.xml')
nose_detector = cv2.CascadeClassifier('haarcascade_nose.xml')




webcam = cv2.VideoCapture(0)


while True :
    (successful_frame_read, frame) = webcam.read()

    if not successful_frame_read :
        break
    
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(frame_grayscale, 1.3, 5)
    #smiles = smile_detector.detectMultiScale(frame_grayscale)

    #run face detection within each of those faces
    for (x,y,w,h) in faces :
        
        #draw rectangle around faces
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100, 200, 50), 4) 
        
        #getting sub fram using numpy N-dimensional array slicing 
        the_face = frame[y:y+h, x:x+w]
        
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor =1.7, minNeighbors = 20)
        """ 
        noses = nose_detector.detectMultiScale(face_grayscale, scaleFactor =1.7, minNeighbors = 20)
        righteyes = righteye_detector.detectMultiScale(face_grayscale, scaleFactor =1.7, minNeighbors = 20)
        lefteyes = lefteye_detector.detectMultiScale(face_grayscale, scaleFactor =1.7, minNeighbors = 20)
        mouths = mouth_detector.detectMultiScale(face_grayscale, scaleFactor =1.7, minNeighbors = 20) 
        """

        #for smile detection
        
        #find all smiles in faces
        for (x_, y_, w_, h_) in smiles :
        
            #drawing rectangles around smiles
            cv2.rectangle(the_face, (x_,y_), (x_+w_, y_+h_), (50, 50, 200), 4)  
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))
       
        """ eyeDistance = 0
        rightEyePoint = [0,0]
        lefteyePoint = [0,0]


        for (x_, y_, w_, h_) in righteyes :
        
            #drawing rectangles around righteyes
            cv2.rectangle(the_face, (x_,y_), (x_+w_, y_+h_), (50, 50, 200), 4) 
 
        #righteyePoint = [x_ + w_, y_ + h_]
        print('TEST')
        #print (righteyes)

        for (x_, y_, w_, h_) in lefteyes :
        
            #drawing rectangles around smiles
            cv2.rectangle(the_face, (x_,y_), (x_+w_, y_+h_), (50, 50, 200), 4) 

        lefteyePoint = [x_, y_ + h_] 
        if len(righteyes) > 0 and len(lefteyes) > 0 :
            pass

        while len(righteyes) > 0 and len(lefteyes) > 0 :
            pass

        
        if len(righteyes) > 0 and len(lefteyes) > 0 :
            pass
        
        #if all things are detected
        if len(righteyes) > 0 and len(lefteyes) > 0 and len(mouths) :
            pass

        if len(mouths) > 0:
            cv2.putText(frame, 'MOUTHS', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255)) """

    #display
    cv2.imshow('beauty-rater', frame)
    cv2.waitKey(1)

#cleaning up 
webcam.release()
cv2.destroyAllWindows() 