import cv2
import sqlite3 as s
cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("myfacehaarcascade.xml")
model = cv2.face.LBPHFaceRecognizer_create()
model.read('recognizer/trainer.yml')
font=cv2.FONT_HERSHEY_DUPLEX
path="dataSet"

def getProfile(idx):
    
    con = s.connect('data_persons.db')
    cursor = con.cursor()
    c = cursor.execute('SELECT * FROM data WHERE ID='+str(idx))
    for row in c:
        profile_tuple = row
        return profile_tuple

while(True):
	# Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# Detect faces in the image
    faces = faceCascade.detectMultiScale(frame,1.5,5)


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x ,y), (x+w, y+h), (223, 200, 31), 2)
        idx,config=model.predict(gray[y:y+h,x:x+w])
        profile=getProfile(idx)
        if config > 65:
            print(profile)
        
        if profile!=None:
            cv2.rectangle(frame, (x ,y), (x+w, y+h), (123,100, 61), 2)
            cv2.putText(frame,profile[1],(x,y+h),font,1,(0,255,0))
			
	# Display the resulting frame
    cv2.imshow('Detection Required', frame)
    if cv2.waitKey(1) == 27:
		# When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
		