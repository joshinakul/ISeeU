import cv2
import sqlite3
cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("myfacehaarcascade.xml")
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainer.yml")
font=cv2.FONT_HERSHEY_DUPLEX
path="dataSet"

def getProfile(id):
	con=sqlite3.connect("data_persons.db")
	cmd="SELECT * FROM data WHERE ID="+str(id)
	cursor=con.execute(cmd)
	profile=None
	for row in cursor:
		profile=row
	con.close()
	return profile
while(True):
	# Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	# Detect faces in the image
    faces = faceCascade.detectMultiScale(frame,1.05,4)


    print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
    flag=0
    i=""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x ,y), (x+w, y+h), (223, 200, 31), 2)
        idx,config=rec.predict(gray[y:y+h,x:x+w])
        if config<85:
            profile=getProfile(idx)
        
        if profile!=None:
            cv2.rectangle(frame, (x ,y), (x+w, y+h), (223, 200, 31), 2)
            cv2.putText(frame,str(profile[1]),(x,y+h),font,1,(0,255,0))
			
	# Display the resulting frame
    cv2.imshow('Detection Required', frame)
    if flag==1 or cv2.waitKey(10) == 27:
		# When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
		