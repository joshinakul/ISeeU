import cv2
import sqlite3
import os

cap = cv2.VideoCapture(0)
# Create the haar cascade
faceCascade = cv2.CascadeClassifier("myfacehaarcascade.xml")

def create_data(Id,Name):
	con=sqlite3.connect("data_persons.db")
	cursor = con.cursor()
	cursor.execute('''CREATE TABLE IF NOT EXISTS Data(
	ID INTEGER PRIMARY KEY,
	NAME TEXT NOT NULL
	);''')
	
	cursor.execute("INSERT INTO data (ID,NAME) VALUES (?,?)",(Id,Name))
	con.commit()
	cursor.close()
	con.close()
	print('data created')
idx=int(input('enter user id '))
name=input("enter user name ")
create_data(idx,name)
os.mkdir('dataSet/'+str(idx))
sampleNum=0
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read(0)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(frame,1.03,4)

	print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces

	for (x, y, w, h) in faces:
		sampleNum=sampleNum+1
		cv2.imwrite("dataSet/"+str(idx)+'/'+name+"."+str(sampleNum)+".jpg",frame[y:y+h,x:x+w])
		cv2.rectangle(frame, (x ,y), (x+w, y+h), (223, 200, 31), 2)
		cv2.waitKey(100)
	
    # Display the resulting frame
    #frame = cv2.resize(frame,(400,400))
	cv2.imshow('Detection Required', frame)
	cv2.waitKey(3)
	if sampleNum>70:
		break
	# When everything done, release the capture
		
cap.release()
cv2.destroyAllWindows()
