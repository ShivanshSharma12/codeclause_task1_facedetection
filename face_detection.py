#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import os


# In[2]:


cap = cv2.VideoCapture(0) # 0 here means that we are selecting the default video source
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_path)


# In[3]:


while True: # inside while loop because we want to capture this continously
    ret, frame = cap.read() # read first frame from video capture object
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.cvtColor(frame,0)
   
    detections = face_cascade.detectMultiScale(gray , 1.3, 5)

    if(len(detections)> 0):
        (x,y,w,h) = detections[0]
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        text = "Face Detected"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
        
    cv2.imshow('frame',frame) # now display this frame using imshow method
    # the problem here is that, the frames are being captured so fast ki vo hume kuch show hi ni hora
    # because har ek frame ko next frame overlap kra h 
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break   #so hum delay daal dete h while loop mei , here means wait for 1 miliseconds before goinf to the next frame
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




