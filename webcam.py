import numpy as np
import cv2
import time
import boto3
import os

# --------- Connect to AWS --------
client = boto3.client('rekognition',
        region_name = 'us-east-1', 
        aws_access_key_id = '*******',
        aws_secret_access_key = '******')

source = []
names = []
folder_name = 'datasets'

# read input images
for name in os.listdir(folder_name):
    image = cv2.imread(folder_name + '/' + name)
    name = os.path.splitext(name)[0] 
    source.append(image)
    names.append(name)


# ---------- IP Cam URL ------
cap = cv2.VideoCapture(0)

while(True):
    start_time = time.time()
    
    ret, frame = cap.read()
    
    # Preprocess input
    height, width, channels = frame.shape
    target_img = cv2.imencode('.jpg', frame)[1].tostring()

    face_pic = client.detect_faces(Image={'Bytes':target_img})
    if len(face_pic['FaceDetails']) > 0:
        i = 0
        for face in source:
            source_img = cv2.imencode('.jpg', face)[1].tostring()
            response = client.compare_faces(SourceImage={ 'Bytes': source_img },TargetImage={ 'Bytes': target_img })
            if len(response['FaceMatches']) > 0:
                if response['FaceMatches'][0]['Similarity'] > 80.0:
            
                    # process output
                    x = int(response['FaceMatches'][0]['Face']['BoundingBox']['Left']*width)
                    y = int(response['FaceMatches'][0]['Face']['BoundingBox']['Top']*height)
                    w = int(response['FaceMatches'][0]['Face']['BoundingBox']['Width']*width)
                    h = int(response['FaceMatches'][0]['Face']['BoundingBox']['Height']*height)

                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
                    cv2.putText(frame, names[i], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    i = i+1
            
    # Dislpay FPS
    cv2.putText(frame, 'FPS: '+ str(1/(time.time()-start_time)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    # Dislpay Quit instruction
    cv2.putText(frame, 'Press Q to quit', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Display video stream
    cv2.imshow('frame',frame)
    
    # Reduce stress to processor
    time.sleep(0.1) 

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
