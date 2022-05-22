from mtcnn import MTCNN
import cv2
from PIL import Image
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
from tkinter import Tcl
import glob




#Read video and save frames as images
vidcap = cv2.VideoCapture('VIDEO.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite(+str(count)+".jpg", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))     # save frame as JPEG file      
  success,image = vidcap.read()
  count += 1




#calling mtcnn face detector
face_detector = MTCNN()
def face_analysis(img):
    
    rect_loc = face_detector.face_analysiss(img)
    predictions = []

    # If you are sure there is 1 face in the video uncomment the following 2 lines
    # if len(rect_loc) > 1:
    #   rect_loc = [rect_loc[0]]
    
    
    #getting predictions for each of the detected faces
    for face in rect_loc:
        x, y, w, h = face['box']
        center = [x+(w/2), y+(h/2)]
        border = max(w, h)
        
        left = max(int(center[0]-(border/2)), 0)
        right = max(int(center[0]+(border/2)), 0)
        top = max(int(center[1]-(border/2)), 0)
        bottom = max(int(center[1]+(border/2)), 0)
        
        # croping the face
        cropped_img = img[top:top+border, left:left+border, :]
        cropped_img = np.array(Image.fromarray(cropped_img))
        
        #getting predictions from DeepFace
        #you can use your own model here
        obj = DeepFace.analyze(img_path = cropped_img, actions = ['age','gender', 'race', 'emotion'], enforce_detection = False)
        age = obj['age']
        emotion = obj['dominant_emotion']
        race = obj['dominant_race']
        gender = obj['gender']
        
        predictions.append([top, right, bottom, left, age, emotion, race, gender])
        
    return predictions




# getting filenames(frame names) from current folder with jpg format
filenames = glob.glob('*.jpg')
#sorting filenames
filenames = Tcl().call('lsort', '-dict', filenames)
img_array = []
i = 0
for filename in filenames:

  Frame = Image.open(filename)
  Frame = np.array(Frame)
  scale_percent = scale_percent = 100* 640/max(np.shape(Frame)) # resize frame
  w = int(Frame.shape[1] * scale_percent / 100)
  h = int(Frame.shape[0] * scale_percent / 100)
  dim = (w, h)
  Frame = cv2.resize(Frame, dim, interpolation = cv2.INTER_AREA)
  face_locations = face_analysis(Frame)

  for top, right, bottom, left, age, emotion, race, gender in face_locations:
      # drawing boxes around the faces
      cv2.rectangle(Frame, (left, top), (right, bottom), (0, 0, 255), 2)
      #writing predictions on top of the box
      cv2.putText(Frame, 'Race: '+ race, (left, top-55), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,0,0), 2)
      cv2.putText(Frame, 'Gender: '+ gender, (left, top-10), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,0,0), 2)
      cv2.putText(Frame, 'Age: '+ str(age), (left, top-25), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,0,0), 2)
      cv2.putText(Frame, 'Emotion: '+ emotion, (left, top-40), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,0,0), 2)

  #saving images with predictions
  cv2.imwrite("done/"+str(i)+".jpg", np.array(Frame))
  i += 1
  print (i)




#putting frames with predictions together as a video
img_array = []
filenames = glob.glob('done/*.jpg')
filenames = Tcl().call('lsort', '-dict', filenames)


for filename in filenames:
    img = cv2.imread(filename)
    img_array.append(img)


height, width, layers = img.shape
size = (width,height)
out = cv2.VideoWriter('VIDEO_out.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
 
for i in range(len(img_array)):
    
    out.write(img_array[i])
    
out.release()

