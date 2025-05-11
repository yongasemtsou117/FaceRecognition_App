#https://www.youtube.com/watch?v=RGOj5yH7evk&t=2443s
import face_recognition as fr
import os
import pandas as pd
import cv2
import face_recognition
import numpy as np
from time import sleep
from datetime import datetime
import logging
from typing import List, Tuple, Optional

#path='C:/Users/blaise/OneDrive/Desktop/Data Science cours/Simplilearn/Datasets/attendance'
path='C:/Users/blaise/Face_Recognition/attendance'
images=[]
classNames=[]
mylist=os.listdir(path)
print(mylist)

for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode= face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open ('C:/Users/blaise/Face_Recognition/Attendance9.csv','r+') as f:
        myDataList= f.readlines()
        nameList=[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
        print(myDataList)


encodeListknown=findEncodings(images)
print('Encoding complete')

cap=cv2.VideoCapture(0)

while True:
    success, img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgS)
    encodesCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)
    

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListknown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListknown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
        
        if matches [matchIndex]:
             name=classNames[matchIndex].upper()
             #print(name)
             y1,x2,y2,x1=faceLoc
             y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
             cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
             cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED)
             cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
             markAttendance(name)
    
    cv2.imshow('webcam',img)
    key=cv2.waitKey(1)
    
    if key==ord('q'):
        break

class FaceRecognitionSystem:
    def __init__(self):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('face_recognition.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.attendance_dir = 'attendance'
        self.attendance_file = 'Attendance.csv'
        self.min_face_confidence = 0.6  # Minimum confidence for face recognition
        self.frame_scale = 0.25  # Scale factor for processing (smaller = faster)
        self.display_scale = 1.0  # Scale factor for display
        
        # Initialize storage
        self.known_face_encodings = []
        self.known_face_names = []
        self.recognized_faces = set()  # Track faces recognized in current session
        
        # Ensure directories exist
        self._setup_directories()
        
        # Load known faces
        self.load_known_faces()

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.attendance_dir, exist_ok=True)
        
        # Initialize attendance file if it doesn't exist
        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=['Name', 'Date', 'Time', 'Confidence'])
            df.to_csv(self.attendance_file, index=False)
            self.logger.info(f"Created new attendance file: {self.attendance_file}")

    def load_known_faces(self) -> None:
        """Load and encode all known faces from the attendance directory."""
        self.logger.info("Loading known faces...")
        
        image_files = [f for f in os.listdir(self.attendance_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not image_files:
            self.logger.warning("No image files found in attendance directory!")
            return
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        for image_file in image_files:
            try:
                # Load image
                image_path = os.path.join(self.attendance_dir, image_file)
                image = face_recognition.load_image_file(image_path)
                
                # Convert to RGB if necessary
                if len(image.shape) == 2:  # Grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                
                # Get face encoding
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    self.known_face_encodings.append(face_encodings[0])
                    name = os.path.splitext(image_file)[0]
                    self.known_face_names.append(name)
                    self.logger.info(f"Successfully encoded: {name}")
                else:
                    self.logger.warning(f"No face found in {image_file}")
                    
            except Exception as e:
                self.logger.error(f"Error processing {image_file}: {str(e)}")
        
        self.logger.info(f"Successfully loaded {len(self.known_face_names)} faces")

    def mark_attendance(self, name: str, confidence: float) -> None:
        """Mark attendance for a recognized person with confidence score."""
        try:
            now = datetime.now()
            date = now.strftime('%Y-%m-%d')
            time = now.strftime('%H:%M:%S')
            
            # Read existing attendance
            df = pd.read_csv(self.attendance_file)
            
            # Check if person already marked attendance today
            today_attendance = df[(df['Name'] == name) & (df['Date'] == date)]
            
            if today_attendance.empty and name not in self.recognized_faces:
                # Add new attendance record
                new_row = pd.DataFrame({
                    'Name': [name],
                    'Date': [date],
                    'Time': [time],
                    'Confidence': [f"{confidence:.2%}"]
                })
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(self.attendance_file, index=False)
                self.recognized_faces.add(name)
                self.logger.info(f"Marked attendance for {name} with confidence {confidence:.2%}")
                
        except Exception as e:
            self.logger.error(f"Error marking attendance: {str(e)}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Process a single frame and return the annotated frame and recognized names."""
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_scale, fy=self.frame_scale)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        recognized_names = []
        
        # Process each face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Match against known faces
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                min_distance = face_distances[best_match_index]
                confidence = 1 - min_distance
                
                if confidence >= self.min_face_confidence:
                    name = self.known_face_names[best_match_index]
                    recognized_names.append(name)
                    self.mark_attendance(name, confidence)
                else:
                    name = "Unknown"
            else:
                name = "Unknown"
                confidence = 0.0
            
            # Scale back face locations
            scale = int(1/self.frame_scale)
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale
            
            # Draw box and label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Add confidence score to display
            conf_text = f"{confidence:.2%}" if name != "Unknown" else ""
            label = f"{name} {conf_text}"
            
            # Draw filled background for text
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame, recognized_names

    def run(self) -> None:
        """Run the face recognition system."""
        self.logger.info("Starting face recognition system...")
        
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.logger.error("Could not open video capture device!")
            return
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to grab frame")
                    continue
                
                # Process frame
                frame, recognized_names = self.process_frame(frame)
                
                # Add UI elements
                self._add_ui_elements(frame, recognized_names)
                
                # Display frame
                cv2.imshow('Face Recognition System', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):  # Reload known faces
                    self.known_face_encodings = []
                    self.known_face_names = []
                    self.load_known_faces()
                
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
            
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Face recognition system stopped")

    def _add_ui_elements(self, frame: np.ndarray, recognized_names: List[str]) -> None:
        """Add UI elements to the frame."""
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, timestamp, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add help text
        cv2.putText(frame, "Press 'q' to quit, 'r' to reload faces", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add recognized names count
        count_text = f"Recognized today: {len(self.recognized_faces)}"
        cv2.putText(frame, count_text, (frame.shape[1] - 300, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

if __name__ == "__main__":
    try:
        system = FaceRecognitionSystem()
        system.run()
    except Exception as e:
        logging.error(f"Application error: {str(e)}")

            




