# To launch python app.py
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import cv2
import face_recognition
import numpy as np
from datetime import datetime
import os
import logging
from werkzeug.utils import secure_filename
import webbrowser
import threading
import time
import atexit
from concurrent.futures import ThreadPoolExecutor
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'attendance'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
face_recognition_active = False
known_face_encodings = []
known_face_names = []
recognized_faces = set()
frame_skip = 2  # Process every nth frame
frame_count = 0
executor = ThreadPoolExecutor(max_workers=4)  # Pour le traitement parallèle
camera_index = 0  # Par défaut, utiliser la caméra principale (index 0)

class CameraManager:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.lock = threading.Lock()
        self.last_frame = None
        self.error_count = 0
        self.max_errors = 3
        
    def start(self):
        """Démarre la caméra"""
        with self.lock:
            if not self.is_running:
                try:
                    if self.camera is not None:
                        self.camera.release()
                        time.sleep(0.5)
                    
                    self.camera = cv2.VideoCapture(0)
                    if not self.camera.isOpened():
                        self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    
                    if self.camera.isOpened():
                        # Configuration de base
                        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.camera.set(cv2.CAP_PROP_FPS, 30)
                        
                        # Lire quelques frames pour initialiser la caméra
                        for _ in range(5):
                            ret, frame = self.camera.read()
                            if ret and frame is not None:
                                self.last_frame = frame
                                self.is_running = True
                                self.error_count = 0
                                logger.info("Caméra démarrée avec succès")
                                return True
                            time.sleep(0.1)
                    
                    logger.error("Impossible d'initialiser la caméra")
                    return False
                    
                except Exception as e:
                    logger.error(f"Erreur lors du démarrage de la caméra: {str(e)}")
                    self.release()
                    return False
            return self.is_running
    
    def read_frame(self):
        """Lit une frame de la caméra"""
        try:
            if not self.is_running:
                if not self.start():
                    return False, None
            
            ret, frame = self.camera.read()
            if ret and frame is not None:
                self.last_frame = frame
                self.error_count = 0
                return True, frame
            else:
                self.error_count += 1
                if self.error_count >= self.max_errors:
                    logger.warning("Trop d'erreurs de lecture, tentative de réinitialisation")
                    self.restart()
                return False, self.last_frame
                
        except Exception as e:
            logger.error(f"Erreur de lecture: {str(e)}")
            self.error_count += 1
            if self.error_count >= self.max_errors:
                self.restart()
            return False, self.last_frame
    
    def restart(self):
        """Redémarre la caméra en cas de problème"""
        logger.info("Redémarrage de la caméra...")
        self.release()
        time.sleep(1)  # Attendre avant de redémarrer
        return self.start()
    
    def release(self):
        """Libère les ressources de la caméra"""
        with self.lock:
            if self.camera is not None:
                try:
                    self.camera.release()
                except:
                    pass
                finally:
                    self.camera = None
                    self.is_running = False
                    self.error_count = 0

# Instance globale du gestionnaire de caméra
camera_manager = CameraManager()

def open_browser():
    """Open browser after a short delay."""
    time.sleep(1.5)  # Wait for Flask to start
    webbrowser.open('http://127.0.0.1:5000')

def load_known_faces():
    """Load and encode all known faces from the attendance directory."""
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    
    attendance_dir = 'attendance'
    if not os.path.exists(attendance_dir):
        os.makedirs(attendance_dir)
        
    image_files = [f for f in os.listdir(attendance_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_file in image_files:
        try:
            image_path = os.path.join(attendance_dir, image_file)
            image = face_recognition.load_image_file(image_path)
            
            # Convert to RGB if necessary
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                name = os.path.splitext(image_file)[0]
                known_face_names.append(name)
                logger.info(f"Successfully encoded: {name}")
            else:
                logger.warning(f"No face found in {image_file}")
                
        except Exception as e:
            logger.error(f"Error processing {image_file}: {str(e)}")

def save_recognized_face(frame, name, face_location):
    """Save the recognized face with timestamp."""
    try:
        # Create directory if it doesn't exist
        save_dir = 'recognized_faces'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Create person directory
        person_dir = os.path.join(save_dir, name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        
        # Extract face region with some margin
        top, right, bottom, left = face_location
        margin = 30
        face_img = frame[max(0, top-margin):min(frame.shape[0], bottom+margin),
                        max(0, left-margin):min(frame.shape[1], right+margin)]
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{name}_{timestamp}.jpg'
        filepath = os.path.join(person_dir, filename)
        
        # Save the image
        cv2.imwrite(filepath, face_img)
        logger.info(f"Saved recognized face: {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving recognized face: {str(e)}")

def mark_attendance(name):
    """Mark attendance for a recognized person."""
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d')
    time_string = now.strftime('%H:%M:%S')
    
    attendance_file = 'Attendance.csv'
    if not os.path.exists(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write('Name,Date,Time\n')
    
    with open(attendance_file, 'r+') as f:
        lines = f.readlines()
        names_today = [line.split(',')[0] for line in lines 
                      if date_string in line]
        
        if name not in names_today and name not in recognized_faces:
            f.write(f'{name},{date_string},{time_string}\n')
            recognized_faces.add(name)
            logger.info(f"Marked attendance for {name}")

def process_face(face_encoding, face_location):
    """Process a single face in parallel"""
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"
    
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
        mark_attendance(name)
    
    return name, face_location

def generate_frames():
    """Generate frames from the camera with face recognition."""
    global face_recognition_active, frame_count
    
    if not camera_manager.start():
        logger.error("Impossible de démarrer la caméra")
        return
    
    frame_count = 0
    
    while True:
        try:
            ret, frame = camera_manager.read_frame()
            if not ret or frame is None:
                time.sleep(0.1)  # Petite pause avant de réessayer
                continue
            
            frame_count += 1
            
            # Ajouter un indicateur de statut
            cv2.putText(frame, "CAMERA ACTIVE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if face_recognition_active and frame_count % frame_skip == 0:
                # Process face recognition
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                        mark_attendance(name)
                    
                    # Scale back face locations
                    top, right, bottom, left = face_location
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Draw the box and label
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, name, (left + 6, bottom - 6),
                              cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Convert frame to jpg
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_data = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                   
        except Exception as e:
            logger.error(f"Erreur lors du traitement de l'image: {str(e)}")
            time.sleep(0.1)  # Pause avant de continuer
            continue

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_recognition', methods=['POST'])
def toggle_recognition():
    """Toggle face recognition on/off."""
    global face_recognition_active
    face_recognition_active = not face_recognition_active
    return jsonify({'active': face_recognition_active})

@app.route('/upload_face', methods=['POST'])
def upload_face():
    """Handle face image upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Reload known faces
        load_known_faces()
        
        return jsonify({'message': 'File uploaded successfully'})

@app.route('/get_attendance')
def get_attendance():
    """Get current attendance list."""
    attendance_file = 'Attendance.csv'
    
    try:
        # Créer le fichier s'il n'existe pas
        if not os.path.exists(attendance_file):
            with open(attendance_file, 'w', encoding='utf-8') as f:
                f.write('Name,Date,Time\n')
            logger.info(f"Fichier {attendance_file} créé avec succès")
            return jsonify([])
        
        # Lire le fichier avec gestion explicite de l'encodage
        try:
            with open(attendance_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if len(lines) <= 1:  # Fichier vide ou juste l'en-tête
                return jsonify([])
                
            attendance = []
            for line in lines[1:]:  # Skip header
                try:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        attendance.append({
                            'name': parts[0],
                            'date': parts[1],
                            'time': parts[2]
                        })
                except Exception as e:
                    logger.warning(f"Ligne ignorée dans {attendance_file}: {line.strip()} - Erreur: {str(e)}")
                    continue
                    
            return jsonify(attendance)
            
        except UnicodeDecodeError:
            # Essayer avec un autre encodage si utf-8 échoue
            with open(attendance_file, 'r', encoding='latin-1') as f:
                lines = f.readlines()
            # Même traitement que ci-dessus
            if len(lines) <= 1:
                return jsonify([])
            
            attendance = []
            for line in lines[1:]:
                try:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        attendance.append({
                            'name': parts[0],
                            'date': parts[1],
                            'time': parts[2]
                        })
                except Exception as e:
                    logger.warning(f"Ligne ignorée dans {attendance_file}: {line.strip()} - Erreur: {str(e)}")
                    continue
                    
            return jsonify(attendance)
            
    except Exception as e:
        logger.error(f"Erreur lors de la lecture de {attendance_file}: {str(e)}")
        # Créer un nouveau fichier en cas de corruption
        try:
            with open(attendance_file, 'w', encoding='utf-8') as f:
                f.write('Name,Date,Time\n')
            logger.info(f"Fichier {attendance_file} réinitialisé suite à une erreur")
        except Exception as write_error:
            logger.error(f"Impossible de réinitialiser {attendance_file}: {str(write_error)}")
        return jsonify([])

@app.route('/export_attendance', methods=['GET'])
def export_attendance():
    """Export attendance data to Excel."""
    try:
        import pandas as pd
        from io import BytesIO
        
        # Read attendance data
        df = pd.read_csv('Attendance.csv')
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Attendance', index=False)
            
            # Get workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Attendance']
            
            # Add formatting
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4a90e2',
                'font_color': 'white'
            })
            
            # Format headers
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
                
            # Adjust column widths
            for i, col in enumerate(df.columns):
                max_length = max(df[col].astype(str).apply(len).max(), len(col)) + 2
                worksheet.set_column(i, i, max_length)
        
        # Prepare response
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={
                "Content-Disposition": "attachment;filename=attendance_report.xlsx"
            }
        )
    except Exception as e:
        logger.error(f"Error exporting attendance: {str(e)}")
        return jsonify({'error': 'Failed to export attendance data'}), 500

@app.route('/clear_attendance', methods=['POST'])
def clear_attendance():
    """Clear all attendance records and recognized faces."""
    try:
        # Clear attendance file
        with open('Attendance.csv', 'w') as f:
            f.write('Name,Date,Time\n')
        
        # Clear recognized faces directory
        if os.path.exists('recognized_faces'):
            shutil.rmtree('recognized_faces')
        
        # Reset recognized faces set
        recognized_faces.clear()
        
        return jsonify({'message': 'Présences et photos effacées avec succès'})
    except Exception as e:
        logger.error(f"Error clearing attendance: {str(e)}")
        return jsonify({'error': 'Erreur lors de l\'effacement des données'}), 500

@app.route('/get_stats')
def get_stats():
    """Get attendance statistics."""
    try:
        import pandas as pd
        from collections import Counter
        
        df = pd.read_csv('Attendance.csv')
        
        # Calculate statistics
        total_records = len(df)
        unique_people = len(df['Name'].unique())
        dates = df['Date'].unique()
        attendance_by_date = Counter(df['Date'])
        
        # Get most recent records
        recent_records = df.tail(5).to_dict('records')
        
        return jsonify({
            'total_records': total_records,
            'unique_people': unique_people,
            'dates': len(dates),
            'attendance_by_date': dict(attendance_by_date),
            'recent_records': recent_records
        })
    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({'error': 'Failed to get statistics'}), 500

@app.route('/recognized_faces/<name>/<filename>')
def serve_face_image(name, filename):
    """Serve saved face images."""
    return send_from_directory(os.path.join('recognized_faces', name), filename)

@app.teardown_appcontext
def teardown_camera(exception=None):
    """Ensure camera is released when Flask context ends."""
    # Ne pas libérer la caméra à chaque fois, seulement quand c'est vraiment nécessaire
    if exception is not None:
        camera_manager.release()
        logger.info("Caméra libérée dans Flask teardown (exception)")

if __name__ == '__main__':
    try:
        # Charger les visages connus
        load_known_faces()
        
        # Démarrer le navigateur
        threading.Thread(target=open_browser).start()
        
        # Lancer l'application Flask
        app.run(debug=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Erreur lors du démarrage de l'application: {str(e)}")
    finally:
        # Nettoyer les ressources
        logger.info("Nettoyage des ressources...")
        camera_manager.release() 