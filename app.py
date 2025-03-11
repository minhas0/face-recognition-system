import cv2
import os
from flask import Flask, request, render_template
from datetime import datetime, date
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = Flask(__name__)

imgBackground = cv2.imread("background.png")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

def extract_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
    return face_points

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

attendance = {}

# ✅ Send Email Function
def send_email():
    sender_email = "Njan@mailslurp.xyz"
    receiver_email = "zoluttan@gmail.com"
    password = "7c08d88bca889ba1b174ce7974e3a6a05a8f6d73b945af6984fbe45da99497b0"

    # Read the Excel file
    excel_path = 'Attendance/Attendance.xlsx'
    df = pd.read_excel(excel_path)

    # Convert to HTML Table
    html_content = df.to_html()

    # Email Setup
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = f"Daily Attendance Report - {date.today()}"
    body = MIMEText(html_content, 'html')
    msg.attach(body)

    # Sending Email via MailSlurp SMTP
    with smtplib.SMTP('smtp.mailslurp.com', 587) as server:
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())

# ✅ Homepage to avoid 404 Error
@app.route('/')
def home():
    return "✅ Real-time Attendance System is Running! Go to /start to begin."

@app.route('/start', methods=['GET'])
def start():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            person = identify_face(face.reshape(1, -1))[0]
            current_time = datetime.now().strftime('%H:%M:%S')

            if person not in attendance:
                attendance[person] = [current_time, '']
            else:
                attendance[person][1] = current_time

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow('Attendance Monitoring', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

    # ✅ Calculate time spent
    records = []
    for person, times in attendance.items():
        entry_time = datetime.strptime(times[0], '%H:%M:%S')
        exit_time = datetime.strptime(times[1], '%H:%M:%S') if times[1] else datetime.now()
        time_spent = exit_time - entry_time
        records.append([person, times[0], times[1], str(time_spent)])

    # ✅ Save data to Excel
    df = pd.DataFrame(records, columns=['Name', 'Entry Time', 'Exit Time', 'Time Spent'])
    excel_path = 'Attendance/Attendance.xlsx'

    if os.path.exists(excel_path):
        old_df = pd.read_excel(excel_path)
        new_df = pd.concat([old_df, df], ignore_index=True)
        new_df.to_excel(excel_path, index=False)
    else:
        df.to_excel(excel_path, index=False)

    # ✅ Send Email
    send_email()

    return "Attendance Recorded and Emailed Successfully ✅"

if __name__ == '__main__':
    app.run(debug=True)
