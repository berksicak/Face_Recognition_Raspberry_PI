import face_recognition
import cv2
import os
import glob
import numpy as np
import time
import RPi.GPIO as gpio
import drivers

display = drivers.Lcd()
known_face_encodings = []
known_face_names = []
# Resize frame for a faster speed
frame_resizing = 0.25
gpio.setmode(gpio.BOARD)
green_led = 11
red_led = 13
buzzer = 7
gpio.setup(buzzer,gpio.IN)
gpio.setup(buzzer,gpio.OUT)
gpio.setup(green_led,gpio.OUT)
gpio.setup(red_led,gpio.OUT)

def load_encoding_images(images_path):
    # Load Images
    images_path = glob.glob(os.path.join(images_path, "*.*"))

    print("{} encoding images found.".format(len(images_path)))

    # Store image encoding and names
    for img_path in images_path:
        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get the filename only from the initial file path.
        basename = os.path.basename(img_path)
        (filename, ext) = os.path.splitext(basename)
        # Get encoding
        img_encoding = face_recognition.face_encodings(rgb_img, model="large")[0]

        # Store file name and file encoding
        known_face_encodings.append(img_encoding)
        known_face_names.append(filename)
    print("Encoding images loaded")


def detect_known_faces(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=frame_resizing, fy=frame_resizing)
    # Find all the faces and face encodings in the current frame of video
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, model="large")

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
        name = "Unknown"

        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)

    # Convert to numpy array to adjust coordinates with frame resizing quickly
    face_locations = np.array(face_locations)
    face_locations = face_locations / frame_resizing
    return face_locations.astype(int), face_names


if __name__ == '__main__':
    load_encoding_images("images/")
    # Load Camera
    video_frame = cv2.VideoCapture(0)
    control = []
    counter = 32
    while True:
        display.lcd_display_string("Hos Geldiniz!", 1) 
        ret, frame = video_frame.read()
        # Detect Faces
        face_locations, face_names = detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 200, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            control.append(name)
        cv2.imshow("Frame", frame)

        counter = counter - 1

        if counter < 0:
            counter = 32
            unique_size = len(np.unique(control))
            if unique_size == 1 or unique_size == 2:
                list_size = len(control)
                unknown_count = 0
                for i in control:
                    if i == "Unknown":
                        unknown_count = unknown_count + 1
                    else:
                        person_name = i
                face_count = list_size - unknown_count
                ratio = float((face_count/list_size)*100)
                if ratio >= 75:
                    print("Merhaba " + person_name + "!")
                    display.lcd_display_string("Merhaba " + person_name + "!", 1) 
                    gpio.output(green_led,True)
                    time.sleep(6)
                    gpio.output(green_led,False)
                    display.lcd_clear()
                else:
                    print("Yuz Dogrulanmadi.")
                    display.lcd_display_string("Yuz Dogrulanmadi.",1)
                    gpio.output(red_led,True)
                    gpio.output(buzzer,True)
                    time.sleep(3)
                    gpio.output(buzzer,False)
                    time.sleep(3)
                    gpio.output(red_led,False)
                    display.lcd_clear()
            elif unique_size == 0:
                print("Yuz Algilanmadi.")
                display.lcd_display_string("Yuz Algilanmadi.",1)
                gpio.output(red_led,True)
                time.sleep(6)
                gpio.output(red_led,False)
                display.lcd_clear()
            else:
                print("Birden Fazla Yuz Tespit Edildi.")
                display.lcd_display_string("Birden Fazla Yuz",1)
                display.lcd_display_string("Tespit Edildi.",2)
                gpio.output(red_led,True)
                gpio.output(buzzer,True)
                time.sleep(3)
                gpio.output(buzzer,False)
                time.sleep(3)
                gpio.output(red_led,False)
                display.lcd_clear()
            control.clear()
        key = cv2.waitKey(10) & 0xff
        if key == 27:
            break

    video_frame.release()
    cv2.destroyAllWindows()
