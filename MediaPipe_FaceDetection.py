import cv2
import mediapipe as mp
import imutils

mp_face_detect = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

#se hace uso de la camara web para la detecccion de rostros
captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_detect.FaceDetection(
    min_detection_confidence = 0.5
    ) as face_detection:

    while True:
        ret, frame = captura.read()
        if ret == False:
            break
        frame = imutils.resize(frame, width=720)
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resultado = face_detection.process(frameRGB)

        if resultado.detections is not None:
            for detection in resultado.detections:
                mp_drawing.draw_detection(frame, detection, 
                mp_drawing.DrawingSpec(color= (0,255,255),circle_radius=2),
                mp_drawing.DrawingSpec(color= (255,0,255))
                )
        cv2.imshow ('Mediapipe Face', frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
captura.release()
cv2.destroyAllWindows()
