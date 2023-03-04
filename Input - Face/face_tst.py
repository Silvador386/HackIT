import cv2
import mediapipe as mp
import time

from utils import write_to_json


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    NUM_FACE = 1

    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
    drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    while True:
        success, img = cap.read()
        if not success:
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceMesh.process(imgRGB)
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)

                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    print(id, x, y)


        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Test", img)

        if cv2.waitKey(1)& 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
