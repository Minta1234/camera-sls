import cv2
import mediapipe as mp
from cvzone.FaceDetectionModule import FaceDetector

# ตั้งค่ากล้อง
ws, hs = 1280, 720
cap = cv2.VideoCapture(0)
cap.set(3, ws)
cap.set(4, hs)

if not cap.isOpened():
    print("❌ ไม่สามารถเข้าถึงกล้องได้")
    exit()

# โหลด Face Detector จาก cvzone
detector = FaceDetector()

# โหลด MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        success, img = cap.read()
        if not success:
            print("❌ อ่านภาพจากกล้องไม่สำเร็จ")
            continue

        # พลิกภาพให้เหมือนกระจก
        img = cv2.flip(img, 1)

        # หาใบหน้าด้วย cvzone
        img, bboxs = detector.findFaces(img, draw=False)

        # วาดกรอบใบหน้าและจุดตำแหน่งใบหน้าคนแรก (ถ้ามี)
        if bboxs:
            fx, fy = bboxs[0]["center"]
            cv2.circle(img, (fx, fy), 80, (0, 0, 255), 2)
            cv2.putText(img, f"Target: ({fx}, {fy})", (fx + 15, fy - 15),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(img, "TARGET LOCKED", (850, 50),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            # เส้นไขว้ตำแหน่ง
            cv2.line(img, (0, fy), (ws, fy), (0, 0, 0), 2)
            cv2.line(img, (fx, hs), (fx, 0), (0, 0, 0), 2)
            cv2.circle(img, (fx, fy), 15, (0, 0, 255), cv2.FILLED)
        else:
            # ไม่มีใบหน้า
            cv2.putText(img, "NO TARGET", (880, 50),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            cx, cy = ws // 2, hs // 2
            cv2.circle(img, (cx, cy), 80, (0, 0, 255), 2)
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
            cv2.line(img, (0, cy), (ws, cy), (0, 0, 0), 2)
            cv2.line(img, (cx, hs), (cx, 0), (0, 0, 0), 2)

        # แปลงภาพ BGR -> RGB เพื่อใช้กับ MediaPipe Pose
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        pose_results = pose.process(img_rgb)
        img_rgb.flags.writeable = True

        # วาด skeleton stick figure ถ้าจับ pose ได้
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

        # แสดงผล
        cv2.imshow("Face Detection + Pose Stick Figure", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
