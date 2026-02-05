import cv2
import os

name = "others"

save_path = os.path.join(os.getcwd(), "data", "train", name)
os.makedirs(save_path, exist_ok=True)

cam = cv2.VideoCapture(0)

count = 0

print("SPACE = capture | ESC = exit")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Camera error")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        file_path = os.path.join(save_path, f"{count}.jpg")
        success = cv2.imwrite(file_path, frame)

        if success:
            print("Saved:", file_path)
        else:
            print("Failed to save!")

        count += 1

    elif key == 27:
        break

cam.release()
cv2.destroyAllWindows()
