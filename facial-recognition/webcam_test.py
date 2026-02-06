import torch
import cv2
import os
from torchvision import transforms, models
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################
# AUTO LOAD CLASSES
################################
classes = sorted(os.listdir("data/train"))
print("Classes:", classes)

################################
# NORMALIZATION
################################
normalize = transforms.Normalize(
    mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.225]
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
])

################################
# LOAD MODEL
################################
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model.load_state_dict(torch.load("face_model.pth"))

model.eval()
model.to(device)

################################
# FACE DETECTION
################################
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]
        img = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img)
            _, pred = torch.max(out, 1)

        label = classes[pred.item()]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
