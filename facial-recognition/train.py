import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################
# NORMALIZATION (VERY IMPORTANT)
################################
normalize = transforms.Normalize(
    mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.225]
)

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3),
    transforms.ToTensor(),
    normalize
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
])

################################
# DATA
################################

train_data = datasets.ImageFolder("data/train", transform=train_transform)
test_data  = datasets.ImageFolder("data/test",  transform=test_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data, batch_size=16)

classes = train_data.classes
print("Classes:", classes)

################################
# MODEL
################################

model = models.mobilenet_v2(pretrained=True)

# freeze backbone
for p in model.features.parameters():
    p.requires_grad = False

model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

################################
# TRAINING + ACCURACY
################################

EPOCHS = 10
best_acc = 0

for epoch in range(EPOCHS):

    model.train()
    loss_sum = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    ################################
    # VALIDATION
    ################################
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            _, pred = torch.max(out, 1)

            total += labels.size(0)
            correct += (pred == labels).sum().item()

    acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss_sum:.3f} | Accuracy: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "face_model.pth")

print("Best Accuracy:", best_acc)
