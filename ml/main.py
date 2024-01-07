from dotenv import load_dotenv
load_dotenv()

import os
import yaml
from torch import no_grad, save
from torch import max as torch_max
from torch import device as get_device
from torch.cuda import mem_get_info
from torch.cuda import is_available as cuda_is_available
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
from torch.nn import (
    Sequential,
    Conv2d,
    ReLU,
    MaxPool2d,
    Flatten,
    Linear,
    CrossEntropyLoss
)
from data import DatasetClass, get_total_classes

# -------------

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print("Model for:", os.environ['SELECTED_FD'])

assert cuda_is_available()
device = get_device("cuda")

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

TOTAL_CLASSES = get_total_classes()

# Input image size is 55x57 and the number of channels is 3 (RGB)
model = Sequential(
    Conv2d(3, 32, (3, 3), padding=1),  # 32 filters
    ReLU(),
    MaxPool2d((2, 2)),  # Output size: 32x27x28
    Conv2d(32, 64, (3, 3), padding=1),  # 64 filters
    ReLU(),
    MaxPool2d((2,2)),  # Output size: 64x13x14
    Flatten(),  # Output size: 64*13*14
    Linear(64*13*14, 128),  
    ReLU(),
    Linear(128, TOTAL_CLASSES),
)

optimizer = Adam(model.parameters(), lr=CFG["adam_lr"])
criterion = CrossEntropyLoss()

model = model.cuda()
summary(model, (3, 55, 57), batch_size=CFG["batch_size"], device="cuda")

print("MEMORY")
print("- Free: {:,.2} GiB\n- Total: {:,.2} GiB".format(
    *[v / (2**30) for v in mem_get_info(device.index)])
)

# Define any image transformations here
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Loading data...")
train_dataset = DatasetClass('train_labels.csv', transform=transform)
val_dataset = DatasetClass('val_labels.csv', transform=transform)

print("Train datapoints:", len(train_dataset))
print("Val datapoints:", len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG["batch_size"])
print("Data loaded.")

print("Training initialized.")
for epoch in range(1, CFG["epochs"] + 1):
    model.train()
    for images, labels in train_loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            _, predicted = torch_max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            f"Epoch [{epoch}/{CFG['epochs']}],",
            f"Loss: {loss.item():.4f},",
            f"Validation Accuracy: {100 * correct / total:.2f}%"
        )

print("Training complete.")
save(model, os.path.join("models", f"{os.environ['SELECTED_FD']}.pt"))
del model
print("Model saved.")
