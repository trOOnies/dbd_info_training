from dotenv import load_dotenv
load_dotenv("../.env", override=True)

import os
import yaml
import numpy as np
from torch import load, no_grad
from torch import max as torch_max
from torch.cuda import device as get_device
from torch.cuda import is_available as cuda_is_available
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from data import DatasetClass

# DATASET_NAME = 'train_labels.csv'
DATASET_NAME = 'val_labels.csv'
# DATASET_NAME = 'data_null.csv'

print("Predictions for:", os.environ["SELECTED_FD"])

# -------------

assert cuda_is_available()
device = get_device("cuda")

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

MODEL_PATH = os.path.join("models", f"{os.environ['SELECTED_FD']}.pt")
model = load(MODEL_PATH)
modal = model.cuda()
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = DatasetClass(DATASET_NAME, transform=transform)
loader = DataLoader(dataset, batch_size=CFG["batch_size"])

all_preds = np.zeros(len(dataset), dtype=int)
all_labels = np.zeros(len(dataset), dtype=int)
i = 0
with no_grad():
    for images, labels in loader:
        labels_len = labels.size()[0]
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        _, predicted = torch_max(outputs.data, 1)
        all_preds[i:i+labels_len] = predicted.cpu().numpy()
        all_labels[i:i+labels_len] = labels.cpu().numpy()
        i += labels_len

print("Indices with errors:", np.where(all_preds != all_labels)[0])
del all_labels

np.savetxt("all_preds.txt", all_preds, fmt="%d")
