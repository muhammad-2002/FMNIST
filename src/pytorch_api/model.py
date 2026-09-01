import torch
import torch.nn as nn


class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.netwrok = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.netwrok(x)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = MyModel()

model.load_state_dict(
    torch.load(
        "models/FMNIST.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully!")
print("Device:", DEVICE)