# SNN Predictor

Takes in individual calcium image (./Calcium/input.txt) and outputs prediction label.

Performance is around 10-12 ms per image.

Model trained on the following CNN Pytorch model, then converted to an SNN using based on a converted SNN using rate-based encoding:

---

class Simplenet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 3, 1)
        self.relu = nn.ReLU(inplace=False)
        self.fc1 = nn.Linear(150, 23)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

---

To do for optimizations: spike trains can be stored more efficiently, they are 1-bit
Optimize 1-bit input fc layer (maybe use a mask?)

Done:
Since threshold is 1, most uses of threshold that involve multiplication can be eliminated
