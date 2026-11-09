"""The ECG + tabular-metadata model (unchanged from the original notebook)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECG1DCNN(nn.Module):
    """3-layer 1D-CNN over the 12-lead signal, fused with a small MLP over metadata.

    Inputs
    ------
    ecg     : (batch, 12, T)
    tabular : (batch, num_features)

    Output
    ------
    (batch, num_labels) independent sigmoid probabilities (multi-label).
    """

    def __init__(self, num_features, num_labels):
        super().__init__()

        # CNN for ECG signal (12 leads)
        self.conv1 = nn.Conv1d(12, 32, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.AdaptiveAvgPool1d(1)  # -> (batch, 128, 1)

        # MLP for tabular features
        self.tabular_net = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_labels),
            nn.Sigmoid(),  # multi-label
        )

    def forward(self, ecg, tabular):
        x = self.pool1(F.relu(self.bn1(self.conv1(ecg))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)

        t = self.tabular_net(tabular)

        return self.classifier(torch.cat((x, t), dim=1))
