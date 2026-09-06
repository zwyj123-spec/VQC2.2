# vqc_core.py
import os
import random
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def load_or_generate_data(file_normal=None, file_fault=None, points_per_sample=30000, num_samples=20):
    normal_data, fault_data = [], []
    if file_normal is not None and file_fault is not None:
        try:
            df_norm = pd.read_excel(file_normal)
            df_fault = pd.read_excel(file_fault)
            vec_norm = df_norm.select_dtypes(include=[np.number]).values.flatten()
            vec_fault = df_fault.select_dtypes(include=[np.number]).values.flatten()
            for i in range(len(vec_norm) // points_per_sample):
                normal_data.append(vec_norm[i * points_per_sample: (i + 1) * points_per_sample])
            for i in range(len(vec_fault) // points_per_sample):
                fault_data.append(vec_fault[i * points_per_sample: (i + 1) * points_per_sample])
        except Exception:
            normal_data, fault_data = [], []

    if len(normal_data) == 0 or len(fault_data) == 0:
        t = np.linspace(0, 0.6, points_per_sample, endpoint=False)
        np.random.seed(42)
        for _ in range(num_samples):
            white_noise = np.random.normal(0, 0.08, points_per_sample)
            grid_hum = 0.08 * np.sin(2 * np.pi * 50 * t)
            base_signal = 1.2 * np.sin(2 * np.pi * 100 * t) * np.exp(-15 * t)
            harmonics = 0.5 * np.sin(2 * np.pi * 2500 * t) * np.exp(-30 * t)
            normal_data.append(base_signal + harmonics + white_noise + grid_hum)

            delayed_base = 1.0 * np.sin(2 * np.pi * 100 * (t - 0.05)) * np.exp(-10 * (t - 0.05)) * (t >= 0.05)
            linkage_friction = 0.85 * np.sin(2 * np.pi * 1800 * t) * np.exp(-6 * t) + 0.55 * np.sin(2 * np.pi * 3200 * t) * np.exp(-10 * t)
            fault_data.append(delayed_base + harmonics + linkage_friction + white_noise + grid_hum)

    return np.array(normal_data), np.array(fault_data)

def sliding_window_segmentation(signals, labels, window_size=1000, stride=300):
    x_segments, y_segments = [], []
    for sig, label in zip(signals, labels):
        num_windows = (len(sig) - window_size) // stride + 1
        for i in range(num_windows):
            start = i * stride
            x_segments.append(sig[start:start + window_size])
            y_segments.append(label)
    x_arr = np.expand_dims(np.array(x_segments, dtype=np.float32), axis=1)
    y_arr = np.array(y_segments, dtype=np.int64)
    return x_arr, y_arr

class VQCLayer(nn.Module):
    def __init__(self, num_qubits=4, num_layers=2):
        super(VQCLayer, self).__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.var_weights = nn.Parameter(torch.randn(num_layers, num_qubits, 2) * 0.1)

    def forward(self, x_classical):
        batch_size = x_classical.size(0)
        theta = torch.tanh(x_classical) * np.pi
        q_states_0 = torch.ones(batch_size, self.num_qubits, device=x_classical.device)
        q_states_1 = torch.zeros(batch_size, self.num_qubits, device=x_classical.device)

        for layer in range(self.num_layers):
            new_q0, new_q1 = [], []
            for q in range(self.num_qubits):
                angle_y = theta[:, q] + self.var_weights[layer, q, 0]
                angle_z = self.var_weights[layer, q, 1]
                cos_y, sin_y = torch.cos(angle_y / 2.0), torch.sin(angle_y / 2.0)
                s0, s1 = q_states_0[:, q], q_states_1[:, q]
                n_s0 = cos_y * s0 - sin_y * s1
                n_s1 = torch.cos(angle_z / 2.0) * (sin_y * s0 + cos_y * s1)
                new_q0.append(n_s0)
                new_q1.append(n_s1)
            q_states_0 = torch.stack(new_q0, dim=1)
            q_states_1 = torch.stack(new_q1, dim=1)

            cnot_q1 = [q_states_1[:, 0]]
            for q in range(1, self.num_qubits):
                cnot_q1.append(q_states_1[:, q] * q_states_0[:, q - 1])
            q_states_1 = torch.stack(cnot_q1, dim=1)

        p0, p1 = q_states_0 ** 2, q_states_1 ** 2
        return (p0 - p1) / (p0 + p1 + 1e-8)

class VQC_QNetwork(nn.Module):
    def __init__(self, input_channels=1, num_actions=2):
        super(VQC_QNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.MaxPool1d(2, 2)
        )
        self.fc_compress = nn.Linear(32 * 62, 4)
        self.vqc = VQCLayer(num_qubits=4, num_layers=2)
        self.q_out = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, num_actions)
        )

    def forward(self, x):
        features = self.feature_extractor(x).view(x.size(0), -1)
        compressed = self.fc_compress(features)
        quantum_features = self.vqc(compressed)
        return self.q_out(quantum_features)

class ReplayBuffer:
    def __init__(self, capacity=8000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        s, a, r, ns, d = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(np.array(ns), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)