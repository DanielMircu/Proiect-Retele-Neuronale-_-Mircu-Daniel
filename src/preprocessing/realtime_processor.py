"""
Module pentru procesarea în timp real a datelor.

Aceste utilitare sunt folosite pentru a menține un buffer cu ultimele N sample-uri și
pentru a executa inferența pe o fereastră completă (folosind același extractor de features
ca în pipeline offline).
"""
import numpy as np
import torch
from collections import deque
from .signal_processor import extract_features


class RealTimeBuffer:
    """Buffer circular pentru menținerea ultimelor N sample-uri.

    - `window_size` este dimensiunea ferestrei pentru inferență
    - `n_features` este numărul de canale/senzori per sample

    Buffer-ul este inițializat cu zerouri astfel încât `is_ready()` să fie True
    doar după ce s-au adăugat suficiente sample-uri reale (sau dacă s-au înlocuit zerourile).
    """

    def __init__(self, window_size=200, n_features=10):
        """Inițializează buffer-ul și îl umple cu zerouri."""
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.n_features = n_features

        # Umple cu zerouri pentru a păstra forma corectă până apar date reale
        for _ in range(window_size):
            self.buffer.append(np.zeros(n_features))

    def add_sample(self, sample):
        """Adaugă un sample nou în buffer (vector de dimensiune n_features)."""
        self.buffer.append(sample)

    def get_window(self):
        """Returnează buffer-ul curent ca un array numpy (window_size x n_features)."""
        return np.array(self.buffer)

    def is_ready(self):
        """Verifică dacă buffer-ul conține `window_size` elemente reale/inițiale."""
        return len(self.buffer) == self.window_size


def process_single_window_rt(window, model):
    """Procesează o fereastră completă (realtime) și returnează predicția.

    Pași:
    1. Extragere features (aceleași ca în pipeline offline)
    2. Convertire la tensor și inferență cu modelul (mod eval)
    3. Returnare index de predicție și vector de probabilități pentru fiecare clasă

    Args:
        window (np.ndarray): Array (window_size x n_features)
        model (torch.nn.Module): Model antrenat

    Returnează:
        tuple: (prediction_index (int), probabilities (np.ndarray de dimensiune n_classes))
    """
    # Extragem feature-urile statistice din fereastră
    features = extract_features(window)

    # Inference: asigurăm modul eval și dezactivăm gradientul
    model.eval()
    with torch.no_grad():
        tensor_in = torch.FloatTensor(features).unsqueeze(0)  # Adăugăm dimensiunea batch (1, n_features)
        output = model(tensor_in)  # logits
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    # Returnăm indexul clasei și vectorul de probabilități (convertit la numpy)
    return pred_idx, probs.numpy()[0]