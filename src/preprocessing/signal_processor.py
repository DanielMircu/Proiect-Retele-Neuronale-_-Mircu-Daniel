"""
Module pentru preprocesarea semnalelor de telemetrie.

Scop: aplicăm filtrare, normalizare, segmentare în ferestre și extragem
caracteristici statistice (features) per fereastră care vor fi folosite de model.
"""
import numpy as np
from scipy import signal


def butterworth_filter(data, cutoff=10, fs=100, order=4):
    """Aplică un filtru Butterworth low-pass pentru a elimina zgomotul de înaltă frecvență.

    Args:
        data (np.ndarray): Vector 1D cu valori temporale ale unui canal
        cutoff (float): Frecvența de tăiere (Hz)
        fs (float): Frecvența de eșantionare (Hz)
        order (int): Ordinul filtrului

    Returnează:
        np.ndarray: Datele filtrate (același shape)
    """
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low')
    # filtfilt aplică filtrul înainte și înapoi pentru a evita fazarea semnalului
    return signal.filtfilt(b, a, data)


def create_windows(data, window_size=200, overlap=0.5):
    """Creează ferestre suprapuse dintr-un array 2D (timp x canale).

    Args:
        data (np.ndarray): Array 2D cu shape (n_samples, n_channels)
        window_size (int): Numărul de sample-uri per fereastră
        overlap (float): Proporția de suprapunere (0-1)

    Returnează:
        np.ndarray: Array 3D cu shape (n_windows, window_size, n_channels)

    Observație: bucla folosește `range(0, len(data) - window_size, step)` -> ultima
    porțiune necompletă este ignorată.
    """
    step = int(window_size * (1 - overlap))
    windows = []

    for i in range(0, len(data) - window_size, step):
        window = data[i:i+window_size]
        windows.append(window)

    return np.array(windows)


def extract_features(window):
    """Extrage feature-uri statistice dintr-o fereastră pentru fiecare canal.

    Pentru fiecare canal se calculează: mean, std, min, max, RMS, peak-to-peak.

    Args:
        window (np.ndarray): Array 2D (samples x channels) sau 1D pentru un singur canal

    Returnează:
        np.ndarray: Vector 1D cu features concatenate pentru toate canalele
    """
    features = []

    # Asigurăm formatul 2D: (samples, channels)
    if window.ndim == 1:
        window = window.reshape(-1, 1)

    for col in range(window.shape[1]):
        channel = window[:, col]

        # Extindem lista de features cu valori simple, ușor de interpretat
        features.extend([
            np.mean(channel),             # Mean
            np.std(channel),              # Std deviation
            np.min(channel),              # Min
            np.max(channel),              # Max
            np.sqrt(np.mean(channel**2)), # RMS (root mean square)
            np.ptp(channel)               # Peak-to-peak
        ])

    return np.array(features)


def preprocess_telemetry(df, window_size=200, overlap=0.5):
    """Pipeline complet de preprocesare folosit de pagina Evaluate.

    Pași:
    1. Selectăm coloanele de senzori relevante
    2. Aplicăm filtrare Butterworth per canal
    3. Normalizăm (zero mean, unit std)
    4. Segmentăm în ferestre suprapuse
    5. Extragem feature-uri per fereastră

    Args:
        df (pd.DataFrame): DataFrame cu telemetrie (coloane: susp_*, acc_*, rot_*)
        window_size (int): Număr de sample-uri per fereastră
        overlap (float): Proporția de suprapunere între ferestre

    Returnează:
        np.ndarray: Array 2D (n_windows x n_features)
    """
    # Coloanele așteptate din CSV (ordinea contează pentru extragere)
    sensor_cols = ['susp_fl', 'susp_fr', 'susp_rl', 'susp_rr', 
                   'acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z']

    # 2) Filtrare per canal -> build matrix (n_samples x n_channels)
    filtered_data = np.zeros((len(df), len(sensor_cols)))
    for i, col in enumerate(sensor_cols):
        filtered_data[:, i] = butterworth_filter(df[col].values)

    # 3) Normalizare per canal (evităm divizarea la zero cu epsilon mic)
    mean = filtered_data.mean(axis=0)
    std = filtered_data.std(axis=0)
    normalized_data = (filtered_data - mean) / (std + 1e-8)

    # 4) Segmentare în ferestre
    windows = create_windows(normalized_data, window_size, overlap)

    # 5) Extragere features pentru fiecare fereastră
    features_list = []
    for window in windows:
        features = extract_features(window)
        features_list.append(features)

    return np.array(features_list)