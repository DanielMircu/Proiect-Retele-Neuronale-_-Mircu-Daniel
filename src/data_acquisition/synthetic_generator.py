"""
Module pentru generarea de date sintetice de telemetrie.

Funcția `generate_synthetic_telemetry` poate fi folosită pentru a crea fișiere
CSV sau DataFrame-uri pentru testare, debugging sau antrenare inițială.
"""
import numpy as np
import pandas as pd


def generate_synthetic_telemetry(duration_sec=60, sampling_rate=50, behavior='neutral'):
    """Generează telemetrie sintetică.

    Parametri:
        duration_sec (int): Durata simulării în secunde
        sampling_rate (int): Rata de eșantionare (Hz)
        behavior (str): 'neutral', 'understeer' sau 'oversteer'

    Returnează:
        pd.DataFrame: Colonele sunt aceleași așteptate de pipeline-ul de preprocesare
                      (time, susp_*, acc_*, rot_*)

    Observații:
    - Semnalele sunt construite din combinații de sinusoide și zgomot gaussian,
      iar modificările pentru 'understeer' / 'oversteer' sunt aplicate prin scalare
      și offset pe anumite componente (suspensie, rotații, accelerații).
    """
    n_samples = int(duration_sec * sampling_rate)
    t = np.linspace(0, duration_sec, n_samples)

    # Suspensie:
    road = 0.02 * np.sin(2 * np.pi * 0.5 * t)  # componentă periodică (bump)
    road += 0.005 * np.random.randn(n_samples)  # zgomot

    cornering = 0.03 * np.sin(2 * np.pi * 0.1 * t)  # componentă mai lentă pentru viraj

    # Generăm semnalele pentru cele 4 colțuri
    susp_fl = road + cornering
    susp_fr = road - cornering
    susp_rl = road + cornering * 0.8
    susp_rr = road - cornering * 0.8

    # Modificări specifice comportamentelor pentru a simula under/oversteer
    if behavior == 'understeer':
        # În understeer, partea față reacționează mai mult la cornering
        susp_fl += cornering * 0.5
        susp_fr -= cornering * 0.5
    elif behavior == 'oversteer':
        # În oversteer, partea spate reacționează mai mult
        susp_rl += cornering * 0.5
        susp_rr -= cornering * 0.5

    # Accelerații:
    acc_x = 0.3 * np.sin(2 * np.pi * 0.15 * t) + 0.1 * np.random.randn(n_samples)
    acc_y = 0.5 * np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(n_samples)

    # Ajustăm magnitudinea laterală a accelerațiilor în funcție de comportament
    if behavior == 'understeer':
        acc_y *= 0.8
    elif behavior == 'oversteer':
        acc_y *= 1.2

    acc_z = 9.81 + 0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.2 * np.random.randn(n_samples)

    # Rotații
    rot_x = 0.1 * np.sin(2 * np.pi * 0.2 * t) + 0.02 * np.random.randn(n_samples)
    rot_y = 0.05 * np.sin(2 * np.pi * 0.15 * t) + 0.01 * np.random.randn(n_samples)
    rot_z = 0.15 * np.sin(2 * np.pi * 0.1 * t) + 0.03 * np.random.randn(n_samples)

    # Ajustare rotație yaw pentru a accentua under/oversteer
    if behavior == 'understeer':
        rot_z *= 0.7
    elif behavior == 'oversteer':
        rot_z *= 1.3

    # Compunere DataFrame cu coloanele așteptate de restul pipeline-ului
    df = pd.DataFrame({
        'time': t,
        'susp_fl': susp_fl,
        'susp_fr': susp_fr,
        'susp_rl': susp_rl,
        'susp_rr': susp_rr,
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
        'rot_x': rot_x,
        'rot_y': rot_y,
        'rot_z': rot_z
    })

    return df