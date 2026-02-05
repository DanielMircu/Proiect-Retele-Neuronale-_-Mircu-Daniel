"""
Module pentru evaluarea modelului și generarea recomandărilor

Funcționalitate:
- Primesc un model PyTorch și un set de feature-uri (un array de ferestre)
- Returnez un dict cu concluzii (behavior, confidence), metrici de distribuție și
  recomandări practice

Observație: Etichetele sunt codificate implicit astfel: 0 = understeer, 1 = oversteer
"""
import torch
import numpy as np


def evaluate_telemetry(model, features):
    """Evaluează telemetria și generează recomandări.

    Pași:
    1. Transformă features într-un Tensor Torch
    2. Rulează modelul în modul eval și aplică softmax pentru a obține probabilități
    3. Determină predicțiile (argmax) pentru fiecare fereastră
    4. Calculează ratio-urile (understeer/oversteer) și behaviour dominant
    5. Generează recomandări bazate pe nivelul de confidence

    Args:
        model (torch.nn.Module): Model antrenat
        features (array-like): Array de shape (n_windows, n_features)

    Returnează:
        dict: Rezultate de evaluare (vezi cod pentru chei)
    """
    model.eval()

    # 1) Convertim la tensor float
    X = torch.FloatTensor(features)

    # 2) Forward fără gradient (mai rapid și sigur în memorie)
    with torch.no_grad():
        outputs = model(X)  # logits (n_windows x n_classes)
        probs = torch.softmax(outputs, dim=1)  # probabilități normalizate pe clase
        predictions = torch.argmax(probs, dim=1)  # clasa pick-uită pentru fiecare fereastră

    # 3) Statistici simple pe ferestre
    n_windows = len(predictions)
    # Numărăm câte ferestre au fost etichetate 0 (understeer) sau 1 (oversteer)
    n_understeer = (predictions == 0).sum().item()
    n_oversteer = (predictions == 1).sum().item()

    # 4) Ratio-uri (proporții)
    understeer_ratio = n_understeer / n_windows if n_windows > 0 else 0.0
    oversteer_ratio = n_oversteer / n_windows if n_windows > 0 else 0.0

    # Behavior dominant = clasa cu proporția mai mare
    if understeer_ratio > oversteer_ratio:
        behavior = "understeer"
        confidence = understeer_ratio
    else:
        behavior = "oversteer"
        confidence = oversteer_ratio

    # 5) Generare recomandări (mesaj și acțiuni)
    recommendations = _generate_recommendations(behavior, confidence)

    # Returnăm structura cu rezultate și date utile pentru vizualizări
    return {
        'behavior': behavior,
        'confidence': confidence,
        'n_windows': n_windows,
        'understeer_ratio': understeer_ratio,
        'oversteer_ratio': oversteer_ratio,
        'predictions': predictions.numpy(),            # array de 0/1 per fereastră
        'probabilities': probs.numpy(),                # probabilități per clasă per fereastră
        'recommendations': recommendations
    }


def _generate_recommendations(behavior, confidence):
    """Generează recomandări bazate pe behavior și nivelul de confidence.

    Logică simplă:
    - Dacă confidence > 0.6, oferim recomandări specifice fiecărei clase
    - Altfel, recomandăm să se colecteze mai multe date sau verificări hardware
    """
    if confidence > 0.6:
        if behavior == "understeer":
            return {
                'message': "UNDERSTEER DETECTAT",
                'actions': [
                    "Crește camber negativ față (ex: -1.5° -> -2.0°)",
                    "Crește toe-out față (ex: 0° -> 0.1° per roată)",
                    "Reduce camber spate",
                    "Scade presiunea pneuri față"
                ]
            }
        else:
            return {
                'message': "OVERSTEER DETECTAT",
                'actions': [
                    "Crește camber negativ spate (ex: -1.0° -> -1.5°)",
                    "Reduce toe-out față",
                    "Reduce camber față",
                    "Scade presiunea pneuri spate"
                ]
            }
    else:
        return {
            'message': "CONFIDENCE SCĂZUTĂ",
            'actions': [
                "Colectează mai multe date",
                "Verifică calibrarea senzorilor"
            ]
        }