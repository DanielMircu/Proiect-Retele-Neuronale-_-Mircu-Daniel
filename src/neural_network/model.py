"""
Model  pentru clasificarea comportamentului de suspensie

Acest modul definește un MLP simplu (Multi-Layer Perceptron) folosit pentru a
clasifica ferestre de telemetrie în două clase: understeer sau oversteer.

Observații:
- `input_size` trebuie să corespundă numărului de feature-uri generate de
  preprocessare pentru o singură fereastră.
- Output-ul este un tensor de dimensiune (batch_size x 2) reprezentând logits
  pentru cele două clase; aplicarea softmax se face la evaluator (nu aici).
"""
import torch
import torch.nn as nn


class SuspensionClassifier(nn.Module):
    """Clasificator MLP pentru detectarea understeer/oversteer.

    Structura este construită dintr-o listă de layere complet conectate (Linear),
    cu activare ReLU și Dropout între ele, pentru a reduce overfitting-ul.
    """

    def __init__(self, input_size=60, hidden_sizes=[32, 16], output_size=2):
        """Initializează arhitectura.

        Args:
            input_size (int): Dimensiune feature vector de intrare
            hidden_sizes (list[int]): Dimensiuni pentru layerele ascunse
            output_size (int): Număr de clase (2)
        """
        super().__init__()

        # Construim liste dinamice de layere pe baza `hidden_sizes`
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            # Linear -> ReLU -> Dropout (regularizare)
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size

        # Layer final care produce logits pentru fiecare clasă
        layers.append(nn.Linear(prev_size, output_size))

        # Folosim Sequential pentru un forward simplu
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass.

        Parametri:
            x (torch.Tensor): Tensor de formă (batch_size, input_size)

        Returnează:
            torch.Tensor: Logits de formă (batch_size, output_size)
        """
        return self.network(x)
