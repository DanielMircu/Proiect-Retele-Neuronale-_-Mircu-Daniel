# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Mircu Daniel  
**Link Repository GitHub:** [\[URL complet\]  ](https://github.com/DanielMircu/Proiect-Retele-Neuronale-_-Mircu-Daniel.git)
**Data predării:** 12/12/25

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Obiectiv principal:** Antrenarea efectivă a modelului RN pentru detectarea comportamentului understeer/oversteer al monopostului Formula Student, evaluarea performanței și integrarea în aplicația completă.

**Pornire obligatorie:** Arhitectura completă și funcțională din Etapa 4:
- State Machine definit și justificat
- Cele 3 module funcționale (Data Acquisition, Neural Network, UI)
- Dataset complet cu date de la senzori
---

## PREREQUISITE – Verificare Etapa 4 (OBLIGATORIU)

**Înainte de a începe Etapa 5, verificați că aveți din Etapa 4:**

- [x] **State Machine** definit și documentat în `docs/state_machine.*`
- [x] **Contribuție ≥40% date originale** în `data/generated/` (verificabil)
- [x] **Modul 1 (Data Logging)** funcțional - produce CSV-uri
- [x] **Modul 2 (RN)** cu arhitectură definită dar NEANTRENATĂ (`models/untrained_model.h5`)
- [x] **Modul 3 (UI/Web Service)** funcțional cu model dummy
- [x] **Tabelul "Nevoie → Soluție → Modul"** complet în README Etapa 4

** Dacă oricare din punctele de mai sus lipsește → reveniți la Etapa 4 înainte de a continua.**

---

## Pregătire Date pentru Antrenare 

### Dataset și Preprocesare

**Sursă date:** Telemetrie sintetică generată prin `generate_synthetic_telemetry()`:
- **10 canale senzori:** susp_fl, susp_fr, susp_rl, susp_rr, acc_x, acc_y, acc_z, rot_x, rot_y, rot_z
- **Frecvență eșantionare:** 100 Hz
- **Comportamente:** understeer (label=0), oversteer (label=1)

**Pipeline preprocesare aplicat:**

```python
# 1. Filtrare Butterworth low-pass (10 Hz, order=4)
filtered_data = butterworth_filter(raw_data, cutoff=10, fs=100, order=4)

# 2. Normalizare Z-score
normalized_data = (filtered_data - mean) / (std + 1e-8)

# 3. Windowing (200 samples, overlap 50%)
windows = create_windows(normalized_data, window_size=200, overlap=0.5)

# 4. Feature extraction (6 features x 10 canale = 60 features)
features = extract_features(window)  # mean, std, min, max, RMS, peak-to-peak
```

**Split date:**
- **Train:** 80% din dataset
- **Validation:** 20% din dataset (split automat în `train_model()`)

**Verificare consistență:** 
- Același `butterworth_filter` cu parametri identici
- Aceeași dimensiune window (200 samples)
- Același număr features (60) pentru toate batch-urile

---

##  Cerințe Structurate pe 3 Niveluri

### Nivel 1 – Obligatoriu pentru Toți (70% din punctaj)

### 1. Antrenare Model

**Arhitectură implementată:**
```python
class SuspensionClassifier(nn.Module):
    Input: 60 features
    Hidden Layer 1: 32 neuroni + ReLU + Dropout(0.3)
    Hidden Layer 2: 16 neuroni + ReLU + Dropout(0.3)
    Output: 2 clase (Understeer=0, Oversteer=1)
```

**Antrenare executată:**
- ✅ Minimum 10 epoci (configurabil până la 100)
- ✅ Batch size: 32 (default, ajustabil 16-128)
- ✅ Optimizer: learning rate 0.001
- ✅ Loss function: CrossEntropyLoss (clasificare binară)

### 2. Tabel Hiperparametri și Justificări

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| Learning rate | 0.001 | Valoare standard pentru Adam optimizer, asigură convergență stabilă pentru MLP de dimensiune medie. Testat cu 0.0005 și 0.005 - 0.001 oferă cel mai bun echilibru viteză/stabilitate. |
| Batch size | 32 | Compromis memorie/stabilitate pentru dataset de ~10,000+ samples. Batch 32 → ~300-500 iterații/epocă. Batch mai mic (16) aduce zgomot în gradient, batch mai mare (64+) încetinește convergența. |
| Number of epochs | 30 (default) | Suficient pentru convergență pe date sintetice. Monitorizare val_loss pentru evitare overfitting.  |
| Loss function | CrossEntropyLoss | Clasificare binară (understeer vs oversteer). |
| Activation functions | ReLU (hidden), Implicit softmax (output) | ReLU pentru non-linearitate fără vanishing gradient. Softmax implicit în CrossEntropyLoss pentru probabilități normalizate clase. |
| Hidden layers | [32, 16] | Reducere progresivă 60 → 32 → 16 → 2. Suficient pentru features statistice simple (mean, std, RMS).|

**Justificare detaliată batch size:**
```
Am ales batch_size=32 pentru că:
- Dataset: ~200 samples x ~50 windows/sample = ~10,000 ferestre
- 10,000/32 ≈ 312 iterații/epocă

Echilibru:
✓ Stabilitate gradient: Batch 32 reduce varianța gradientului vs batch 8-16
✓ Memorie: 32 x 60 features = 1,920 valori/batch → neglijabil pentru RAM
✓ Viteză: 312 iterații/epocă asigură convergență în ~30 epoci (~2 minute pe CPU)
✗ Batch 64+: Convergență mai lentă, risc "wide minima" (generalizare slabă)
```

---

### Nivel 2 – Recomandat (85-90% din punctaj)

Includeți **TOATE** cerințele Nivel 1 + următoarele:

1. **Early Stopping** - oprirea antrenării dacă `val_loss` nu scade în 5 epoci consecutive
2. **Learning Rate Scheduler** - `ReduceLROnPlateau` sau `StepLR`
### 3. Augmentări Date

**Augmentări aplicate în `generate_synthetic_telemetry()`:**

**Noise Gaussian calibrat:**
```python
road += 0.005 * np.random.randn(n_samples)  # Vibrații drum
acc_x += 0.1 * np.random.randn(n_samples)   # Noise accelerometru
```

**Variații parametrice:**
```python
if behavior == 'understeer':
    susp_fl += cornering * 0.5  # Accent suspensie față
    acc_y *= 0.8                 # Lateral G redus
elif behavior == 'oversteer':
    susp_rl += cornering * 0.5  # Accent suspensie spate
    acc_y *= 1.2                 # Lateral G amplificat
```

### 4. Grafic Loss și Val_Loss

**Implementat în UI (Generate & Train page):**

```python
fig = go.Figure()
fig.add_trace(go.Scatter(y=history['train_loss'], name='Train Loss'))
fig.add_trace(go.Scatter(y=history['val_loss'], name='Validation Loss'))
fig.add_trace(go.Scatter(y=history['val_acc'], name='Val Accuracy', yaxis='y2'))
st.plotly_chart(fig)
```

5. **Analiză erori context industrial**

**Indicatori țintă Nivel 2:**
- **Acuratețe ≥ 75%**
- **F1-score (macro) ≥ 0.70**
---

### Nivel 3 – Bonus (până la 100%)

**Punctaj bonus per activitate:**

| **Activitate** |  **Livrabil** |
|----------------|--------------|
| Comparare 2+ arhitecturi diferite | Tabel comparativ + justificare alegere finală în README |
| Export ONNX/TFLite + benchmark latență | Fișier `models/final_model.onnx` + demonstrație <50ms |
| Confusion Matrix + analiză 5 exemple greșite | `docs/confusion_matrix.png` + analiză în README |

---

## Verificare Consistență cu State Machine (Etapa 4)

Antrenarea și inferența trebuie să respecte fluxul din State Machine-ul vostru definit în Etapa 4.

**Exemplu pentru monitorizare vibrații lagăr:**

| **Stare din Etapa 4** | **Implementare în Etapa 5** |
|-----------------------|-----------------------------|
| `ACQUIRE_DATA` | Citire batch date din `data/train/` pentru antrenare |
| `PREPROCESS` | Aplicare scaler salvat din `config/preprocessing_params.pkl` |
| `RN_INFERENCE` | Forward pass cu model ANTRENAT (nu weights random) |
| `THRESHOLD_CHECK` | Clasificare Normal/Uzură pe baza output RN antrenat |
| `ALERT` | Trigger în UI bazat pe predicție modelului real |

**În `src/app/main.py` (UI actualizat):**

Verificați că **TOATE stările** din State Machine sunt implementate cu modelul antrenat:

```python
# ÎNAINTE (Etapa 4 - model dummy):
model = keras.models.load_model('models/untrained_model.h5')  # weights random
prediction = model.predict(input_scaled)  # output aproape aleator

# ACUM (Etapa 5 - model antrenat):
model = keras.models.load_model('models/trained_model.h5')  # weights antrenate
prediction = model.predict(input_scaled)  # predicție REALĂ și corectă
```

---

## Analiză Erori în Context Industrial (OBLIGATORIU Nivel 2)

**Nu e suficient să raportați doar acuratețea globală.** Analizați performanța în contextul aplicației voastre industriale:

### 1. Pe ce clase greșește cel mai mult modelul?

**Exemplu robotică (predicție traiectorii):**
```
Confusion Matrix arată că modelul confundă 'viraj stânga' cu 'viraj dreapta' în 18% din cazuri.
Cauză posibilă: Features-urile IMU (gyro_z) sunt simetrice pentru viraje în direcții opuse.
```

**Completați pentru proiectul vostru:**
```
[Descrieți confuziile principale între clase și cauzele posibile]
```

### 2. Ce caracteristici ale datelor cauzează erori?

**Exemplu vibrații motor:**
```
Modelul eșuează când zgomotul de fond depășește 40% din amplitudinea semnalului util.
În mediul industrial, acest nivel de zgomot apare când mai multe motoare funcționează simultan.
```

**Completați pentru proiectul vostru:**
```
[Identificați condițiile în care modelul are performanță slabă]
```

### 3. Ce implicații are pentru aplicația industrială?

**Exemplu detectare defecte sudură:**
```
FALSE NEGATIVES (defect nedetectat): CRITIC → risc rupere sudură în exploatare
FALSE POSITIVES (alarmă falsă): ACCEPTABIL → piesa este re-inspectată manual

Prioritate: Minimizare false negatives chiar dacă cresc false positives.
Soluție: Ajustare threshold clasificare de la 0.5 → 0.3 pentru clasa 'defect'.
```

**Completați pentru proiectul vostru:**
```
[Analizați impactul erorilor în contextul aplicației voastre și prioritizați]
```

### 4. Ce măsuri corective propuneți?

**Exemplu clasificare imagini piese:**
```
Măsuri corective:
1. Colectare 500+ imagini adiționale pentru clasa minoritară 'zgârietură ușoară'
2. Implementare filtrare Gaussian blur pentru reducere zgomot cameră industrială
3. Augmentare perspective pentru simulare unghiuri camera variabile (±15°)
4. Re-antrenare cu class weights: [1.0, 2.5, 1.2] pentru echilibrare
```

**Completați pentru proiectul vostru:**
```
[Propuneți minimum 3 măsuri concrete pentru îmbunătățire]
```

---

## Structura Repository-ului la Finalul Etapei 5

**Clarificare organizare:** Vom folosi **README-uri separate** pentru fiecare etapă în folderul `docs/`:

```
proiect-rn-[prenume-nume]/
├── README.md                           # Overview general proiect (actualizat)
├── etapa3_analiza_date.md         # Din Etapa 3
├── etapa4_arhitectura_sia.md      # Din Etapa 4
├── etapa5_antrenare_model.md      # ← ACEST FIȘIER (completat)
│
├── docs/
│   ├── state_machine.png              # Din Etapa 4
│   ├── loss_curve.png                 # NOU - Grafic antrenare
│   ├── confusion_matrix.png           # (opțional - Nivel 3)
│   └── screenshots/
│       ├── inference_real.png         # NOU - OBLIGATORIU
│       └── ui_demo.png                # Din Etapa 4
│
├── data/                               # Din Etapa 3-4 (NESCHIMBAT)
│   ├── raw/
│   ├── generated/                     # Contribuția voastră 40%
│   ├── processed/
│   ├── train/
│   ├── validation/
│   └── test/
│
├── src/
│   ├── data_acquisition/              # Din Etapa 4
│   ├── preprocessing/                 # Din Etapa 3
│   │   └── combine_datasets.py        # NOU (dacă ați adăugat date în Etapa 4)
│   ├── neural_network/
│   │   ├── model.py                   # Din Etapa 4
│   │   ├── train.py                   # NOU - Script antrenare
│   │   └── evaluate.py                # NOU - Script evaluare
│   └── app/
│       └── main.py                    # ACTUALIZAT - încarcă model antrenat
│
├── models/
│   ├── untrained_model.h5             # Din Etapa 4
│   ├── trained_model.h5               # NOU - OBLIGATORIU
│   └── final_model.onnx               # (opțional - Nivel 3 bonus)
│
├── results/                            # NOU - Folder rezultate antrenare
│   ├── training_history.csv           # OBLIGATORIU - toate epoch-urile
│   ├── test_metrics.json              # Metrici finale pe test set
│   └── hyperparameters.yaml           # Hiperparametri folosiți
│
├── config/
│   └── preprocessing_params.pkl       # Din Etapa 3 (NESCHIMBAT)
│
├── requirements.txt                    # Actualizat
└── .gitignore
```

**Diferențe față de Etapa 4:**
- Adăugat `docs/etapa5_antrenare_model.md` (acest fișier)
- Adăugat `docs/loss_curve.png` (Nivel 2)
- Adăugat `models/trained_model.h5` - OBLIGATORIU
- Adăugat `results/` cu history și metrici
- Adăugat `src/neural_network/train.py` și `evaluate.py`
- Actualizat `src/app/main.py` să încarce model antrenat

---

## Instrucțiuni de Rulare (Actualizate față de Etapa 4)

### 1. Setup mediu (dacă nu ați făcut deja)

```bash
pip install -r requirements.txt
```

### 2. Pregătire date (DACĂ ați adăugat date noi în Etapa 4)

```bash
# Combinare + reprocesare dataset complet
python src/preprocessing/combine_datasets.py
python src/preprocessing/data_cleaner.py
python src/preprocessing/feature_engineering.py
python src/preprocessing/data_splitter.py --stratify --random_state 42
```

### 3. Antrenare model

```bash
python src/neural_network/train.py --epochs 50 --batch_size 32 --early_stopping

# Output așteptat:
# Epoch 1/50 - loss: 0.8234 - accuracy: 0.6521 - val_loss: 0.7891 - val_accuracy: 0.6823
# ...
# Epoch 23/50 - loss: 0.3456 - accuracy: 0.8234 - val_loss: 0.4123 - val_accuracy: 0.7956
# Early stopping triggered at epoch 23
# ✓ Model saved to models/trained_model.h5
```

### 4. Evaluare pe test set

```bash
python src/neural_network/evaluate.py --model models/trained_model.h5

# Output așteptat:
# Test Accuracy: 0.7823
# Test F1-score (macro): 0.7456
# ✓ Metrics saved to results/test_metrics.json
# ✓ Confusion matrix saved to docs/confusion_matrix.png
```

### 5. Lansare UI cu model antrenat

```bash
streamlit run src/app/main.py

# SAU pentru LabVIEW:
# Deschideți WebVI și rulați main.vi
```

**Testare în UI:**
1. Introduceți date de test (manual sau upload fișier)
2. Verificați că predicția este DIFERITĂ de Etapa 4 (când era random)
3. Verificați că confidence scores au sens (ex: 85% pentru clasa corectă)
4. Faceți screenshot → salvați în `docs/screenshots/inference_real.png`

---

## Checklist Final – Bifați Totul Înainte de Predare

### Prerequisite Etapa 4 (verificare)
- [ ] State Machine există și e documentat în `docs/state_machine.*`
- [ ] Contribuție ≥40% date originale verificabilă în `data/generated/`
- [ ] Cele 3 module din Etapa 4 funcționale

### Preprocesare și Date
- [ ] Dataset combinat (vechi + nou) preprocesat (dacă ați adăugat date)
- [ ] Split train/val/test: 70/15/15% (verificat dimensiuni fișiere)
- [ ] Scaler din Etapa 3 folosit consistent (`config/preprocessing_params.pkl`)

### Antrenare Model - Nivel 1 (OBLIGATORIU)
- [ ] Model antrenat de la ZERO (nu fine-tuning pe model pre-antrenat)
- [ ] Minimum 10 epoci rulate (verificabil în `results/training_history.csv`)
- [ ] Tabel hiperparametri + justificări completat în acest README
- [ ] Metrici calculate pe test set: **Accuracy ≥65%**, **F1 ≥0.60**
- [ ] Model salvat în `models/trained_model.h5` (sau .pt, .lvmodel)
- [ ] `results/training_history.csv` există cu toate epoch-urile

### Integrare UI și Demonstrație - Nivel 1 (OBLIGATORIU)
- [ ] Model ANTRENAT încărcat în UI din Etapa 4 (nu model dummy)
- [ ] UI face inferență REALĂ cu predicții corecte
- [ ] Screenshot inferență reală în `docs/screenshots/inference_real.png`
- [ ] Verificat: predicțiile sunt diferite față de Etapa 4 (când erau random)

### Documentație Nivel 2 (dacă aplicabil)
- [ ] Early stopping implementat și documentat în cod
- [ ] Learning rate scheduler folosit (ReduceLROnPlateau / StepLR)
- [ ] Augmentări relevante domeniu aplicate (NU rotații simple!)
- [ ] Grafic loss/val_loss salvat în `docs/loss_curve.png`
- [ ] Analiză erori în context industrial completată (4 întrebări răspunse)
- [ ] Metrici Nivel 2: **Accuracy ≥75%**, **F1 ≥0.70**

### Documentație Nivel 3 Bonus (dacă aplicabil)
- [ ] Comparație 2+ arhitecturi (tabel comparativ + justificare)
- [ ] Export ONNX/TFLite + benchmark latență (<50ms demonstrat)
- [ ] Confusion matrix + analiză 5 exemple greșite cu implicații

### Verificări Tehnice
- [ ] `requirements.txt` actualizat cu toate bibliotecile noi
- [ ] Toate path-urile RELATIVE (nu absolute: `/Users/...` )
- [ ] Cod nou comentat în limba română sau engleză (minimum 15%)
- [ ] `git log` arată commit-uri incrementale (NU 1 commit gigantic)
- [ ] Verificare anti-plagiat: toate punctele 1-5 respectate

### Verificare State Machine (Etapa 4)
- [ ] Fluxul de inferență respectă stările din State Machine
- [ ] Toate stările critice (PREPROCESS, INFERENCE, ALERT) folosesc model antrenat
- [ ] UI reflectă State Machine-ul pentru utilizatorul final

### Pre-Predare
- [ ] `docs/etapa5_antrenare_model.md` completat cu TOATE secțiunile
- [ ] Structură repository conformă: `docs/`, `results/`, `models/` actualizate
- [ ] Commit: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
- [ ] Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
- [ ] Push: `git push origin main --tags`
- [ ] Repository accesibil (public sau privat cu acces profesori)

---

## Livrabile Obligatorii (Nivel 1)

Asigurați-vă că următoarele fișiere există și sunt completate:

1. **`docs/etapa5_antrenare_model.md`** (acest fișier) cu:
   - Tabel hiperparametri + justificări (complet)
   - Metrici test set raportate (accuracy, F1)
   - (Nivel 2) Analiză erori context industrial (4 paragrafe)

2. **`models/trained_model.h5`** (sau `.pt`, `.lvmodel`) - model antrenat funcțional

3. **`results/training_history.csv`** - toate epoch-urile salvate

4. **`results/test_metrics.json`** - metrici finale:

Exemplu:
```json
{
  "test_accuracy": 0.7823,
  "test_f1_macro": 0.7456,
  "test_precision_macro": 0.7612,
  "test_recall_macro": 0.7321
}
```

5. **`docs/screenshots/inference_real.png`** - demonstrație UI cu model antrenat

6. **(Nivel 2)** `docs/loss_curve.png` - grafic loss vs val_loss

7. **(Nivel 3)** `docs/confusion_matrix.png` + analiză în README

---

## Predare și Contact

**Predarea se face prin:**
1. Commit pe GitHub: `"Etapa 5 completă – Accuracy=X.XX, F1=X.XX"`
2. Tag: `git tag -a v0.5-model-trained -m "Etapa 5 - Model antrenat"`
3. Push: `git push origin main --tags`

---

**Mult succes! Această etapă demonstrează că Sistemul vostru cu Inteligență Artificială (SIA) funcționează în condiții reale!**


