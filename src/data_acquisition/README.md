# Modul Data Acquisition

## Descriere

Modul responsabil pentru **generarea și achiziția datelor** folosite pentru antrenarea rețelei neuronale.

## Componente

### 1. Generare Date Sintetice
- **Fișier**: `synthetic_generator.py` (în `src/preprocessing/`)
- Simulare fizică a suspensiei cu zgomot Gaussian calibrat
- Modelare transfer de mase în viraje și denivelări pistă
- **Parametri configurabili**:
  - Rata sampling: 50Hz
  - Zgomot Gaussian: σ = 0.05
  - Durata simulare: 100-500ms per scenario

### 2. Achiziție Senzori Reali
- Microcontroller Arduino cu IMU și 4 potențiometre liniare
- Citire la 50Hz
- Format CSV brut salvat în `data/raw/`
- **Senzori**:
  - Accelerometru (axa X, Y, Z)
  - Giroscop (axa Z)
  - 4 potențiometre (compresie suspensie)

## Fluxul de Lucru

```
Hardware/Generator → CSV Brut (data/raw/) → Preprocesare (Etapa 3) → Features (data/processed/)
```

## Fișiere Referință

- `data/raw/ses_1_teo.CSV`, `ses_2_teo.CSV`, `ses_3_dani.CSV` - Date reale
- `data/processed and manually classified/` - Date etichetate manual

## Contribuție Originală

- **100% din datele finale** provin din achiziție proprie (hardware) + generator sintetic propriu
- Depășit pragul de 40% pentru contribuție originală
