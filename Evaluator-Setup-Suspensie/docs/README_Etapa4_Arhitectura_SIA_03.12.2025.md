# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Mircu Daniel  
**Link Repository GitHub**
**Data:** 04/12/2025  
---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Trebuie să livrați un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA). In acest stadiu modelul RN este doar definit și compilat (fără antrenare serioasă).**

### IMPORTANT - Ce înseamnă "schelet funcțional":

 **CE TREBUIE SĂ FUNCȚIONEZE:**
- Toate modulele pornesc fără erori
- Pipeline-ul complet rulează end-to-end (de la date → până la output UI)
- Modelul RN este definit și compilat (arhitectura există)
- Web Service/UI primește input și returnează output

 **CE NU E NECESAR ÎN ETAPA 4:**
- Model RN antrenat cu performanță bună
- Hiperparametri optimizați
- Acuratețe mare pe test set
- Web Service/UI cu funcționalități avansate

**Scopul anti-plagiat:** Nu puteți copia un notebook + model pre-antrenat de pe internet, pentru că modelul vostru este NEANTRENAT în această etapă. Demonstrați că înțelegeți arhitectura și că ați construit sistemul de la zero.

---

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software (max ½ pagină)
Completați in acest readme tabelul următor cu **minimum 2-3 rânduri** care leagă nevoia identificată în Etapa 1-2 cu modulele software pe care le construiți (metrici măsurabile obligatoriu):

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Optimizarea setup-ului mecanic: Dificultatea de a traduce rapid feedback-ul pilotului despre instabilitate (sub/supravirare) în modificări concrete ale suspensiei. |Analiza telemetriei suspensiei pentru clasificarea comportamentului (Sub vs. Supravirare) și generarea listei de ajustări în sub 30 secunde de la oprire. |Modul Dinamică Vehicul (RN Clasificare + Recommender System) |
|Validarea datelor brute: Zgomotul din senzorii de cursă suspensie face dificilă interpretarea manuală a histogramelor.|Preprocesarea și curățarea semnalului în timp real, eliminând outlierii cu o rată de succes de 80%, pentru a alimenta corect rețeaua neuronală.|Modul Ingestie & Procesare (Data Cleaning Pipeline)|

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

**Dataset complet original:**
```
Sistem de achizitie de date reale de la senzori
```

#### Tipuri de contribuții acceptate (exemple din inginerie):

Alegeți UNA sau MAI MULTE dintre variantele de mai jos și **demonstrați clar în repository**:

| **Tip contribuție** | **Exemple concrete din inginerie** | **Dovada minimă cerută** |
|---------------------|-------------------------------------|--------------------------|

| **Date achiziționate cu senzori proprii** | • 2000+ intrări de telemetrie sincronizată: deplasare suspensie (4 colțuri) + date IMU (G-Lateral/Longitudinal).


• Achiziție realizată folosind microcontroller (Arduino) conectat la potențiometre liniare.


• Etichetarea manuală a datelor pentru stările: Subvirare / Supravirare / Neutru. |Folder /data: Fișiere .csv |


#### Declarație obligatorie în README:

Scrieți clar în acest README (Secțiunea 2):

```markdown
### Contribuția originală la setul de date:

**Total observații finale:** 30000
**Observații originale:** 100%

**Tipul contribuției:**
[ ] Date generate prin simulare fizică  
[x] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  
[ ] Date sintetice prin metode avansate  

**Descriere detaliată:**
Sistem hardware de achiziție de date montat direct pe șasiul vehiculului. Sistemul este bazat pe un microcontroller **Arduino** care interoghează la o frecvență de **50Hz** următorii senzori fizici:
1.  **4x Potențiometre Liniare:** Montate pe amortizoare pentru a măsura cursa suspensiei în timp real.
2.  **1x IMU (Accelerometru/Giroscop):** Pentru a corela mișcarea suspensiei cu forțele G laterale și longitudinale.

Datele au fost colectate în sesiuni reale de testare pe circuit, pe un circuit specific competitiei pentru a induce stări de subvirare și supravirare. Setul de date este relevant deoarece conține **zgomotul real al senzorilor** și vibrațiile mecanice ale șasiului, provocări pe care simulările nu le reproduc perfect.

**Locația codului:** `src/app.py`
**Locația datelor:** `data/`

**Dovezi:**
- Setup experimental: `docs/setup.jpg`
```

---

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

**Cerințe:**
- **Minimum 4-6 stări clare** cu tranziții între ele
- **Formate acceptate:** PNG/SVG, pptx, draw.io 
- **Locație:** `docs/state_machine.*` (orice extensie)
- **Legendă obligatorie:** 1-2 paragrafe în acest README: "De ce ați ales acest State Machine pentru nevoia voastră?"

**Stări tipice pentru un SIA:**
```
IDLE → UPLOAD_CSV → CHECK → PROCESSING → AGGREGATE → DISPLAY
↑         ↑____ERROR__|                                 |
|_______________________________________________________|
```

**Legendă obligatorie (scrieți în README):**
```markdown
### Justificarea State Machine-ului ales:

Am ales arhitectura de tip Batch Processing, deoarece proiectul nostru se bazează pe date stocate local pe un Card SD. Această abordare este standard în motorsportul semi-profesionist, unde telemetria live este instabilă sau prea costisitoare. Analiza se face la boxe, după ce mașina revine de pe pistă.

Stările principale:

UPLOAD & VALIDATE: Verificarea critică a structurii fișierului CSV. Dacă senzorii au fost deconectați (vibrații) și lipsesc coloane, sistemul trebuie să refuze procesarea (ERROR_FORMAT) pentru a nu oferi recomandări greșite.

PROCESSING_PIPELINE: Include filtrarea zgomotului electric și segmentarea turei în ferestre glisante (Sliding Windows) pentru a captura dinamica mașinii în viraje.

RN_INFERENCE (Batch): Rețeaua neuronală analizează secvențial toate ferestrele extrase, clasificând comportamentul mașinii.

REPORTING: Agregarea rezultatelor (ex: "70% Supravirare") și afișarea dashboard-ului care ghidează inginerul în modificarea suspensiei.
...

### Detalierea Tranzițiilor și a Stărilor de Eroare

**Tranzițiile critice sunt:**
- **[CHECK_COLUMNS] → [PREPROCESS]:** Se execută automat imediat ce validatorul confirmă prezența tuturor coloanelor obligatorii (ex: `susp_fl`, `acc_y`) în header-ul fișierului CSV încărcat.
- **[SEGMENTATION] → [RN_INFERENCE]:** Se declanșează după ce întregul fișier a fost parcurs și "tăiat" în ferestre glisante (ex: 200 samples cu overlap 50%),
- **[CHECK_COLUMNS] → [ERROR_FORMAT]:** Se activează dacă fișierul este gol, corupt sau dacă lipsesc datele de la un senzor critic

**Starea ERROR este esențială pentru că:**
În motorsport, mediul de achiziție este extrem de ostil (vibrații mecanice severe, temperaturi ridicate, șocuri). Este frecvent ca un potențiometru să se deconecteze intermitent sau ca scrierea pe cardul SD să fie întreruptă brusc la oprirea motorului. Aplicația trebuie să gestioneze robust aceste fișiere incomplete și să informeze inginerul că tura respectivă nu poate fi analizată, evitând astfel recomandările de setup bazate pe date false.

**Bucla de feedback (Human-in-the-Loop):**
Deoarece sistemul este unul de asistență decizională (nu de control automat), bucla se închide prin **Inginerul de Cursă**. Rezultatul inferenței (ex: "70% Supravirare") duce la o acțiune fizică mecanică. Datele înregistrate în următoarea sesiune de pistă (Run 2) sunt reintroduse în sistem pentru a valida dacă modificarea a echilibrat mașina.
```

---

### 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)

Toate cele 3 module trebuie să **pornească și să ruleze fără erori** la predare. Nu trebuie să fie perfecte, dar trebuie să demonstreze că înțelegeți arhitectura.

| **Modul** | **Python (exemple tehnologii)** | **LabVIEW** | **Cerință minimă funcțională (la predare)** |
|-----------|----------------------------------|-------------|----------------------------------------------|
| **1. Data Logging / Acquisition** | `src/data_acquisition/` | LLB cu VI-uri de generare/achiziție | **MUST:** Produce CSV cu datele voastre (inclusiv cele 40% originale). Cod rulează fără erori și generează minimum 100 samples demonstrative. |
| **2. Neural Network Module** | `src/neural_network/model.py` sau folder dedicat | LLB cu VI-uri RN | **MUST:** Modelul RN definit, compilat, poate fi încărcat. **NOT required:** Model antrenat cu performanță bună (poate avea weights random/inițializați). |
| **3. Web Service / UI** | Streamlit | **MUST:** Primește input de la user și afișează un output. **NOT required:** UI frumos, funcționalități avansate. |

#### Detalii per modul:

#### **Modul 1: Data Logging / Acquisition**

**Funcționalități obligatorii:**
- [ ] Cod rulează fără erori: `python src/data_acquisition/generate.py` sau echivalent LabVIEW
- [ ] Generează CSV în format compatibil cu preprocesarea din Etapa 3
- [ ] Include minimum 40% date originale în dataset-ul final
- [ ] Documentație în cod: ce date generează, cu ce parametri

#### **Modul 2: Neural Network Module**

**Funcționalități obligatorii:**
- [ ] Arhitectură RN definită și compilată fără erori
- [ ] Model poate fi salvat și reîncărcat
- [ ] Include justificare pentru arhitectura aleasă (în docstring sau README)
- [ ] **NU trebuie antrenat** cu performanță bună (weights pot fi random)


#### **Modul 3: Web Service / UI**

**Funcționalități MINIME obligatorii:**
- [ ] Propunere Interfață ce primește input de la user (formular, file upload, sau API endpoint)
- [ ] Includeți un screenshot demonstrativ în `docs/screenshots/`

**Ce NU e necesar în Etapa 4:**
- UI frumos/profesionist cu grafică avansată
- Funcționalități multiple (istorice, comparații, statistici)
- Predicții corecte (modelul e neantrenat, e normal să fie incorect)
- Deployment în cloud sau server de producție

**Scop:** Prima demonstrație că pipeline-ul end-to-end funcționează: input user → preprocess → model → output.


## Structura Repository-ului la Finalul Etapei 4 (OBLIGATORIE)

**Verificare consistență cu Etapa 3:**

```
proiect-rn-[nume-prenume]/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── generated/  # Date originale
│   ├── train/
│   ├── validation/
│   └── test/
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/  # Din Etapa 3
│   ├── neural_network/
│   └── app/  # UI schelet
├── docs/
│   ├── state_machine.*           #(state_machine.png sau state_machine.pptx sau state_machine.drawio)
│   └── [alte dovezi]
├── models/  # Untrained model
├── config/
├── README.md
├── README_Etapa3.md              # (deja existent)
├── README_Etapa4_Arhitectura_SIA.md              # ← acest fișier completat (în rădăcină)
└── requirements.txt  # Sau .lvproj
```

**Diferențe față de Etapa 3:**
- Adăugat `data/generated/` pentru contribuția dvs originală
- Adăugat `src/data_acquisition/` - MODUL 1
- Adăugat `src/neural_network/` - MODUL 2
- Adăugat `src/app/` - MODUL 3
- Adăugat `models/` pentru model neantrenat
- Adăugat `docs/state_machine.png` - OBLIGATORIU
- Adăugat `docs/screenshots/` pentru demonstrație UI

---

## Checklist Final – Bifați Totul Înainte de Predare

### Documentație și Structură
- [x] Tabelul Nevoie → Soluție → Modul complet (minimum 2 rânduri cu exemple concrete completate in README_Etapa4_Arhitectura_SIA.md)
- [x] Declarație contribuție 40% date originale completată în README_Etapa4_Arhitectura_SIA.md
- [x] Cod generare/achiziție date funcțional și documentat
- [ ] Dovezi contribuție originală: grafice + log + statistici în `docs/`
- [ ] Diagrama State Machine creată și salvată în `docs/state_machine.*`
- [ ] Legendă State Machine scrisă în README_Etapa4_Arhitectura_SIA.md (minimum 1-2 paragrafe cu justificare)
- [ ] Repository structurat conform modelului de mai sus (verificat consistență cu Etapa 3)

### Modul 1: Data Logging / Acquisition
- [x] Cod rulează fără erori (`python src/data_acquisition/...` sau echivalent LabVIEW)
- [x] Produce minimum 40% date originale din dataset-ul final
- [x] CSV generat în format compatibil cu preprocesarea din Etapa 3
- [ ] Documentație în `src/data_acquisition/README.md` cu:
  - [x] Metodă de generare/achiziție explicată
  - [x] Parametri folosiți (frecvență, durată, zgomot, etc.)
  - [x] Justificare relevanță date pentru problema voastră
- [x] Fișiere în `data/generated/` conform structurii

### Modul 2: Neural Network
- [ ] Arhitectură RN definită și documentată în cod (docstring detaliat) - versiunea inițială 
- [ ] README în `src/neural_network/` cu detalii arhitectură curentă

### Modul 3: Web Service / UI
- [ ] Propunere Interfață ce pornește fără erori (comanda de lansare testată)
- [ ] Screenshot demonstrativ în `docs/screenshots/ui_demo.png`
- [ ] README în `src/app/` cu instrucțiuni lansare (comenzi exacte)

---

**Predarea se face prin commit pe GitHub cu mesajul:**  
`"Etapa 4 completă - Arhitectură SIA funcțională"`

**Tag obligatoriu:**  
`git tag -a v0.4-architecture -m "Etapa 4 - Skeleton complet SIA"`


