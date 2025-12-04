import streamlit as st
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from pathlib import Path
import json
from datetime import datetime

# --- IMPORTURI NOI PENTRU REAL TIME ---
import time
from collections import deque

# ============================================================================
# 1. GENERARE DATE SINTETICE (NESCHIMBAT)
# ============================================================================

def generate_synthetic_telemetry(duration_sec=60, sampling_rate=100, behavior='neutral'):
    """Generează telemetrie sintetică pentru testare"""
    n_samples = int(duration_sec * sampling_rate)
    t = np.linspace(0, duration_sec, n_samples)
    
    # Suspensie (simulează bump-uri + roll)
    road = 0.02 * np.sin(2 * np.pi * 0.5 * t)  # Bump-uri
    road += 0.005 * np.random.randn(n_samples)  # Noise
    
    cornering = 0.03 * np.sin(2 * np.pi * 0.1 * t)  # Viraj
    
    susp_fl = road + cornering
    susp_fr = road - cornering
    susp_rl = road + cornering * 0.8
    susp_rr = road - cornering * 0.8
    
    # Ajustare pentru comportament
    if behavior == 'understeer':
        susp_fl += cornering * 0.5
        susp_fr -= cornering * 0.5
    elif behavior == 'oversteer':
        susp_rl += cornering * 0.5
        susp_rr -= cornering * 0.5
    
    # Accelerații
    acc_x = 0.3 * np.sin(2 * np.pi * 0.15 * t) + 0.1 * np.random.randn(n_samples)
    acc_y = 0.5 * np.sin(2 * np.pi * 0.1 * t) + 0.1 * np.random.randn(n_samples)
    
    if behavior == 'understeer':
        acc_y *= 0.8
    elif behavior == 'oversteer':
        acc_y *= 1.2
    
    acc_z = 9.81 + 0.5 * np.sin(2 * np.pi * 0.2 * t) + 0.2 * np.random.randn(n_samples)
    
    # Rotații
    rot_x = 0.1 * np.sin(2 * np.pi * 0.2 * t) + 0.02 * np.random.randn(n_samples)
    rot_y = 0.05 * np.sin(2 * np.pi * 0.15 * t) + 0.01 * np.random.randn(n_samples)
    rot_z = 0.15 * np.sin(2 * np.pi * 0.1 * t) + 0.03 * np.random.randn(n_samples)
    
    if behavior == 'understeer':
        rot_z *= 0.7
    elif behavior == 'oversteer':
        rot_z *= 1.3
    
    # DataFrame
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

# ============================================================================
# 2. PREPROCESARE (NESCHIMBAT)
# ============================================================================

def butterworth_filter(data, cutoff=10, fs=100, order=4):
    """Filtru Butterworth low-pass"""
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low')
    return signal.filtfilt(b, a, data)

def create_windows(data, window_size=200, overlap=0.5):
    """Creează ferestre cu overlap"""
    step = int(window_size * (1 - overlap))
    windows = []
    
    for i in range(0, len(data) - window_size, step):
        window = data[i:i+window_size]
        windows.append(window)
    
    return np.array(windows)

def extract_features(window):
    """Extrage features statistice dintr-o fereastră"""
    features = []
    
    # Pentru fiecare canal în fereastră
    if window.ndim == 1:
        window = window.reshape(-1, 1)
    
    for col in range(window.shape[1]):
        channel = window[:, col]
        
        features.extend([
            np.mean(channel),           # Mean
            np.std(channel),            # Std deviation
            np.min(channel),            # Min
            np.max(channel),            # Max
            np.sqrt(np.mean(channel**2)), # RMS
            np.ptp(channel)             # Peak-to-peak
        ])
    
    return np.array(features)

def preprocess_telemetry(df, window_size=200, overlap=0.5):
    """Pipeline complet de preprocesare"""
    # Extrage coloane senzori
    sensor_cols = ['susp_fl', 'susp_fr', 'susp_rl', 'susp_rr', 
                   'acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z']
    
    # Filtrare
    filtered_data = np.zeros((len(df), len(sensor_cols)))
    for i, col in enumerate(sensor_cols):
        filtered_data[:, i] = butterworth_filter(df[col].values)
    
    # Normalizare
    mean = filtered_data.mean(axis=0)
    std = filtered_data.std(axis=0)
    normalized_data = (filtered_data - mean) / (std + 1e-8)
    
    # Windowing
    windows = create_windows(normalized_data, window_size, overlap)
    
    # Extragere features
    features_list = []
    for window in windows:
        features = extract_features(window)
        features_list.append(features)
    
    return np.array(features_list)

# ============================================================================
# 3. REȚEA NEURONALĂ (NESCHIMBAT)
# ============================================================================

class SuspensionClassifier(nn.Module):
    """Clasificator simplu MLP"""
    def __init__(self, input_size=60, hidden_sizes=[32, 16], output_size=2):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# ============================================================================
# 4. TRAINING (NESCHIMBAT)
# ============================================================================

def train_model(X_train, y_train, epochs=30, batch_size=32, lr=0.001):
    """Antrenează modelul"""
    # Pregătire date
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model
    input_size = X_train.shape[1]
    model = SuspensionClassifier(input_size=input_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
    
    return model, history

# ============================================================================
# 5. EVALUARE (NESCHIMBAT)
# ============================================================================

def evaluate_telemetry(model, features):
    model.eval()
    
    X = torch.FloatTensor(features)
    
    with torch.no_grad():
        outputs = model(X)
        probs = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probs, dim=1)
    
    # Analiză
    n_windows = len(predictions)
    n_understeer = (predictions == 0).sum().item()
    n_oversteer = (predictions == 1).sum().item()
    
    understeer_ratio = n_understeer / n_windows
    oversteer_ratio = n_oversteer / n_windows
    
    # Comportament dominant
    if understeer_ratio > oversteer_ratio:
        behavior = "understeer"
        confidence = understeer_ratio
    else:
        behavior = "oversteer"
        confidence = oversteer_ratio
    
    # Recomandări
    if confidence > 0.6:
        if behavior == "understeer":
            recommendations = {
                'message': "UNDERSTEER DETECTAT",
                'actions': [
                    "Crește camber negativ față (ex: -1.5° -> -2.0°)",
                    "Crește toe-out față (ex: 0° -> 0.1° per roată)",
                    "Reduce camber spate",
                    "Scade presiunea pneuri față"
                ]
            }
        else:
            recommendations = {
                'message': "OVERSTEER DETECTAT",
                'actions': [
                    "Crește camber negativ spate (ex: -1.0° -> -1.5°)",
                    "Reduce toe-out față",
                    "Reduce camber față",
                    "Scade presiunea pneuri spate"
                ]
            }
    else:
        recommendations = {
            'message': "CONFIDENCE SCĂZUTĂ",
            'actions': [
                "Colectează mai multe date",
                "Verifică calibrarea senzorilor"
            ]
        }
    
    return {
        'behavior': behavior,
        'confidence': confidence,
        'n_windows': n_windows,
        'understeer_ratio': understeer_ratio,
        'oversteer_ratio': oversteer_ratio,
        'predictions': predictions.numpy(),
        'probabilities': probs.numpy(),
        'recommendations': recommendations
    }

# ============================================================================
# 5.1 ENGINE REAL-TIME (NOU - ADAUGAT AICI)
# ============================================================================

class RealTimeBuffer:
    """Coada care păstrează ultimele N sample-uri pentru rețea"""
    def __init__(self, window_size=200, n_features=10):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.n_features = n_features
        # Umplem cu zerouri inițial ca să nu crape la start
        for _ in range(window_size):
            self.buffer.append(np.zeros(n_features))
            
    def add_sample(self, sample):
        """Adaugă un vector de senzori (1x10)"""
        self.buffer.append(sample)
        
    def get_window(self):
        return np.array(self.buffer)
        
    def is_ready(self):
        return len(self.buffer) == self.window_size

def process_single_window_rt(window, model):
    """Procesează o fereastră pentru inferență live"""
    # Folosim funcția existentă de extract_features
    features = extract_features(window)
    
    # Inference
    model.eval()
    with torch.no_grad():
        tensor_in = torch.FloatTensor(features).unsqueeze(0) # Batch size 1
        output = model(tensor_in)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
    return pred_idx, probs.numpy()[0]


# ============================================================================
# 6. WEB INTERFACE (STREAMLIT) - UI PAGINI
# ============================================================================

def show_home_page():
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Scopul aplicației?
        
        Analizează telemetria monopostului si oferă recomandări 
        pentru setup-ul suspensiei bazate pe retele neuronale.
        
        ### Features
        
        - **Generare Date Test**: Creează telemetrie sintetică
        - **Training RN**: Antrenează rețea neuronală pe comportamente de subvirare/supravirare
        - **Evaluare Rapidă**: Analizează telemetria (CSV static)
        - **Real-Time Monitor**: Simulare live stream
        - **Recomandări Smart**: Sugestii concrete pentru camber și toe
        
        ### Cum sa o folosesti?
        
        1. Mergi la **"Generate & Train"** și antrenează un model
        2. Apoi la **"Evaluate"** pentru fișiere CSV
        3. Sau la **"Real-Time Monitor"** pentru live feed
        
        """)
    
    with col2:
        st.info("""
        ### Necesare
        
        - Python 3.8+
        - 5 MB RAM
        - 30 secunde/analiză
        
        ### Acurtete: >60% pentru recomandări
        
        ### Rezultate
        
        - **30 secunde** vs 30 minute manual
        - Decizii reproducibile
        - Bazate pe telemetrie reala
        """)
        
        
def show_train_page():
    st.header("Generate Data & Train Model")
    
    tab1, tab2 = st.tabs(["Generate Data", "Train Model"])
    
    with tab1:
        st.subheader("Generate Synthetic Telemetry")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            duration = st.slider("Duration (seconds)", 30, 120, 60)
        with col2:
            sampling_rate = st.slider("Sampling Rate (Hz)", 50, 200, 100)
        with col3:
            n_samples = st.number_input("Number of samples", 100, 1000, 200)
        
        if st.button("Generate Training Data", type="primary"):
            with st.spinner("Generating data..."):
                # Generate balanced dataset
                X_list = []
                y_list = []
                
                progress_bar = st.progress(0)
                
                for i in range(n_samples):
                    # Alternează între understeer și oversteer
                    behavior = 'understeer' if i % 2 == 0 else 'oversteer'
                    label = 0 if behavior == 'understeer' else 1
                    
                    # Generează telemetrie
                    df = generate_synthetic_telemetry(
                        duration_sec=duration, 
                        sampling_rate=sampling_rate,
                        behavior=behavior
                    )
                    
                    # Preprocesare
                    features = preprocess_telemetry(df)
                    
                    # Adaugă toate ferestrele
                    X_list.append(features)
                    y_list.extend([label] * len(features))
                    
                    progress_bar.progress((i + 1) / n_samples)
                
                # Concatenate
                X_train = np.vstack(X_list)
                y_train = np.array(y_list)
                
                # Salvează în session state
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                
                st.success(f"Generated {len(X_train)} training samples!")
                
                # Show stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Samples", len(X_train))
                with col2:
                    st.metric("Features", X_train.shape[1])
                with col3:
                    st.metric("Classes", len(np.unique(y_train)))
    
    with tab2:
        st.subheader("Train Neural Network")
        
        if 'X_train' not in st.session_state:
            st.warning("Generate training data first!")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            epochs = st.slider("Epochs", 10, 100, 30)
            batch_size = st.slider("Batch Size", 16, 128, 32)
        
        with col2:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.0001, 0.0005, 0.001, 0.005],
                value=0.001
            )
        
        if st.button("Start Training", type="primary"):
            with st.spinner(f"Training for {epochs} epochs..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Train
                model, history = train_model(
                    st.session_state.X_train,
                    st.session_state.y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=learning_rate
                )
                
                progress_bar.progress(100)
                
                # Save model
                st.session_state.model = model
                st.session_state.history = history
                
                st.success("Training Complete!")
                
                
                # Plot training history
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history['train_loss'],
                    name='Train Loss',
                    mode='lines'
                ))
                fig.add_trace(go.Scatter(
                    y=history['val_loss'],
                    name='Validation Loss',
                    mode='lines'
                ))
                fig.add_trace(go.Scatter(
                    y=history['val_acc'],
                    name='Validation Accuracy',
                    mode='lines',
                    yaxis='y2'
                ))
                
                fig.update_layout(
                    title="Training History",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    yaxis2=dict(title="Accuracy", overlaying='y', side='right'),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Final metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
                with col2:
                    st.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
                with col3:
                    st.metric("Final Accuracy", f"{history['val_acc'][-1]*100:.1f}%")

def show_evaluate_page():
    st.header("Evaluate Telemetry")
    
    if 'model' not in st.session_state:
        st.error("No trained model found! Please train a model first.")
        if st.button("Go to Training"):
            st.session_state.page = "Generate & Train"
            st.rerun()
        return
    
    tab1, tab2 = st.tabs(["Load Data", "Results"])
    
    with tab1:
        st.subheader("Load Telemetry Data")
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Evaluate", type="primary"):
                with st.spinner("Processing..."):
                    features = preprocess_telemetry(df)
                    results = evaluate_telemetry(st.session_state.model, features)
                    
                    st.session_state.evaluation_results = results
                    st.session_state.test_df = df
                    
                    st.success("Evaluation Complete!")
                    st.rerun()
    
    with tab2:
        if 'evaluation_results' not in st.session_state:
            st.info("Evaluate data first to see results")
            return
        
        results = st.session_state.evaluation_results
        
        # Summary
        st.subheader("Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Detected Behavior", results['behavior'].upper())
        with col2:
            st.metric("Confidence", f"{results['confidence']*100:.1f}%")
        with col3:
            st.metric("Windows", results['n_windows'])
        with col4:
            reliability = "HIGH" if results['confidence'] > 0.6 else "LOW"
            st.metric("Reliability", reliability)
        
        # Recommendations
        st.markdown("---")
        st.subheader("Recommendations")
        
        rec = results['recommendations']
        
        if results['confidence'] > 0.6:
            st.success(rec['message'])
        else:
            st.warning(rec['message'])
        
        st.markdown("#### Actions:")
        for i, action in enumerate(rec['actions'], 1):
            st.markdown(f"{i}. {action}")
        
        # Visualizations
        st.markdown("---")
        st.subheader("Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=['Understeer', 'Oversteer'],
                values=[results['understeer_ratio'], results['oversteer_ratio']],
                hole=0.4,
                marker=dict(colors=['#f69521', '#60935D'])
            )])
            fig.update_layout(title="Behavior Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence timeline
            windows = list(range(results['n_windows']))
            confidence = [max(p) for p in results['probabilities']]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=windows,
                y=confidence,
                mode='lines+markers',
                name='Confidence'
            ))
            fig.add_hline(y=0.6, line_dash="dash", line_color="red")
            fig.update_layout(
                title="Confidence Timeline",
                xaxis_title="Window",
                yaxis_title="Confidence",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Download report
        if st.button("Download Report", use_container_width=True):
            report = {
                'timestamp': datetime.now().isoformat(),
                'behavior': results['behavior'],
                'confidence': float(results['confidence']),
                'recommendations': rec
            }
            
            json_str = json.dumps(report, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def show_realtime_page():
    st.header("🏁 Real-Time Monitor")
    
    if 'model' not in st.session_state:
        st.error("Model not trained! Please train a model first in 'Generate & Train'.")
        return

    # Controale
    col1, col2 = st.columns([1, 4])
    with col1:
        run_simulation = st.toggle('Start Live Stream', value=False)
    with col2:
        sim_speed = st.slider("Simulation Speed (Delay)", 0.01, 0.2, 0.05)

    # Layout Dashboard
    m1, m2, m3 = st.columns(3)
    metric_status = m1.empty()
    metric_conf = m2.empty()
    metric_g = m3.empty()
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Live Predictions (Probabilities)**")
        chart_probs = st.empty()
    with c2:
        st.markdown("**G-G Diagram (Lateral vs Long)**")
        chart_gg = st.empty()

    # Logica de simulare
    if run_simulation:
        # Generăm un set lung de date de test (ex: Oversteer dominant)
        full_data = generate_synthetic_telemetry(duration_sec=60, behavior='oversteer')
        sensor_cols = ['susp_fl', 'susp_fr', 'susp_rl', 'susp_rr', 
                       'acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z']
        
        # Filtrăm și normalizăm "on the fly" (simulat)
        raw_values = np.zeros((len(full_data), len(sensor_cols)))
        for i, col in enumerate(sensor_cols):
             raw_values[:, i] = butterworth_filter(full_data[col].values)
        
        # Normalizare rapidă (folosim statistici globale simulate pentru demo)
        mean = raw_values.mean(axis=0)
        std = raw_values.std(axis=0)
        norm_values = (raw_values - mean) / (std + 1e-8)
        
        # Buffer
        rt_buffer = RealTimeBuffer()
        
        # Istoric pentru grafice
        history_probs_under = []
        history_probs_over = []
        history_lat = []
        history_long = []
        
        # Loop
        for i in range(len(norm_values)):
            if not run_simulation: break
            
            # 1. New Sample (Sample normalizat pt rețea)
            sample_norm = norm_values[i]
            rt_buffer.add_sample(sample_norm)
            
            # Sample brut pentru afișare G-force (AccX=4, AccY=5 in lista de cols)
            sample_raw_acc_x = raw_values[i, 4]
            sample_raw_acc_y = raw_values[i, 5]
            
            # 2. Inference
            current_status = "Buffering..."
            conf = 0.0
            
            if rt_buffer.is_ready():
                pred_idx, probs = process_single_window_rt(rt_buffer.get_window(), st.session_state.model)
                
                is_understeer = (pred_idx == 0)
                current_status = "UNDERSTEER" if is_understeer else "OVERSTEER"
                conf = probs[pred_idx]
                
                history_probs_under.append(probs[0])
                history_probs_over.append(probs[1])
            else:
                # Cât timp se umple bufferul
                history_probs_under.append(0)
                history_probs_over.append(0)

            # Update history G-Force
            history_lat.append(sample_raw_acc_y)
            history_long.append(sample_raw_acc_x)
            
            # Păstrăm istoricul scurt pentru viteză
            if len(history_lat) > 100:
                history_probs_under.pop(0)
                history_probs_over.pop(0)
                history_lat.pop(0)
                history_long.pop(0)

            # 3. Update UI Elements
            color = "normal"
            if current_status == "UNDERSTEER": color = "off"
            if current_status == "OVERSTEER": color = "inverse"
            
            metric_status.metric("Status", current_status, delta_color=color)
            metric_conf.metric("Confidence", f"{conf:.1%}")
            metric_g.metric("Lateral G", f"{sample_raw_acc_y:.2f}")
            
            # Chart 1: Probs
            fig_probs = go.Figure()
            fig_probs.add_trace(go.Scatter(y=history_probs_under, name='Understeer', line=dict(color='#f69521')))
            fig_probs.add_trace(go.Scatter(y=history_probs_over, name='Oversteer', line=dict(color='#60935D')))
            fig_probs.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0), yaxis_range=[0, 1.1])
            chart_probs.plotly_chart(fig_probs, use_container_width=True)
            
            # Chart 2: G-G
            fig_gg = go.Figure()
            fig_gg.add_trace(go.Scatter(
                x=history_lat, y=history_long, mode='markers',
                marker=dict(color='gray', size=5, opacity=0.5)
            ))
            fig_gg.add_trace(go.Scatter(
                x=[sample_raw_acc_y], y=[sample_raw_acc_x], mode='markers', 
                marker=dict(color='red', size=15, symbol='cross')
            ))
            fig_gg.update_layout(
                xaxis_title="Lat G", yaxis_title="Long G",
                height=250, margin=dict(l=0,r=0,t=0,b=0),
                xaxis_range=[-3, 3], yaxis_range=[-3, 3]
            )
            chart_gg.plotly_chart(fig_gg, use_container_width=True)
            
            time.sleep(sim_speed)


# ============================================================================
# 7. MAIN (UPDATE CU PAGINA NOUA)
# ============================================================================

def main():
    st.set_page_config(
        page_title="Suspension Setup Evaluator",
        page_icon="UPB Drive",
        layout="wide"
    )
    
    # CSS Custom
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(90deg, #f69521 0%, #d14e0d 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 1rem 0;
        }
        .stAlert {border-radius: 10px;}
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">Suspension Setup Evaluator</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Navigation")
        
        # AICI AM ADAUGAT OPTIUNEA NOUA
        page = st.radio(
            "Select Page",
            ["Home", "Generate & Train", "Evaluate", "Real-Time Monitor"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Status
        if 'model' in st.session_state:
            st.success("Model Trained")
        else:
            st.warning("No Model")
    
    # Pages
    if page == "Home":
        show_home_page()
    elif page == "Generate & Train":
        show_train_page()
    elif page == "Evaluate":
        show_evaluate_page()
    elif page == "Real-Time Monitor":
        show_realtime_page()

if __name__ == "__main__":
    main()