"""
Pagina pentru generarea datelor și antrenarea modelului
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.data_acquisition.synthetic_generator import generate_synthetic_telemetry
from src.preprocessing.signal_processor import preprocess_telemetry
from src.neural_network.trainer import train_model


def show_train_page():
    """Afișează pagina de training cu taburi pentru generare date și antrenare."""
    st.header("Generate Data & Train Model")

    tab1, tab2 = st.tabs(["Generate Data", "Train Model"])

    with tab1:
        _show_data_generation_tab()

    with tab2:
        _show_training_tab()


def _show_data_generation_tab():
    """Tab pentru generarea datelor sintentice de antrenare.

    - Se generează exemple balansate alternând comportamentele
    - Fiecare exemplu poate conține mai multe ferestre (features per window)
    - Label-urile se extind pentru fiecare fereastră generată
    """
    st.subheader("Generate Synthetic Telemetry")

    col1, col2, col3 = st.columns(3)

    with col1:
        duration = st.slider("Duration (seconds)", 30, 120, 60)
    with col2:
        sampling_rate = st.slider("Sampling Rate (Hz)", 50, 200, 50)
    with col3:
        n_samples = st.number_input("Number of samples", 100, 1000, 200)

    if st.button("Generate Training Data", type="primary"):
        with st.spinner("Generating data..."):
            # Generare dataset balansat
            X_list = []  # lista de matrici (features per exemplu)
            y_list = []  # lista de label-uri extinse per fereastră

            progress_bar = st.progress(0)

            for i in range(n_samples):
                # Alternăm între understeer și oversteer pentru balans
                behavior = 'understeer' if i % 2 == 0 else 'oversteer'
                label = 0 if behavior == 'understeer' else 1

                # Generăm telemetrie sintetică pentru un exemplu
                df = generate_synthetic_telemetry(
                    duration_sec=duration, 
                    sampling_rate=sampling_rate,
                    behavior=behavior
                )

                # Preprocesare -> obținem un array (n_windows x n_features)
                features = preprocess_telemetry(df)

                # Adăugăm toate ferestrele generate pentru exemplu
                X_list.append(features)

                # Extindem label-urile: câte ferestre are exemplul, atâtea label-uri
                y_list.extend([label] * len(features))

                progress_bar.progress((i + 1) / n_samples)

            # Concatenare: obținem (total_windows x n_features)
            X_train = np.vstack(X_list)
            y_train = np.array(y_list)

            # Salvăm în session state pentru a fi disponibile la antrenare
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train

            st.success(f"Generated {len(X_train)} training samples!")

            # Afișăm statistici utile
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(X_train))
            with col2:
                st.metric("Features", X_train.shape[1])
            with col3:
                st.metric("Classes", len(np.unique(y_train)))


def _show_training_tab():
    """Tab pentru antrenarea modelului.

    Permite configurarea hiperparametrilor și rulează `train_model`.
    La final salvează modelul antrenat în `st.session_state.model`.
    """
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

            # Train
            model, history = train_model(
                st.session_state.X_train,
                st.session_state.y_train,
                epochs=epochs,
                batch_size=batch_size,
                lr=learning_rate
            )

            progress_bar.progress(100)

            # Salvează model
            st.session_state.model = model
            st.session_state.history = history

            st.success("Training Complete!")

            # Plot training history
            _plot_training_history(history)

            # Metrici finale
            _show_final_metrics(history)

def _plot_training_history(history):
    """Afișează grafic cu istoricul antrenării"""
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


def _show_final_metrics(history):
    """Afișează metricile finale"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Final Train Loss", f"{history['train_loss'][-1]:.4f}")
    with col2:
        st.metric("Final Val Loss", f"{history['val_loss'][-1]:.4f}")
    with col3:
        st.metric("Final Accuracy", f"{history['val_acc'][-1]*100:.1f}%")
