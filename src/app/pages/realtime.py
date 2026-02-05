"""
Pagina pentru monitorizarea în timp real

Această pagină simulează un stream de telemetrie, rulează inferența pe ferestre
suprapuse și afișează probabilitățile live împreună cu un grafic G-G (lateral vs longitudinal).

Scopul modificărilor: adăugăm comentarii detaliate pentru a ușura înțelegerea
fluxului (generare date -> filtrare -> buffer -> inferență -> UI updates).
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time
from src.data_acquisition.synthetic_generator import generate_synthetic_telemetry
from src.preprocessing.signal_processor import butterworth_filter
from src.preprocessing.realtime_processor import RealTimeBuffer, process_single_window_rt


def show_realtime_page():
    """Afișează pagina de monitorizare în timp real.

    Ce face:
    - Verifică dacă modelul antrenat există în `st.session_state`
    - Afișează un toggle pentru pornirea simulării și un slider pentru viteza simulării
    - Construiește layout-ul de dashboard (metrici + grafice)

    Observații:
    - `st.toggle` returnează un boolean care decide dacă se rulează bucla de simulare
    - `sim_speed` este un delay (în secunde) între pași pentru a simula streaming-ul
    """
    st.header("🔴 Real-Time Monitor")

    # Asigurăm existența modelului antrenat înainte de a porni simularea
    if 'model' not in st.session_state:
        st.error("Model not trained! Please train a model first in 'Generate & Train'.")
        return

    # Controale UI: toggle pornire/opri și slider pentru delay între pași (sim_speed)
    col1, col2 = st.columns([1, 4])
    with col1:
        # `run_simulation` este boolean; dacă True, bucla principală rulează
        run_simulation = st.toggle('Start Live Stream', value=False)
    with col2:
        # `sim_speed` controlează întârzierea dintre update-uri pentru vizibilitate
        sim_speed = st.slider("Simulation Speed (Delay)", 0.01, 0.2, 0.05)

    # Construim layout-ul dashboard-ului și pornim simularea dacă utilizatorul a apăsat toggle
    _create_dashboard_layout(run_simulation, sim_speed)


def _create_dashboard_layout(run_simulation, sim_speed):
    """Creează layout-ul dashboard-ului (metrici și grafice)."""
    # Zone pentru metrici (status, confidence, lateral g)
    m1, m2, m3 = st.columns(3)
    metric_status = m1.empty()
    metric_conf = m2.empty()
    metric_g = m3.empty()

    # Zone pentru grafice
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Live Predictions (Probabilities)**")
        chart_probs = st.empty()
    with c2:
        st.markdown("**G-G Diagram (Lateral vs Long)**")
        chart_gg = st.empty()

    # Dacă utilizatorul a pornit simularea, rulează bucla principală
    if run_simulation:
        _run_simulation(
            metric_status, metric_conf, metric_g,
            chart_probs, chart_gg,
            sim_speed
        )


def _run_simulation(metric_status, metric_conf, metric_g, 
                    chart_probs, chart_gg, sim_speed):
    """Rulează simularea: generează date, filtrează, normaliză și iterează pe eșantioane.

    Pași:
    1. Generare telemetrie sintetică (aici: `oversteer` dar se poate schimba)
    2. Aplicare filtru Butterworth pe fiecare canal
    3. Normalizare (zero mean, unit std) per canal
    4. Iterare sample cu sample, umplerea buffer-ului realtime și inferență când e gata
    """
    # Generează date de test (aici: oversteer) - poți schimba la alt comportament
    full_data = generate_synthetic_telemetry(duration_sec=60, behavior='oversteer')

    # Ordinea coloanelor este importantă: fiecare index are un sens fix
    sensor_cols = ['susp_fl', 'susp_fr', 'susp_rl', 'susp_rr', 
                   'acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z']

    # Filtrare (Butterworth) pe fiecare canal pentru a reduce zgomotul
    raw_values = np.zeros((len(full_data), len(sensor_cols)))
    for i, col in enumerate(sensor_cols):
         raw_values[:, i] = butterworth_filter(full_data[col].values)

    # Normalizare per canal (pentru a avea date comparabile între canale)
    mean = raw_values.mean(axis=0)
    std = raw_values.std(axis=0)
    norm_values = (raw_values - mean) / (std + 1e-8)

    # Buffer pentru ferestre în timp real (window_size determinat în RealTimeBuffer)
    rt_buffer = RealTimeBuffer()

    # Istoric pentru grafice (lungime controlată). Cheile:
    # - 'probs_under'/'probs_over': history of model probabilities for each class
    # - 'lat'/'long': istoric pentru accelerațiile lat/long folosite în G-G plot
    history = {
        'probs_under': [],
        'probs_over': [],
        'lat': [],
        'long': []
    }

    # Loop principal: iterăm prin fiecare eșantion (simulare streaming)
    for i in range(len(norm_values)):
        # Adăugăm sample normalizat în buffer
        sample_norm = norm_values[i]
        rt_buffer.add_sample(sample_norm)

        # Date brute folosite pentru calculul G-force afișat
        # Atenție: index-urile se bazează pe `sensor_cols` de mai sus
        sample_raw_acc_x = raw_values[i, 4]  # acc_x
        sample_raw_acc_y = raw_values[i, 5]  # acc_y

        # Stări implicite (înainte ca buffer-ul să fie plin)
        current_status = "Buffering..."
        conf = 0.0

        # Dacă buffer-ul e pregătit, facem inferență
        if rt_buffer.is_ready():
            # Get entire window and facem inferență (folosind același extractor de features)
            pred_idx, probs = process_single_window_rt(
                rt_buffer.get_window(), 
                st.session_state.model
            )

            # pred_idx: 0 = understeer, 1 = oversteer (convenție folosită în evaluator)
            is_understeer = (pred_idx == 0)
            current_status = "UNDERSTEER" if is_understeer else "OVERSTEER"
            conf = probs[pred_idx]

            # Salvăm probabilitățile pentru afișare în graficul de probabilități
            history['probs_under'].append(probs[0])
            history['probs_over'].append(probs[1])
        else:
            # Buffer-ul nu e plin: adăugăm zero pentru păstrarea aliniamentului datelor
            history['probs_under'].append(0)
            history['probs_over'].append(0)

        # Update istorice G-force (folosite în G-G scatter plot)
        history['lat'].append(sample_raw_acc_y)
        history['long'].append(sample_raw_acc_x)

        # Păstrăm doar ultimele 100 puncte pentru performanță UI
        if len(history['lat']) > 100:
            for key in history:
                history[key].pop(0)

        # Actualizăm widget-urile UI și graficele
        _update_ui(
            metric_status, metric_conf, metric_g,
            chart_probs, chart_gg,
            current_status, conf, sample_raw_acc_y,
            history
        )

        # Întârziere controlată pentru vizibilitate (sim_speed)
        time.sleep(sim_speed)


def _update_ui(metric_status, metric_conf, metric_g, chart_probs, chart_gg,
               current_status, conf, acc_y, history):
    """Actualizează elementele UI (metrici și grafice).

    - Actualizează trei metrici: Status (UNDER/OVER/Buffering), Confidence și Lateral G
    - Re-desenăm graficul de probabilități și G-G scatter la fiecare pas
    """
    # Metrici: schimbăm culoarea în funcție de status pentru evidențiere vizuală
    color = "normal"
    if current_status == "UNDERSTEER": 
        color = "off"
    if current_status == "OVERSTEER": 
        color = "inverse"

    # Actualizăm valorile afișate în widget-urile metric
    metric_status.metric("Status", current_status, delta_color=color)
    metric_conf.metric("Confidence", f"{conf:.1%}")
    metric_g.metric("Lateral G", f"{acc_y:.2f}")

    # Grafic probabilități (time series) - două serii: under și over
    fig_probs = go.Figure()
    fig_probs.add_trace(go.Scatter(
        y=history['probs_under'], 
        name='Understeer', 
        line=dict(color='#f69521')
    ))
    fig_probs.add_trace(go.Scatter(
        y=history['probs_over'], 
        name='Oversteer', 
        line=dict(color='#60935D')
    ))
    fig_probs.update_layout(
        height=250, 
        margin=dict(l=0,r=0,t=0,b=0), 
        yaxis_range=[0, 1.1]
    )
    chart_probs.plotly_chart(fig_probs, use_container_width=True)

    # Grafic G-G (scatter Lat vs Long) - punctele istorice + marcajul curent
    fig_gg = go.Figure()
    fig_gg.add_trace(go.Scatter(
        x=history['lat'], 
        y=history['long'], 
        mode='markers',
        marker=dict(color='gray', size=5, opacity=0.5)
    ))
    # Adăugăm un marker roșu pentru punctul curent (cross) pentru vizibilitate
    fig_gg.add_trace(go.Scatter(
        x=[acc_y], 
        y=[history['long'][-1]], 
        mode='markers', 
        marker=dict(color='red', size=15, symbol='cross')
    ))
    fig_gg.update_layout(
        xaxis_title="Lat G", 
        yaxis_title="Long G",
        height=250, 
        margin=dict(l=0,r=0,t=0,b=0),
        xaxis_range=[-3, 3], 
        yaxis_range=[-3, 3]
    )
    chart_gg.plotly_chart(fig_gg, use_container_width=True)