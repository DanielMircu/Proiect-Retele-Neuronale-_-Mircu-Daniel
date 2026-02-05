"""
Pagina pentru evaluarea telemetriei

Acest fișier definește pagina Streamlit folosită pentru:
- încărcarea unui fișier CSV cu telemetrie
- preprocesarea datelor folosind `preprocess_telemetry`
- evaluarea folosind modelul antrenat (în `st.session_state.model`)
- afișarea unui sumar, recomandări, vizualizări și posibilitatea de a descărca un raport JSON
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime

# Funcții de preprocesare și evaluare din pachetele interne
from src.preprocessing.signal_processor import preprocess_telemetry
from src.neural_network.evaluator import evaluate_telemetry


def show_evaluate_page():
    """Afișează pagina de evaluare.

    Verifică mai întâi dacă există un model antrenat stocat în `st.session_state`.
    Dacă nu există, oferă link pentru a merge la secțiunea de antrenare.
    Apoi afișează două taburi: încărcare date și rezultate.
    """
    st.header("Evaluate Telemetry")

    # Verificăm dacă un model antrenat este disponibil în sesiune
    if 'model' not in st.session_state:
        # Eroare clară pentru utilizator dacă nu există model
        st.error("No trained model found! Please train a model first.")
        if st.button("Go to Training"):
            # Navigare simplă către pagina de training (setare stare și rerun)
            st.session_state.page = "Generate & Train"
            st.rerun()
        return

    # Taburi pentru încărcare și pentru vizualizarea rezultatelor
    tab1, tab2 = st.tabs(["Load Data", "Results"])

    with tab1:
        _show_load_data_tab()

    with tab2:
        _show_results_tab()


def _show_load_data_tab():
    """Tab pentru încărcarea datelor.

    - Primește un fișier CSV încărcat de utilizator
    - Afișează primele rânduri pentru verificare
    - La apăsarea butonului Evaluate:
        * preprocesează datele
        * apelează evaluatorul modelului
        * salvează rezultatele în sesiune pentru tabul Results
    """
    st.subheader("Load Telemetry Data")

    # Widget pentru upload CSV
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded_file:
        # Citim fișierul CSV într-un DataFrame pandas
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)

        # Când utilizatorul apasă Evaluate, procesăm datele
        if st.button("Evaluate", type="primary"):
            with st.spinner("Processing..."):
                # 1) Preprocesare: transformă semnalele raw în feature-uri folosite de model
                features = preprocess_telemetry(df)

                # 2) Evaluare: obține predicțiile/metricile din model
                
                results = evaluate_telemetry(st.session_state.model, features)

                # 3) Salvăm rezultatele și datele de test în sesiune pentru afișare
                st.session_state.evaluation_results = results
                st.session_state.test_df = df

                st.success("Evaluation Complete!")
                # Reexecutăm aplicația astfel încât tabul Results să poată accesa rezultatele
                st.rerun()


def _show_results_tab():
    """Tab pentru afișarea rezultatelor.

    Verifică dacă evaluarea a fost rulată; dacă da, afișează sumar, recomandări,
    grafice de analiză și butonul de download raport.
    """
    if 'evaluation_results' not in st.session_state:
        st.info("Evaluate data first to see results")
        return

    results = st.session_state.evaluation_results

    # Afișăm un scurt sumar cu metricile cheie
    _show_summary(results)

    # Separare vizuală
    st.markdown("---")
    # Afișăm recomandările (mesaj și listă de acțiuni)
    _show_recommendations(results)

    st.markdown("---")
    # Grafice și timeline-uri pentru a inspecta detalii
    _show_visualizations(results)

    # Buton pentru descărcarea unui raport JSON conținând concluziile
    _show_download_button(results)


def _show_summary(results):
    """Afișează sumarul evaluării: comportament detectat, încredere, ferestre analizate, reliability."""
    st.subheader("Summary")

    # Patru coloane pentru metrics: comportament, confidence, windows, reliability
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # 'behavior' este string, ex: 'understeer' sau 'oversteer'
        st.metric("Detected Behavior", results['behavior'].upper())
    with col2:
        # 'confidence' este probabilitatea asocierii la comportament detectat
        st.metric("Confidence", f"{results['confidence']*100:.1f}%")
    with col3:
        # 'n_windows' = numărul de ferestre/time-windows analizate
        st.metric("Windows", results['n_windows'])
    with col4:
        # Reliability este derivată heuristically din confidence
        reliability = "HIGH" if results['confidence'] > 0.6 else "LOW"
        st.metric("Reliability", reliability)


def _show_recommendations(results):
    """Afișează recomandările generate pe baza rezultatelor evaluării."""
    st.subheader("Recommendations")

    # 'recommendations' ar trebui să fie un dict cu 'message' și 'actions'
    rec = results['recommendations']

    # Afisăm mesajul de recomandare ca success sau warning în funcție de confidence
    if results['confidence'] > 0.6:
        st.success(rec['message'])
    else:
        st.warning(rec['message'])

    # Lista de acțiuni concrete pe care operatorul le poate urma
    st.markdown("#### Actions:")
    for i, action in enumerate(rec['actions'], 1):
        st.markdown(f"{i}. {action}")


def _show_visualizations(results):
    """Afișează grafice pentru a înțelege distribuția comportamentelor și evoluția încrederii."""
    st.subheader("Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Diagrama tip 'donut' pentru ponderea fiecărui comportament
        fig = go.Figure(data=[go.Pie(
            labels=['Understeer', 'Oversteer'],
            # Valorile sunt ratio-uri calculate în evaluator (ex: % ferestre detectate ca oversteer)
            values=[results['understeer_ratio'], results['oversteer_ratio']],
            hole=0.4,
            marker=dict(colors=['#f69521', '#60935D'])
        )])
        fig.update_layout(title="Behavior Distribution", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Timeline cu valoarea de 'confidence' pentru fiecare fereastră analizată
        windows = list(range(results['n_windows']))
        # 'probabilities' ar trebui să fie o listă de tuple/vecințe cu probabilitățile per clasă
        # folosim max(p) pentru a obține încrederea pentru clasa aleasă per fereastră
        confidence = [max(p) for p in results['probabilities']]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=windows,
            y=confidence,
            mode='lines+markers',
            name='Confidence'
        ))
        # Linie orizontală pentru un prag vizual (ex: 0.6)
        fig.add_hline(y=0.6, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Confidence Timeline",
            xaxis_title="Window",
            yaxis_title="Confidence",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def _show_download_button(results):
    """Construiește și oferă un raport JSON descărcabil cu rezultatele evaluării."""
    if st.button("Download Report", use_container_width=True):
        # Construim un dict simplu cu timestamp și concluzii
        report = {
            'timestamp': datetime.now().isoformat(),
            'behavior': results['behavior'],
            'confidence': float(results['confidence']),
            'recommendations': results['recommendations']
        }

        # Serializăm JSON pentru download
        json_str = json.dumps(report, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
