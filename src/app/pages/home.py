"""
Pagina Home a aplicației
"""
import streamlit as st


def show_home_page():
    """Afișează pagina principală cu instrucțiuni rapide și feature list."""

   
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Scopul aplicației?

        Analizează telemetria monopostului și oferă recomandări 
        pentru setup-ul suspensiei bazate pe rețele neuronale.

        ### Features

        - **Generare Date Test**: Creează telemetrie sintetică
        - **Training RN**: Antrenează rețea neuronală pe comportamente de subvirare/supravirare
        - **Evaluare Rapidă**: Analizează telemetria (CSV static)
        - **Real-Time Monitor**: Simulare live stream
        - **Recomandări Smart**: Sugestii concrete pentru camber și toe

        ### Cum să o folosești?

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

        ### Acuratețe: >60% pentru recomandări

        ### Rezultate

        - **30 secunde** vs 30 minute manual
        - Decizii reproducibile
        - Bazate pe telemetrie reală
        """)
