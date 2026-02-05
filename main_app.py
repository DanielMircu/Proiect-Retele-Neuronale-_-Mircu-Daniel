"""
Aplicația principală Streamlit pentru Suspension Setup Evaluator
"""
import streamlit as st
from src.app.pages.home import show_home_page
from src.app.pages.train import show_train_page
from src.app.pages.evaluate import show_evaluate_page
from src.app.pages.realtime import show_realtime_page


def main():
    """Funcția principală a aplicației"""
    # Configurare pagină
    st.set_page_config(
        page_title="Suspension Setup Evaluator",
        page_icon="🏎️",
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
    
    # Header
    st.markdown(
        '<h1 class="main-header">🏎️ Suspension Setup Evaluator</h1>', 
        unsafe_allow_html=True
    )
    
    # Sidebar - Navigation
    with st.sidebar:
        st.markdown("### Navigation")
        
        page = st.radio(
            "Select Page",
            ["Home", "Generate & Train", "Evaluate", "Real-Time Monitor"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Status indicator
        if 'model' in st.session_state:
            st.success("✅ Model Trained")
            
        else:
            st.warning("⚠️ No Model")
            st.info("Train a model in 'Generate & Train'")
        
        st.markdown("---")
        
        # Info
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Version:** 1.0.0
            
            **Author:** Dani Mircu
            
            **Purpose:** Setup suspension analysis 
            using neural networks on telemetry data
            """)
    
    # Routing către pagini
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
