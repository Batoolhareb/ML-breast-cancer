import os
import streamlit as st
import requests


# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon=":ribbon:",
    layout="wide"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .title {
            color: #e75480;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
        .stNumberInput, .stButton>button {
            width: 100%;
        }
        .stButton>button {
            background-color: #e75480;
            color: white;
            font-weight: bold;
            padding: 0.5rem;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #d43d6d;
        }
        .info-box {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)


col1, col2 = st.columns([1, 3])
with col1:

    pink_ribbon = """
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
        <path fill="#e75480" d="M50,10 C60,20 70,30 80,40 C90,50 95,60 95,70 C95,80 90,85 80,85 C70,85 60,80 50,70 C40,80 30,85 20,85 C10,85 5,80 5,70 C5,60 10,50 20,40 C30,30 40,20 50,10 Z"/>
    </svg>
    """
    st.markdown(f'<div style="width:100px; height:100px; margin:0 auto">{pink_ribbon}</div>', unsafe_allow_html=True)
    
with col2:
    st.markdown("<h1 class='title'>Breast Cancer Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        This tool helps assess the risk of breast cancer based on cell characteristics. 
        Please enter the patient's test results below.
    </div>
    """, unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    st.subheader("Cell Characteristics")
    Clump_Thickness = st.slider("Clump Thickness", 1, 10, 5, 
                               help="Thickness of clumps in the tissue sample")
    Cell_Size_Uniformity = st.slider("Cell Size Uniformity", 1, 10, 5,
                                   help="Uniformity of cell sizes")
    Cell_Shape_Uniformity = st.slider("Cell Shape Uniformity", 1, 10, 5,
                                     help="Uniformity of cell shapes")
    Marginal_Adhesion = st.slider("Marginal Adhesion", 1, 10, 5,
                                 help="Adhesion at margins of tissue")
    Single_Epi_Cell_Size = st.slider("Single Epithelial Cell Size", 1, 10, 5,
                                    help="Size of single epithelial cells")

with col2:
    st.subheader("Nuclear Features")
    Bare_Nuclei = st.slider("Bare Nuclei", 1, 10, 5,
                           help="Presence of bare nuclei")
    Bland_Chromatin = st.slider("Bland Chromatin", 1, 10, 5,
                               help="Texture of chromatin")
    Normal_Nucleoli = st.slider("Normal Nucleoli", 1, 10, 5,
                               help="Presence of normal nucleoli")
    Mitoses = st.slider("Mitoses", 1, 10, 5,
                       help="Rate of cell division")


st.markdown("<br>", unsafe_allow_html=True)


col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_button = st.button("Assess Risk", key="predict")


if predict_button:
    with st.spinner("Analyzing the data..."):
        payload = {
            "Clump_Thickness": Clump_Thickness,
            "Cell_Size_Uniformity": Cell_Size_Uniformity,
            "Cell_Shape_Uniformity": Cell_Shape_Uniformity,
            "Marginal_Adhesion": Marginal_Adhesion,
            "Single_Epi_Cell_Size": Single_Epi_Cell_Size,
            "Bare_Nuclei": Bare_Nuclei,
            "Bland_Chromatin": Bland_Chromatin,
            "Normal_Nucleoli": Normal_Nucleoli,
            "Mitoses": Mitoses
        }

        try:
            DEFAULT_BACKEND_URL = "http://0.0.0.0:8080"
            BACKEND_URL = os.environ.get("BACKEND_URL", DEFAULT_BACKEND_URL)

            response = requests.post(f"{BACKEND_URL}/predict/", json=payload)
            response.raise_for_status()
            
            result = response.json()
            diagnosis = result['diagnosis']
            
            st.markdown("---")
            if diagnosis == "Malignant":
                st.error(f"⚠️ **Assessment Result:** {diagnosis}")
                st.warning("This result suggests a higher risk of malignancy. Please consult with an oncologist for further evaluation and recommended next steps.")
            else:
                st.success(f"✅ **Assessment Result:** {diagnosis}")
                st.info("This result suggests benign characteristics. Regular follow-ups are still recommended as per standard screening guidelines.")
                
           
            st.markdown("""
            <div style='font-size: 0.8rem; color: #666; margin-top: 2rem;'>
                <strong>Disclaimer:</strong> This tool is for informational purposes only and should not replace 
                professional medical advice, diagnosis, or treatment.
            </div>
            """, unsafe_allow_html=True)
            
        except requests.exceptions.RequestException as e:
            st.error("⚠️ Service temporarily unavailable. Please try again later.")
            st.error(f"Error details: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    Breast Cancer Risk Assessment Tool v1.0 • For medical professionals
</div>
""", unsafe_allow_html=True)