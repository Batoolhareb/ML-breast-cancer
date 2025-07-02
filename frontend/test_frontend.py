import requests
import time

# URL of the running Streamlit app
STREAMLIT_URL = "http://localhost:8501"

def wait_for_streamlit(timeout=60):
    """Wait for the Streamlit server to be available."""
    for _ in range(timeout // 3):
        try:
            response = requests.get(STREAMLIT_URL)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(3)
    return False

def test_streamlit_running():
    """Test if the Streamlit app starts and responds with status code 200."""
    assert wait_for_streamlit(), "Streamlit server did not start in time"

def test_homepage_content():
    """Test if expected content appears on the homepage."""
    response = requests.get(STREAMLIT_URL)
    assert response.status_code == 200
    assert "Breast Cancer" in response.text or "Prediction" in response.text
