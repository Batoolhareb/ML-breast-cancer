import time
import requests
from playwright.sync_api import sync_playwright

def wait_for_streamlit(url, timeout=60):  # Increased timeout for CI
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError(f"Streamlit app not responding at {url} after {timeout} seconds")

def test_breast_cancer_app():
    url = "http://localhost:8501"
    wait_for_streamlit(url)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Ensure headless for CI
        page = browser.new_page()
        page.goto(url)

        page.wait_for_selector("text=Breast Cancer Risk Assessment", timeout=10000)

        # Simulate interactions
        page.click("text=Clump Thickness")
        page.keyboard.press("ArrowRight")
        page.click("text=Cell Size")
        page.keyboard.press("ArrowRight")

        page.click("text=Assess Risk")

        page.wait_for_timeout(3000)

        content = page.content()
        print(content)  # Helpful for debugging in CI logs

        # Assert presence of result
        assert "Risk" in content or "Assessment" in content or "Result" in content

        browser.close()
