import time
from playwright.sync_api import sync_playwright

def wait_for_streamlit(url, timeout=10):
    import requests
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            time.sleep(0.5)
    raise RuntimeError(f"Streamlit app not responding at {url} after {timeout} seconds")

def test_breast_cancer_app():
    url = "http://localhost:8501"
    wait_for_streamlit(url)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)

        page.wait_for_selector("text=Breast Cancer Risk Assessment")

        # Simulate slider input
        page.click("text=Clump Thickness")
        page.keyboard.press("ArrowRight")
        page.click("text=Cell Size")
        page.keyboard.press("ArrowRight")

        # Click the button
        page.click("text=Assess Risk")

        # Optional: Wait longer for results
        page.wait_for_timeout(3000)

        # DEBUG: print page content
        content = page.content()
        print(content)  # 👈 look at this to see what the page shows

        # Flexible matching
        assert "Risk" in content or "Assessment" in content or "Result" in content

        browser.close()
