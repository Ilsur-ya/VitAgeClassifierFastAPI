import time
import pytest
from selenium import webdriver

def get_age(wait_time=0):
    "creates a webdriver and loads the homepage"
    driver = webdriver.Chrome()
    driver.get("http://127.0.0.1:8000/get-age-by-photo?url=https%3A%2F%2Fpixnio.com%2Ffree-images%2F2019%2F01%2F13%2F2019-01-13-09-42-34.jpg")
    time.sleep(wait_time) # wait for the GitHub api to load on the page
    return driver

@pytest.fixture
def age():
    age = get_age()
    yield age
    age.quit()



