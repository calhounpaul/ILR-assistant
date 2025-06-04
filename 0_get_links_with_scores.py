import os,json,time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

url = 'https://gloss.dliflc.edu'
id_lang_selector_arabic = "LanguageSelectorInput_2"
id_searchbutton = 'lessonSearchBtn'
id_level_selector = 'LevelSelectorDDL'

def xpath_id(id_in):
    return '//*[@id="' + id_in + '"]'

driver = webdriver.Chrome()

saved_urls_cache = []
if not os.path.exists("cached_data"):
    os.mkdir("cached_data")

#save_links = driver.find_elements(By.CSS_SELECTOR, "a.saveIcon")
#hrefs = [link.get_attribute("href") for link in save_links]

# Load the page
driver.get(url)

# Wait until the element is present and clickable
lang_selector_arabic = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, xpath_id(id_lang_selector_arabic)))
)
lang_selector_arabic.click()

select_element = driver.find_element(By.ID, "ModalitySelectorDDL")
dropdown = Select(select_element)
dropdown.select_by_visible_text("Reading")

searchbutton = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable((By.XPATH, xpath_id(id_searchbutton)))
)
searchbutton.click()

all_data = []

page_num = 0
for n in range(500):
    page_num+=1
    this_data = []
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, '(//*[contains(@class, "titleLink")])[1]'))
    )
    time.sleep(3)

    trs = driver.find_elements(By.XPATH, '//tr[starts-with(@id, "lessonSearchResult_")]')

    for tr in trs:
        try:
            first_column = tr.find_element(By.XPATH, './td[1]/div[1]').text.strip()
            title_column = tr.find_element(By.XPATH, './td[2]').text.strip()

            save_link = tr.find_element(By.XPATH, './/a[contains(@class, "saveIcon")]').get_attribute("href")

            print(f"Level: {first_column} | Save Link: {save_link}")
            this_data.append({"level":first_column,"download_link":save_link,"title_data":title_column})

        except:
            print(tr)

    nextPageLink = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, xpath_id("nextPageLink")))
    )
    nextPageLink_class = nextPageLink.get_attribute("class")
    if nextPageLink_class == "pagerNextLinkDisabled":
        break
    nextPageLink.click()
    
    with open("cached_data/"+str(page_num).zfill(4)+".json","w") as f:
        json.dump(this_data,f,indent=4,default=str)