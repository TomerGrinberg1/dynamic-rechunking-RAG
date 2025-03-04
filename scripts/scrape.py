import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time
import wikipediaapi

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

def extract_wikipedia_text(page_title):
    wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent='MyWikipediaBot/1.0')
    page = wiki_wiki.page(page_title)
    if page.exists():
        return page.text
    else:
        return f"The page '{page_title}' does not exist."

def get_evidence_page_source(url):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    time.sleep(3)  # wait for JavaScript to render content
    page_source = driver.page_source
    driver.quit()
    return page_source

def main():
    MAIN_PAGE_URL = "https://nlp.cs.washington.edu/triviaqa/sample.html"
    BASE_URL = "https://nlp.cs.washington.edu/triviaqa/"
    
    resp = requests.get(MAIN_PAGE_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html5lib")
    main_table = soup.find("table", attrs={"border": "1"})
    rows = main_table.find_all("tr")
    
    output = []
    # Skip the header row
    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) < 5:
            continue
        question = cells[1].get_text(strip=True)
        answer = cells[2].get_text(strip=True)
        
        # Process the Wikipedia evidence link in the 5th cell
        wiki_cell = cells[4]
        link_tag = wiki_cell.find("a")
        if not link_tag:
            continue
        relative_href = link_tag.get("href", "")
        evidence_url = urljoin(BASE_URL, relative_href)
        
        # Load the evidence page with Selenium
        ev_page_source = get_evidence_page_source(evidence_url)
        ev_soup = BeautifulSoup(ev_page_source, "html5lib")
        
        # Find all <a> tags that link to Wikipedia and have "- Wikipedia" in their text
        wiki_links = ev_soup.find_all("a", href=lambda x: x and "wikipedia.org" in x)
        wikipedia_pages = []
        for link in wiki_links:
            text = link.get_text(strip=True)
            if "- Wikipedia" in text:
                # Remove the " - Wikipedia" suffix
                title = text.replace(" - Wikipedia", "").strip()
                content = extract_wikipedia_text(title)
                wikipedia_pages.append({
                    "title": title,
                    "content": content
                })
        
        if wikipedia_pages:
            output.append({
                "question": question,
                "answer": answer,
                "wikipedia_pages": wikipedia_pages
            })
    
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
