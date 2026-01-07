import os
import requests
import re
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Define constants
BASE_URL = "https://www.federalreserve.gov"
OUTPUT_DIR = "raw"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def setup_directory():
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Directory '{OUTPUT_DIR}' created.") 

def download_pdf(url, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)

    # Skip if already downloaded for future updates
    if os.path.exists(filepath):
        print(f"[SKIP] Already exists: {filename}")
        return

    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print(f"[OK] Downloaded: {filename}")
        time.sleep(0.5) # Trying to not get blocked
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")


def get_pdf_from_press_conf_page(page_url):
    try:
        # Fetch and parse the press conference page
        response = requests.get(page_url, headers=HEADERS)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = soup.find_all('a', href=True)
        for link in links:
            href = link['href']
            text = link.get_text().lower()
            
            # Avoid minutes since they have a 3-week delay
            if '.pdf' in href and ('FOMCpresconf' in href):
                return urljoin(page_url, href)
        return None
    except Exception as e:
        print(f"[ERROR] Error parsing page {page_url}: {e}")
        return None

def process_calendar_page(url, year_context=None):
    print(f"\nProcessing calendar: {url}")
    try:
        response = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Looking for links that say "Press Conference"
        links = soup.find_all('a', string=re.compile(r"Press Conference", re.I))
        
        for i, link in enumerate(links):
            href = link['href']
            full_url = urljoin(url, href)

            # Try to extract date from URL
            date_match = re.search(r'(\d{8})', href)
            
            if date_match:
                date_str = date_match.group(1)
            elif year_context:

                # If URL extraction fails, use the year context + an index to avoid duplicates
                # to avoid overwriting 'unknown_date'

                date_str = f"{year_context}_meeting_{i+1}"
            else:
                # Timestamp to avoid collision
                date_str = f"unknown_{int(time.time())}_{i}" 

            filename = f"{date_str}_PressConference.pdf"
            if href.endswith('.pdf'):
                download_pdf(full_url, filename)
            else:
                pdf_url = get_pdf_from_press_conf_page(full_url)
                if pdf_url:
                    download_pdf(pdf_url, filename)
                else:
                    print(f"[INFO] No PDF found inside {full_url}")

    except Exception as e:
        print(f"[ERROR] Error processing calendar {url}: {e}")

def main():
    setup_directory()
        
    start_year = 2011
    current_year = 2025
    
    for year in range(start_year, current_year):
        hist_url = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
        process_calendar_page(hist_url, year_context=year)


    # Current calendar
    current_calendar = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

    # Pass the current year as context in case of missing dates
    process_calendar_page(current_calendar, year_context=current_year)  

if __name__ == "__main__":
    main()