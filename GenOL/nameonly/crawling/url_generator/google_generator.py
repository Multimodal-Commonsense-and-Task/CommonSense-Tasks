import os
import logging
import requests
import time
import re
from typing import List
from .base_generator import URLGenerator
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class GoogleURLGenerator(URLGenerator):
    def __init__(self, mode='headless', use_color=False, use_size=False, scroll_patience=50):
        super().__init__()
        self.options = Options()
        self.use_color = use_color
        self.use_size = use_size
        self.scroll_patience = scroll_patience
        self.url_list = None
        if mode == 'headless':
            self.options.add_argument("--headless")
            self.options.add_argument("--no-sandbox")
            self.options.add_argument("--disable-dev-shm-usage")
            # self.options.add_argument("--disable-gpu")
        
        self.colors_list = ['red', 'orange', 'yellow', 'green', 'teal', 'blue', 'purple', 'pink', 'white', 'gray', 'black', 'brown']        
        self.size_list = ['l', 'm', 'i']
        # self.driver = webdriver.Chrome(options=self.options)
        
    
    def scroll_down(self):
        elem = self.driver.find_element(By.TAG_NAME, "body")
        last_scroll = 0
        scroll_patience = 0

        while True:
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.2)
            scroll = self.driver.execute_script("return window.pageYOffset;")
            if scroll == last_scroll: # If the page is not scrolled
                try:
                    # Click on the 'Load more' button
                    print(f"Trying to click on the load more button")
                    load_more_button = self.driver.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[1]/div[2]/div[2]/input')
                    if load_more_button.is_displayed():
                        load_more_button.click()
                        print(f"Successfully clicked on the load more button")
                        scroll_patience = 0
                    else:
                        logger.info('Tried to click on the load more button but it is not displayed')
                        scroll_patience += 1
                except:
                    logger.info('No more load more button')
                    scroll_patience += 1
            else:
                scroll_patience = 0
                last_scroll = scroll
            
            if scroll_patience >= self.scroll_patience:
                break
    
    def crawl_color_size(self, query: str, color=None, size=None):
        self.driver = webdriver.Chrome(options=self.options)
        # self.driver.maximize_window()
        if color is None:
            if size is None:
                URL = f"https://www.google.com/search?q={query}&tbm=isch&hl=en"
            else:
                URL = f"https://www.google.com/search?q={query}&tbm=isch&tbs=isz:{size}&hl=en"
        else:
            if size is None:
                URL = f"https://www.google.com/search?q={query}&tbm=isch&tbs=ic:specific%2Cisc:{color}&hl=en"
            else:
                URL = f"https://www.google.com/search?q={query}&tbm=isch&tbs=ic:specific%2Cisc:{color}%2Cisz:{size}&hl=en"

        logger.info(f"URL: {URL}")
        self.driver.get(URL)
        time.sleep(1)

        print(f"Scrolling down")
        self.scroll_down()
        
        print(f"Scraping links")

        imgs = self.driver.find_elements(By.XPATH, '//div[@jsname="dTDiAc"]/div[@jsname="qQjpJ"]//img')

        links = []
        for idx, img in enumerate(imgs):
            try:
                src = img.get_attribute('src')
                links.append(src)
            except Exception as e:
                print('[Exception occurred while collecting links from google] {}'.format(e))
        
        self.url_list.extend(links)
        logger.info(f'Number of images until size {size}: {len(self.url_list)}')
        
        self.driver.quit()
    
    
    def generate_url(self, query: str, total_images: int = 10000, image_type=None, filename=None):
        if filename is None:
            filename = query
        self.url_list = []

        if self.use_size:
            for size in self.size_list:
                print(f"Size: {size}")
                if self.use_color:
                    for color in self.colors_list:
                        print(f"Color: {color}")
                        self.crawl_color_size(query.replace('_', ' '), color=color, size=size)
                        if len(set(self.url_list)) >= total_images:
                            logger.info(f"Break the loop because the number of images is enough: {len(set(self.url_list))}")
                            break
                else:
                    print(f"Color: None")
                    self.crawl_color_size(query.replace('_', ' '), color=None, size=size)
                    if len(set(self.url_list)) >= total_images:
                        logger.info(f"Break the loop because the number of images is enough: {len(set(self.url_list))}")
                        break
                
                if len(set(self.url_list)) >= total_images:
                    logger.info(f"Break the loop because the number of images is enough: {len(set(self.url_list))}")
                    break
        else:
            if self.use_color:
                for color in self.colors_list:
                    print(f"Color: {color}")
                    self.crawl_color_size(query.replace('_', ' '), color=color, size=None)
                    if len(set(self.url_list)) >= total_images:
                        logger.info(f"Break the loop because the number of images is enough: {len(set(self.url_list))}")
                        break
            else:
                self.crawl_color_size(query.replace('_', ' '), color=None, size=None)

        # Save the image URLs
        logger.info(f"Total number of images (after removing duplicated urls): {len(set(self.url_list))}")
        self.url_list = list(set(self.url_list))
    
        # Remove urls not starting with 'http'
        self.url_list = [url for url in self.url_list if url.startswith('http')]
        logger.info(f"Total number of images (after removing urls not starting with 'http'): {len(self.url_list)}")
        
        # Remove favicons
        favicon_pattern = re.compile(r'favicon', re.IGNORECASE)
        self.url_list = [url for url in self.url_list if not favicon_pattern.search(url)]
        
        self.driver.quit()
        return self.url_list
    
