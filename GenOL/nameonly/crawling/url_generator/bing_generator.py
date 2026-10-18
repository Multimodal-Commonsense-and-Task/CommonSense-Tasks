import os
import logging
import requests
import time
from typing import List
from .base_generator import URLGenerator
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class BingURLGenerator(URLGenerator):
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
        
        self.colors_list = ['RED', 'ORANGE', 'YELLOW', 'GREEN', 'TEAL', 'BLUE', 'PURPLE', 'PINK', 'BROWN', 'BLACK', 'GRAY', 'WHITE']       
        self.size_list = ['large', 'medium', 'small', 'wallpaper']
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
                    load_more_button = self.driver.find_element(By.CLASS_NAME, "btn_seemore")
                    if load_more_button.is_displayed():
                        load_more_button.click()
                        scroll_patience = 0
                    else:
                        logger.info('No more load more button')
                        scroll_patience += 1
                except:
                    logger.info('No more load more button')
                    scroll_patience += 1
                    print(f"Scroll patience: {scroll_patience}")
            else:
                scroll_patience = 0
                last_scroll = scroll
            
            if scroll_patience >= self.scroll_patience:
                break

    def crawl_specific_color(self, query: str, color: str, size: str):
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.maximize_window()
        URL = f"https://www.bing.com/images/search?q={query}&qft=+filterui%3aimagesize-{size}+filterui%3acolor2-FGcls_{color}&form=IRFLTR&first=1&setlang=en"
        logger.info(f"URL: {URL}")
        self.driver.get(URL)
        try:
            self.scroll_down()
        except:
            logger.info('Error in scrolling down')

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        image_info_list = soup.find_all('img', class_='mimg')

        for i in range(len(image_info_list)):
            if 'src' in image_info_list[i].attrs:
                self.url_list.append(image_info_list[i]['src'])
        
        logger.info(f'Number of images until color {color}: {len(self.url_list)}')

        self.driver.quit()
    
    def crawl_color_size(self, query: str, color=None, size=None):
        self.driver = webdriver.Chrome(options=self.options)
        # self.driver.maximize_window()
        if color is None:
            if size is None:
                URL = f"https://www.bing.com/images/search?q={query}&form=IRFLTR&first=1&setlang=en"
            else:
                URL = f"https://www.bing.com/images/search?q={query}&qft=+filterui%3aimagesize-{size}&form=IRFLTR&first=1&setlang=en"
        else:
            if size is None:
                URL = f"https://www.bing.com/images/search?q={query}&qft=+filterui%3acolor2-FGcls_{color}&form=IRFLTR&first=1&setlang=en"
            else:
                URL = f"https://www.bing.com/images/search?q={query}&qft=+filterui%3aimagesize-{size}+filterui%3acolor2-FGcls_{color}&form=IRFLTR&first=1&setlang=en"
        
        logger.info(f"URL: {URL}")
        self.driver.get(URL)
        time.sleep(1)

        print(f"Scrolling down")
        self.scroll_down()
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        image_info_list = soup.find_all('img', class_='mimg')

        for i in range(len(image_info_list)):
            if 'src' in image_info_list[i].attrs:
                self.url_list.append(image_info_list[i]['src'])
        
        logger.info(f'Number of images until color {color}: {len(self.url_list)}')

        self.driver.quit()
    def crawl_one_size(self, query: str, size: str):
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.maximize_window()
        URL = f"https://www.bing.com/images/search?q={query}&qft=+filterui%3aimagesize-{size}&form=IRFLTR&first=1&setlang=en"
        logger.info(f"URL: {URL}")

        self.driver.get(URL)
        try:
            self.scroll_down()
        except:
            logger.info('Error in scrolling down')
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        image_info_list = soup.find_all('img', class_='mimg')

        for i in range(len(image_info_list)):
            if 'src' in image_info_list[i].attrs:
                self.url_list.append(image_info_list[i]['src'])
        
        logger.info(f'Number of images until size {size}: {len(self.url_list)}')
        
        self.driver.quit()
    

    def generate_url(self, query: str, total_images: int = 10000, image_type=None, filename=None):
        if filename is None:
            filename = query
            
        self.url_list = []
        if self.use_size:
            for size in self.size_list:
                if self.use_color:
                    for color in self.colors_list:
                        self.crawl_color_size(query.replace('_', ' '), color, size=size)
                        if len(set(self.url_list)) >= total_images:
                            logger.info(f"Break the loop because the number of images is enough: {len(set(self.url_list))}")
                            break
                else:
                    self.crawl_color_size(query.replace('_', ' '), color=None, size=size)
                    if len(set(self.url_list)) >= total_images:
                        logger.info(f"Break the loop because the number of images is enough: {len(set(self.url_list))}")
                        break
        else:
            if self.use_color:
                for color in self.colors_list:
                    self.crawl_color_size(query.replace('_', ' '), color, size=None)
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
        
        self.driver.quit()
        return self.url_list
