import os
import logging
import requests
import time
from datetime import datetime, timedelta
from utils import date_to_unix
from typing import List
from .base_generator import URLGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class FlickrURLGenerator(URLGenerator):
    def __init__(self, api_key, save_dir, error_dir, max_page=100):
        super().__init__()
        self.API_KEY = api_key
        self.save_dir = save_dir
        self.error_dir = error_dir
        self.base_url = "https://www.flickr.com/services/rest/"
        self.max_page = max_page
    
    def get_datetime_range(self) -> str:
        date_ranges = []
        start_date0 = datetime(2006, 1, 1); end_date0 = datetime(2011, 12, 31)
        start_date1 = datetime(2012, 1, 1); end_date1 = datetime(2017, 12, 31)
        start_date2 = datetime(2018, 1, 1); end_date2 = datetime(2023, 12, 31)
        
        date_ranges.append((start_date0, end_date0))
        date_ranges.append((start_date1, end_date1))
        date_ranges.append((start_date2, end_date2))
        
        return date_ranges

    def fetch_image_urls(self, keyword: str, images_per_date_range: int, date_ranges: List[datetime.date]) -> List[str]:
        image_urls = []
        start_date = date_ranges[0]; end_date = date_ranges[1]
        logger.info(f"Start date: {start_date}, End date: {end_date}")
        
        # Convert to unix timestamp
        start_date_unix = date_to_unix(start_date); end_date_unix = date_to_unix(end_date)
        
        prev_num_images = 0
        num_images_not_increased = 0
        
        params = {
            "method": "flickr.photos.search", "api_key": self.API_KEY, "text": keyword.replace('_', ' '),
            "format": "json", "nojsoncallback": 1, "sort": "relevance", "page": 1, "per_page": 500,
            "min_upload_date": start_date_unix, "max_upload_date": end_date_unix
        }
        
        while len(set(image_urls)) < images_per_date_range:
            logger.info(f"Keyword: {keyword}, Date range: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}, Current number of images: {len(set(image_urls))}")
            logger.info(f"Page: {params['page']}")
            
            # Get data for the current page
            response = requests.get(self.base_url, params=params)
            logger.info(f"Rseponse URL: {response.url}")
            
            if response.status_code != 200:
                logger.error(f"Failed to fetch images for keyword: {keyword}, date range: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
                time.sleep(1)
                continue
        
            # Parse the date and get the total number of pages
            data = response.json()
            photos = data.get("photos", {}).get("photo", [])
            total_pages = data.get("photos", {}).get("pages", 0)
            if params["page"] >= total_pages + 3 or params["page"] >= self.max_page:
                logger.info(f"Break the loop, page: {params['page']}, total pages: {total_pages}")
                break
            
            # Get the image URLs
            for photo in photos:
                if len(set(image_urls)) >= images_per_date_range:
                    logger.info(f"Break the loop because the number of images is enough: {len(set(image_urls))}")
                    break
                url = f"https://live.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}_b.jpg"
                image_urls.append(url)
            
            params["page"] += 1
        
        return list(set(image_urls))

    def count(self, keyword: str):
        date_ranges = self.get_datetime_range()
        total_images = 0
        for date_range in date_ranges:
            total_images += self.check_total_images(keyword, date_range)
        
        return total_images
    
    def check_total_images(self, keyword: str, date_ranges: List[datetime.date]):
        start_date = date_ranges[0]; end_date = date_ranges[1]
        start_date_unix = date_to_unix(start_date); end_date_unix = date_to_unix(end_date)
        
        params = {
            "method": "flickr.photos.search", "api_key": self.API_KEY, "text": keyword,
            "format": "json", "nojsoncallback": 1, "sort": "relevance", "page": 1, "per_page": 500,
            "min_upload_date": start_date_unix, "max_upload_date": end_date_unix
        }
        response = requests.get(self.base_url, params=params)
        logger.info(f"Rseponse URL: {response.url}")
        data = response.json()
        total_images = data.get("photos", {}).get("total", 0)
        
        logger.info(f"Keyword: {keyword}, Date range: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}, Total images: {total_images}")
        
        return total_images
        
    
    def generate_url(self, keyword: str, total_images: int = 10000, images_per_date_range: int = 5000, filename=None, skip_range_check=True):
        date_ranges = self.get_datetime_range()
        if filename is None:
            filename = keyword
        # Count the estimated total number of images for each date range
        if not skip_range_check:
            estimated_total_images = 0
            for date_range in date_ranges:
                estimated_total_images += self.check_total_images(keyword, date_range)
            if estimated_total_images < total_images:
                logger.warning("="*80)
                logger.warning(f"Estimated total number of images: {estimated_total_images}, required total number of images: {total_images}")
                logger.warning("="*80)
        
        # Fetch image URLs
        image_urls = []
        for date_range in date_ranges:
            image_urls += self.fetch_image_urls(keyword, images_per_date_range, date_range)
            if len(set(image_urls)) >= total_images:
                logger.info(f"Break the loop because the number of images is enough: {len(set(image_urls))}")
                break
        
        # Save the image URLs
        logger.info(f"Total number of images (after removing duplicated urls): {len(set(image_urls))}")
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        image_urls = list(set(image_urls))
        with open(os.path.join(self.save_dir, f'{filename}.txt'), 'w') as f:
            for url in image_urls:
                f.write(url + '\n')
        
        # If the number of images is not enough, generate an error file
        if len(set(image_urls)) < total_images:
            logger.warning(f"Total number of images: {len(set(image_urls))}, required total number of images: {total_images}")
            if not os.path.exists(self.error_dir):
                os.makedirs(self.error_dir)
            with open(os.path.join(self.error_dir, f'{keyword}_error.txt'), 'w') as f:
                f.write(f"Total number of images: {len(set(image_urls))}, required total number of images: {total_images}")
