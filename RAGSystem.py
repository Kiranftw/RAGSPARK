import pandas
from pyspark.sql import SparkSession
import os
import requests
from typing import List, Dict, Any
from functools import wraps
import logging
from dotenv import load_dotenv, find_dotenv
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
import google.generativeai as genai

class RAGSystem(object):
    def __init__(self, modelname: str = "models/gemini-2.0-flash") -> None:
        load_dotenv(find_dotenv())
        self.session = SparkSession.builder.appName("RAGSPARK").getOrCreate()
        self.DIR = os.path.dirname(os.path.abspath(__file__))
        self.API = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.API)
        self.model: genai.GenerativeModel = None
        for model in genai.list_models():
            if model.name == modelname:
                self.model = genai.GenerativeModel(
                    model_name = modelname,
                    generation_config = {"response_mime_type": "text/plain"},
                    safety_settings={},
                    tools=None,
                    system_instruction=None,
                )
                break
        if self.model is None:
            raise ValueError(f"Model {modelname} not found in available models.")

    @staticmethod
    def ExceptionHandelling(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {e}")
                return None
        return wrapper
    
    def webscraping(self, url: str) -> str:
        #NOTE: I HAVE SCRAPED AROUND 1000 BOOKS FROM THE GUTENBERG PROJECT
        FOLDER = "DATAFILES"
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
        
        response = requests.get(url)
        if response.status_code == 200:
            filename = url.split("/")[-1]
            filepath = os.path.join(FOLDER, filename)
            with open(filepath, "w", encoding="utf-8") as file:
                file.write(response.text)
            logging.info(f"Web scraping completed. Data saved to {filepath}")
            return response.text
        elif response.status_code == 404:
            logging.error(f"URL not found: {url}")
            return None
        else:
            logging.error(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
            return None