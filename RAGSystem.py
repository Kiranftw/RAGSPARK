import pandas
import numpy
from pyspark.sql import SparkSession
import os
import requests
from typing import List, Dict, Any
from functools import wraps
import logging
from dotenv import load_dotenv, find_dotenv
import google.generativeai as genai
from pyspark.sql.types import StructType, StructField, StringType
from delta import configure_spark_with_delta_pip
from pyspark.sql import Row #NOTE: this is used when you don't explicitly define schema
from sentence_transformers import SentenceTransformer
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate
from pyspark.ml.feature import Tokenizer, StopWordsRemover, RegexTokenizer
from pyspark.sql.functions import lower, regexp_replace, length, split, col, udf, trim
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class RAGSystem(object):
    def __init__(self, modelname: str = "models/gemini-2.0-flash") -> None:
        load_dotenv(find_dotenv())
        builder = SparkSession.builder \
            .appName("ParagraphToDelta") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        self.spark = configure_spark_with_delta_pip(builder).getOrCreate()
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
        self.DATAFOLDER = "DATAFILES"
        self.DELTAPATH = "delta_output_path"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.EmbeddingeModel = SentenceTransformer("all-MiniLM-L6-v2", device=device)

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
        # base_url = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
        # urls = [ base_url.format(id=i) for i in range(1101, 2101) ]
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
    
    @ExceptionHandelling
    def datacleaning(self) -> None:
        #TODO: removed 165 books greater than 1MB 
        count: int = 0
        for filename in os.listdir(self.DATAFOLDER):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(self.DATAFOLDER, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                if size > 1 * 1024 * 1024:
                    os.remove(filepath)
                    count += 1
                    print(f"Removed {filename}, size: {size} bytes")
        print(f"Removed {count} files")
    
    @ExceptionHandelling
    def get_existing_filenames(self):
        if os.path.exists(self.DELTAPATH):
            filenames = set()
            DataFrame = self.spark.read.format("delta").load(self.DELTAPATH)
            for row in DataFrame.distinct().collect():
                filenames.add(row["filename"])
            return filenames
        return set()
    
    @ExceptionHandelling
    def data_transfromation(self):
        #NOTE:
        schema = StructType([
            StructField("sno", StringType(), True),
            StructField("filename", StringType(), True),
            StructField("content", StringType(), True)
        ])
        existingFilenames = self.get_existing_filenames()
        sno = 0
        for filename in os.listdir(self.DATAFOLDER):
            if not filename.endswith(".txt") or filename in existingFilenames:
                continue
            filepath = os.path.join(self.DATAFOLDER, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    content = file.read()
                    lines = []
                    for line in content.splitlines():
                        line = line.strip()
                        if line:
                            lines.append(line)
                    cleaned = " ".join(lines)
                    row = [(str(sno), filename, cleaned)]
                    DataFrame = self.spark.createDataFrame(row, schema)
                    DataFrame.write.format("delta").mode("append").save(self.DELTAPATH)
                    logging.info(f"Uploaded: {filename}")
                    sno += 1
            except Exception as E:
                logging.error(f"Error processing {filename}: {E}")
    
    def main(self) -> None:
        self.data_preprocessing()

if __name__ == "__main__":
    rag = RAGSystem()
    rag.main()
