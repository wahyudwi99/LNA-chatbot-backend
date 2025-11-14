import io
import os
import re
import jwt
import json
import pytz
import psycopg2
import warnings
import pandas as pd
from google import genai
from datetime import datetime
from google.genai import types
from template_prompt import template_prompt_chatbot
from dotenv import load_dotenv
load_dotenv("./.env")


QUERY_DIR_PATH="./queries"

def postgresql_connect():
    """
    Connect to postgresql database
    """
    database_client = psycopg2.connect(
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USERNAME"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )

    return database_client


def insert_data_to_db(data: dict):
    """
    Insert processed data to database
    """
    connection = postgresql_connect()
    cursor = connection.cursor()

    with open(f"{QUERY_DIR_PATH}/insert_data.sql", "r") as openfile:
        query_file = openfile.read()

    values = (
        data["input_prompt"],
        data["result"],
        data["input_token"],
        data["output_token"],
        datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S")
    )
    cursor.execute(query_file, values)
    connection.commit()

    cursor.close()
    connection.close()


def construct_response(input_prompt: str):
    """
    Construct LLM response for every input prompt from users
    """
    # Define the google client service
    client = genai.Client(api_key=os.getenv("GOOGLE_CLOUD_API_KEY"))

    # Get response from llm
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            template_prompt_chatbot,
            input_prompt
        ]
    )

    result = response.text
    response_data = {
        "input_prompt": input_prompt,
        "result": result,
        "input_token": response.usage_metadata.prompt_token_count,
        "output_token": response.usage_metadata.candidates_token_count
    }

    # Insert response data to database
    insert_data_to_db(response_data)

    return result