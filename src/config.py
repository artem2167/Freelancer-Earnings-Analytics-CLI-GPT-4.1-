import os
from dotenv import load_dotenv

load_dotenv()  # берёт OPENAI_API_KEY из .env

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_PATH         = os.path.join(os.path.dirname(__file__), "../data/freelancer_earnings.csv")
