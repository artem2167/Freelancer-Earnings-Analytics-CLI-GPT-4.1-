from openai import OpenAI
from config import OPENAI_API_KEY

# Инициализируем нового клиента OpenAI с ключом из конфигурации
client = OpenAI(api_key=OPENAI_API_KEY)

def ask_chatgpt(prompt: str) -> str:
    """
    Отправляет запрос в модель ChatGPT и возвращает текст ответа.
    """
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Ты аналитик данных."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800
    )
    # Ответ в новом API лежит в response.choices[0].message.content
    return response.choices[0].message.content.strip()
