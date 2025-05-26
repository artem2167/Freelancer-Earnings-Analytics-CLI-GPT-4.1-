import os
import pandas as pd

from loader     import load_data, preprocess
from analyzer   import (
    compare_payment_methods,
    distribution_by_region,
    expert_below_100_projects,
    salary_vs_success_rate,
    salary_vs_rating,
    job_duration_correlation,
    salary_by_experience
)
from llm_client import ask_chatgpt
from prompts    import CLASSIFIER, TEMPLATES

# Словарь «action → функция»
VALID_ACTIONS = {
    "compare_payment":            compare_payment_methods,
    "by_region":                  distribution_by_region,
    "expert_under_100":           expert_below_100_projects,
    "salary_vs_success_rate":     salary_vs_success_rate,
    "salary_vs_rating":           salary_vs_rating,
    "job_duration_correlation":   job_duration_correlation,
    "salary_by_experience":       salary_by_experience
}

def classify_question_with_llm(question: str) -> str:
    """Классифицирует вопрос, возвращая имя функции-анализа."""
    prompt = CLASSIFIER.format(question=question)
    action = ask_chatgpt(prompt).strip()
    return action if action in VALID_ACTIONS else ""


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Конвертирует DataFrame в Markdown без внешних зависимостей."""
    cols = df.columns.tolist()
    md  = "| " + " | ".join(cols) + " |\n"
    md += "| " + " | ".join(["---"] * len(cols)) + " |\n"
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        md += "| " + " | ".join(vals) + " |\n"
    return md


def build_prompt(action: str, question: str, df: pd.DataFrame) -> str:
    """Формирует prompt для LLM на основе результата анализа."""
    func = VALID_ACTIONS.get(action)
    if func is None:
        sample = dataframe_to_markdown(df.head(40))
        return (
            f"Вопрос: {question}\n\n"
            f"Данные (первые 40 строк):\n{sample}\n\n"
            "Попробуйте ответить, опираясь на эти данные."
        )

    result = func(df)

    if isinstance(result, dict):
        return TEMPLATES[action].format(question=question, data=result)

    if isinstance(result, float):
        return TEMPLATES[action].format(question=question, pct=result, corr=result)

    if isinstance(result, pd.DataFrame):
        try:
            table_md = result.to_markdown(index=False)
        except Exception:
            table_md = dataframe_to_markdown(result)
        return TEMPLATES[action].format(question=question, table=table_md)

    return f"Вопрос: {question}\n\nРезультат анализа: {result}"


def main():
    # Загрузка и предобработка данных
    try:
        df = preprocess(load_data())
    except FileNotFoundError as e:
        print(f"Не найден файл с данными: {e}")
        return

    # Основной цикл взаимодействия
    while True:
        question = input("Задайте вопрос по данным (или exit/quit для выхода): ") \
                   .strip() \
                   .strip('"') \
                   .strip("'")
        # Выход по команде
        if question.lower() in {"exit", "quit"}:
            break
        # При пустом вводе — повтор
        if not question:
            print("Пустой ввод — попробуйте ещё раз или введите 'exit' для выхода.")
            continue

        # 1) Классификация вопроса
        action = classify_question_with_llm(question)

        # 2) Подготовка prompt на основе агрегации
        prompt = build_prompt(action, question, df)

        # 3) Запрос LLM и вывод ответа
        answer = ask_chatgpt(prompt)
        print("\n--- Ответ системы ---\n")
        print(answer)
        print("\n")

    print("Работа завершена. До новых встреч!")


if __name__ == "__main__":
    main()