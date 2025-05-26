import pandas as pd

def compare_payment_methods(df: pd.DataFrame) -> dict:
    """
    Сравнение средней выручки фрилансеров по методам оплаты.
    Вернёт словарь {method: mean_earnings}.
    """
    res = df.groupby("payment_method")["earnings_usd"].mean().to_dict()
    return res

def distribution_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Распределение средней и медианной выручки по регионам.
    """
    agg = df.groupby("client_region")["earnings_usd"].agg(["mean", "median", "count"])
    return agg.reset_index()

def expert_below_100_projects(df: pd.DataFrame) -> float:
    """
    Процент «экспертов», у которых job_complete < 100.
    """
    experts = df[df["experience_level"].str.lower()=="expert"]
    pct = (experts["job_completed"] < 100).mean() * 100
    return pct


#Мои функции
def salary_vs_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Влияние процента успешных заказов на доход внутри каждого региона.
    Предполагаем, что в df есть колонки:
      - job_success_rate   (% успешных заказов)
      - job_completed (успешно завершённых)
      - earnings_usd
    """
    df = df.copy()

    # Разбиваем на 4 квантиля внутри каждого региона:
    df["rate_bin"] = df.groupby("client_region")["job_success_rate"] \
                       .transform(lambda x: pd.qcut(x, 4, labels=False, duplicates="drop"))
    agg = (
        df
        .groupby(["client_region", "rate_bin"])["earnings_usd"]
        .agg(["mean", "median", "count"])
        .rename_axis(index=["region","success_quartile"])
        .reset_index()
    )
    return agg

def salary_vs_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Влияние рейтинга (от 3.0 до 5.0) на доход фрилансеров внутри каждого региона.
    Рейтинги дробные, выбираем диапазоны 3.0–3.5, 3.5–4.0, 4.0–4.5, 4.5–5.0.
    """
    df = df.copy()
    # Фильтруем рейтинги >= 3.0
    df = df[df["client_rating"] >= 3.0]

    # Определяем бин-листы и метки
    bins = [3.0, 3.5, 4.0, 4.5, 5.0]
    labels = ["3.0–3.5", "3.5–4.0", "4.0–4.5", "4.5–5.0"]

    # Создаём категориальную переменную
    df["rating_bin"] = pd.cut(
        df["client_rating"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Группируем с observed=False, чтобы не было предупреждений
    agg = (
        df
        .groupby(["client_region", "rating_bin"], observed=False)["earnings_usd"]
        .agg(
            mean="mean",
            median="median",
            count="count"
        )
        .reset_index()
        .rename(columns={
            "client_region": "region",
            "rating_bin": "rating_range"
        })
    )
    return agg


def job_duration_correlation(df: pd.DataFrame) -> float:
    """
    Корреляция между длительностью задачи (job_duration_days)
    и доходом (earnings_usd) по всему датасету.
    """
    return df["job_duration_days"].corr(df["earnings_usd"])

def salary_by_experience(df: pd.DataFrame) -> pd.DataFrame:
    """
    Влияние числа заявленных навыков, уровня специалиста (experience_level) на доход внутри регионов.
    Предполагаем, что в df есть колонка `experience_level`.
    """
    df = df.copy()
    agg = (
        df
        .groupby(["client_region", "experience_level"])["earnings_usd"]
        .agg(["mean","median", "count"])
        .rename_axis(index=["region","experience_level"])
        .reset_index()
    )
    return agg