import pdfplumber
import pandas as pd

def extract_table_between_markers(pdf_path: str,
                                  start_marker: str = "паспорт продукта",
                                  end_marker: str = "приложение номер 1") -> pd.DataFrame | None:
    """
    Ищет таблицы в PDF между строками start_marker и end_marker.
    Возвращает объединённый DataFrame со всей найденной таблицей,
    или None, если ничего не найдено.
    """
    tables = []
    collecting = False

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            # Проверка начала таблицы
            if (not collecting) and (start_marker.lower() in text.lower()):
                collecting = True
                # возможно — в этой же странице есть первая часть таблицы

            if collecting:
                # Попытка извлечь таблицы на странице
                page_tables = page.extract_tables()
                for table in page_tables:
                    df = pd.DataFrame(table[1:], columns=table[0])  # предположим, что первая строка — header
                    tables.append(df)

                # Проверка конца таблицы
                if end_marker.lower() in text.lower():
                    collecting = False
                    break  # если таблица закончилась — можно выйти (или продолжить, если нужно)
    if tables:
        # Соединяем все куски таблицы по вертикали
        full_table = pd.concat(tables, ignore_index=True)
        return full_table
    else:
        return None

if __name__ == "__main__":
    path = "your_document.pdf"
    df = extract_table_between_markers(path)
    if df is not None:
        print("Таблица извлечена, число строк:", len(df))
        print(df.head())
        df.to_csv("extracted_table.csv", index=False)
    else:
        print("Таблица не найдена между маркерами.")
