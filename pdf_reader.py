import pdfplumber
import pandas as pd
import logging
from typing import Optional, List, Tuple

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

def extract_tables_between_markers_with_logging(
    pdf_path: str,
    start_marker: str = "паспорт продукта",
    end_marker: str = "приложение номер 1"
) -> Optional[pd.DataFrame]:
    in_range = False
    collected_rows: List[List[str]] = []
    page_with_start = None
    start_y = None

    with pdfplumber.open(pdf_path) as pdf:
        logger.info("Открыт PDF '%s', страниц: %d", pdf_path, len(pdf.pages))
        for page in pdf.pages:
            page_num = page.page_number
            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning("Не удалось извлечь текст на странице %d: %s", page_num, e)
                text = ""
            low = text.lower()

            # Проверка начала диапазона
            if (not in_range) and (start_marker.lower() in low):
                in_range = True
                page_with_start = page_num
                logger.info("Найден start_marker '%s' на странице %d", start_marker, page_num)

                # найти y-координату маркера
                found = page.search(start_marker, case=False, regex=False)
                if found:
                    start_y = found[0]['top']
                    logger.info(" start_marker bbox top y = %f", start_y)
                else:
                    logger.info(" Не удалось определить bbox start_marker — не найден через page.search()")

            # Если мы внутри диапазона — ищем таблицы
            if in_range:
                try:
                    tables = page.find_tables()
                except Exception as e:
                    logger.warning("Ошибка find_tables на странице %d: %s", page_num, e)
                    tables = []

                if tables:
                    logger.info(" На странице %d найдено %d таблиц", page_num, len(tables))
                else:
                    logger.debug(" На странице %d — таблиц не найдено", page_num)

                for tbl_idx, table_obj in enumerate(tables, start=1):
                    bbox: Tuple[float, float, float, float] = table_obj.bbox  # x0, top, x1, bottom
                    top_y, bottom_y = bbox[1], bbox[3]
                    logger.debug("   Таблица #%d bbox top=%f bottom=%f", tbl_idx, top_y, bottom_y)

                    # Фильтрация: если на той же странице, что start — таблица ниже start_y
                    if page_num == page_with_start and (start_y is not None):
                        if bottom_y < start_y:
                            logger.info("   Таблица #%d пропущена: выше start_marker (bottom_y=%f < start_y=%f)",
                                        tbl_idx, bottom_y, start_y)
                            continue

                    # Проверка: если на этой странице встречается end_marker — проверка по bbox
                    if end_marker.lower() in low:
                        found_end = page.search(end_marker, case=False, regex=False)
                        if found_end:
                            end_y = found_end[0]['top']
                            logger.info(" Найден end_marker '%s' на странице %d, top y = %f",
                                        end_marker, page_num, end_y)
                            if top_y > end_y:
                                logger.info("   Таблица #%d пропущена: ниже end_marker (top_y=%f > end_y=%f)",
                                            tbl_idx, top_y, end_y)
                                continue
                            # после встреченного end_marker — считаем, что дальше таблицы не нужны
                            in_range = False
                            logger.info(" Диапазон завершён после end_marker — дальнейшие страницы игнорируются")

                    # если таблица прошла фильтры — извлекаем строки
                    try:
                        raw = table_obj.extract()
                    except Exception as e:
                        logger.warning("   Ошибка extract() для таблицы #%d на странице %d: %s", tbl_idx, page_num, e)
                        continue

                    logger.info("   Таблица #%d принята — добавляем %d строк", tbl_idx, len(raw))
                    for row in raw:
                        collected_rows.append(row)

            # Если end_marker на странице и диапазон был открыт — и мы обработали, можно выйти
            if (not in_range) and (page_with_start is not None):
                logger.info("Обработаны страницы до %d — выходим", page_num)
                break

    if collected_rows:
        df = pd.DataFrame(collected_rows)
        logger.info("Собрано всего строк: %d", len(df))
        return df
    else:
        logger.warning("Таблицы между маркерами не найдены")
        return None


if __name__ == "__main__":
    path = "your_document.pdf"
    df = extract_tables_between_markers_with_logging(path)
    if df is not None:
        print("Найдено строк:", len(df))
        print(df.head())
    else:
        print("Таблицы между маркерами не найдены.")
