import pdfplumber
import pandas as pd
import logging
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def extract_tables_list_between_markers(
    pdf_path: str,
    start_marker: str = "паспорт продукта",
    end_marker: str = "приложение номер 1"
) -> List[pd.DataFrame]:
    """
    Возвращает список DataFrame — по одной таблице на каждый найденный block-table,
    которые лежат между start_marker и end_marker.

    Таблицы на той же странице что start_marker — только ниже него (по bbox),
    на той же странице что end_marker — только выше него.
    """
    in_range = False
    page_with_start = None
    start_y: Optional[float] = None
    result_tables: List[pd.DataFrame] = []

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

            # Найти start_marker на странице
            if (not in_range) and (start_marker.lower() in low):
                in_range = True
                page_with_start = page_num
                logger.info("Найден start_marker '%s' на странице %d", start_marker, page_num)
                found = page.search(start_marker, case=False, regex=False)
                if found:
                    start_y = found[0]['top']
                    logger.info(" start_marker top y = %f", start_y)
                else:
                    logger.info(" Не удалось получить bbox start_marker (page.search вернул пусто)")

            if in_range:
                try:
                    tables = page.find_tables()
                except Exception as e:
                    logger.warning("Ошибка find_tables на странице %d: %s", page_num, e)
                    tables = []

                if tables:
                    logger.info(" Страница %d: найдено %d таблиц", page_num, len(tables))

                for tbl_idx, table_obj in enumerate(tables, start=1):
                    bbox: Tuple[float, float, float, float] = table_obj.bbox  # (x0, top, x1, bottom)
                    top_y, bottom_y = bbox[1], bbox[3]
                    logger.debug("  Таблица #%d bbox top=%f bottom=%f", tbl_idx, top_y, bottom_y)

                    # Фильтровать таблицы на странице начала
                    if page_num == page_with_start and (start_y is not None):
                        if bottom_y < start_y:
                            logger.info("  Таблица #%d пропущена: выше start_marker", tbl_idx)
                            continue

                    # Если на этой странице есть end_marker — фильтрация по нему
                    if end_marker.lower() in low:
                        found_end = page.search(end_marker, case=False, regex=False)
                        if found_end:
                            end_y = found_end[0]['top']
                            logger.info(" Найден end_marker '%s' на странице %d, top y = %f",
                                        end_marker, page_num, end_y)
                            # таблица должна быть выше маркера конца
                            if top_y > end_y:
                                logger.info("  Таблица #%d пропущена: ниже end_marker", tbl_idx)
                                continue
                            # после end_marker можно остановить сбор
                            in_range = False
                            logger.info(" Диапазон завершён после end_marker — дальнейшие страницы игнорируются")

                    # Извлечение таблицы
                    try:
                        raw = table_obj.extract()
                    except Exception as e:
                        logger.warning("  Ошибка extract() для таблицы #%d на странице %d: %s", tbl_idx, page_num, e)
                        continue

                    df = pd.DataFrame(raw)
                    logger.info("  Таблица #%d принята — shape %s", tbl_idx, df.shape)
                    result_tables.append(df)

            # Если диапазон завершился — можно выйти
            if (not in_range) and (page_with_start is not None):
                logger.info("Обработка завершена на странице %d", page_num)
                break

    if not result_tables:
        logger.warning("Не найдено никаких таблиц между маркерами")
    else:
        logger.info("Всего найдено %d таблиц между маркерами", len(result_tables))

    return result_tables


if __name__ == "__main__":
    pdf_path = "your_document.pdf"
    tables = extract_tables_list_between_markers(pdf_path)
    for idx, df in enumerate(tables, start=1):
        print(f"--- Table #{idx}, rows: {df.shape[0]}, cols: {df.shape[1]} ---")
        print(df.head())
