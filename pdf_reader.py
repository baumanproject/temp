import pdfplumber
import pandas as pd
import logging
from typing import List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

def extract_top_level_tables(
    pdf_path: str,
    start_marker: str = "паспорт продукта",
    end_marker: str = "приложение номер 1"
) -> List[pd.DataFrame]:
    result_tables: List[pd.DataFrame] = []

    with pdfplumber.open(pdf_path) as pdf:
        in_range = False
        start_y: Optional[float] = None
        page_with_start: Optional[int] = None

        for page in pdf.pages:
            page_num = page.page_number
            text = page.extract_text() or ""
            low = text.lower()

            # Если находим маркер старта — включаем диапазон
            if (not in_range) and (start_marker.lower() in low):
                in_range = True
                page_with_start = page_num
                logger.info("Start marker '%s' found on page %d", start_marker, page_num)
                # Попробуем найти координату маркера
                try:
                    found = page.search(start_marker, case=False, regex=False)
                except Exception:
                    found = None
                if found:
                    start_y = found[0]["top"]
                    logger.info(" start_marker top y = %f", start_y)

            if in_range:
                # Найти все таблицы на странице
                try:
                    tables = page.find_tables()
                except Exception as e:
                    logger.warning("Error finding tables on page %d: %s", page_num, e)
                    tables = []

                if not tables:
                    # нет таблиц — просто пропускаем
                    continue

                logger.info("Page %d: %d tables detected", page_num, len(tables))

                # Сбор bbox + объекты таблиц
                tbls_meta: List[Tuple[pdfplumber.table_table.Table, Tuple[float,float,float,float]]] = []
                for tbl in tables:
                    try:
                        bbox = tbl.bbox  # (x0, top, x1, bottom)
                    except Exception:
                        bbox = None
                    tbls_meta.append((tbl, bbox))

                # Определить top-level: те, которые не являются вложенными
                top_level_tbls = []
                for i, (outer_tbl, outer_bbox) in enumerate(tbls_meta):
                    bbox_o = outer_bbox
                    is_nested = False
                    if bbox_o is None:
                        # Если нет bbox — не фильтруем вложенность, считаем внешней
                        top_level_tbls.append(outer_tbl)
                        continue

                    x0o, top_o, x1o, bot_o = bbox_o
                    for j, (inner_tbl, inner_bbox) in enumerate(tbls_meta):
                        if i == j or inner_bbox is None:
                            continue
                        x0i, top_i, x1i, bot_i = inner_bbox
                        # если inner полностью внутри outer — inner nested, но это не влияет на outer
                        if x0i >= x0o and x1i <= x1o and top_i >= top_o and bot_i <= bot_o:
                            # Found inner inside outer — but outer still top-level
                            # We do NOT mark outer as nested
                            continue
                    # outer не помечен как вложенный — принимаем
                    top_level_tbls.append(outer_tbl)

                # Извлекаем каждую top-level таблицу
                for tbl in top_level_tbls:
                    # Фильтрация по start_marker на странице начала
                    if page_num == page_with_start and (start_y is not None):
                        bbox = tbl.bbox if hasattr(tbl, "bbox") else None
                        if bbox is not None:
                            # если таблица выше start_marker — пропускаем
                            if bbox[3] < start_y:
                                logger.info("Table on page %d bbox %s skipped: above start_marker", page_num, bbox)
                                continue

                    # Фильтрация по end_marker, если он на этой странице
                    if end_marker.lower() in low:
                        try:
                            found_end = page.search(end_marker, case=False, regex=False)
                        except Exception:
                            found_end = None
                        if found_end:
                            end_y = found_end[0]["top"]
                            bbox = tbl.bbox if hasattr(tbl, "bbox") else None
                            if bbox is not None and bbox[1] > end_y:
                                logger.info("Table on page %d bbox %s skipped: below end_marker", page_num, bbox)
                                continue
                            # после обработки end_marker — можно завершить диапазон
                        in_range = False

                    # Извлекаем содержимое и добавляем
                    try:
                        rows = tbl.extract()
                    except Exception as e:
                        logger.warning("Error extracting table on page %d: %s", page_num, e)
                        continue

                    df = pd.DataFrame(rows)
                    result_tables.append(df)
                    logger.info("Table extracted: page %d, shape %s", page_num, df.shape)

            # Если диапазон закончился — прекращаем
            if (not in_range) and (page_with_start is not None):
                logger.info("Extraction range finished at page %d", page_num)
                break

    logger.info("Total tables extracted: %d", len(result_tables))
    return result_tables


if __name__ == "__main__":
    path = "your.pdf"
    tables = extract_top_level_tables(path)
    for idx, df in enumerate(tables, start=1):
        print(f"--- Table #{idx} ---")
        print(df.head())
        print("Shape:", df.shape)
