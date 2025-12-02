import pdfplumber
import pandas as pd
import logging
import json
from typing import List, Dict, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

class TableMeta:
    def __init__(self, table_obj, page_num: int):
        self.obj = table_obj
        self.page_num = page_num
        self.bbox: Tuple[float,float,float,float] = table_obj.bbox  # (x0, top, x1, bottom)
        self.rows: List[List[Optional[str]]] = table_obj.extract()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def to_serializable(self):
        return {
            "page": self.page_num,
            "bbox": self.bbox,
            "rows": self.rows
        }

def extract_top_level_tables(pdf_path: str,
                             start_marker: str = "паспорт продукта",
                             end_marker: str = "приложение номер 1"
                             ) -> List[Dict]:
    """
    Возвращает список словарей — для каждой top-level таблицы: её строка (as list of lists), bbox, страница,
    и список вложенных таблиц (каждая как serializable dict).
    """
    in_range = False
    page_with_start = None
    start_y: Optional[float] = None
    top_tables: List[Dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        logger.info("Открыт PDF '%s', страниц: %d", pdf_path, len(pdf.pages))

        # временно храним таблицы на текущей странице
        page_tables_buffer: List[TableMeta]

        for page in pdf.pages:
            page_num = page.page_number
            text = page.extract_text() or ""
            low = text.lower()

            if (not in_range) and (start_marker.lower() in low):
                in_range = True
                page_with_start = page_num
                logger.info("Найден start_marker на странице %d", page_num)
                found = page.search(start_marker, case=False, regex=False)
                if found:
                    start_y = found[0]['top']
                    logger.info(" start_marker top y = %f", start_y)

            if in_range:
                # собираем все таблицы на странице
                page_tables_buffer = []
                try:
                    tables = page.find_tables()
                except Exception as e:
                    logger.warning("Ошибка find_tables на странице %d: %s", page_num, e)
                    tables = []
                for tbl in tables:
                    meta = TableMeta(tbl, page_num)
                    page_tables_buffer.append(meta)
                if page_tables_buffer:
                    logger.info("Страница %d: найдено %d таблиц", page_num, len(page_tables_buffer))

                # фильтрация таблиц по start / end маркеру + bbox
                for meta in page_tables_buffer:
                    x0, top, x1, bottom = meta.bbox
                    # если на той же странице, что start — таблица должна быть ниже start_y
                    if page_num == page_with_start and (start_y is not None):
                        if bottom < start_y:
                            logger.info("Таблица на странице %d bbox %s пропущена — выше start_marker", page_num, meta.bbox)
                            continue
                    # если end_marker есть — фильтрация по нему
                    if end_marker.lower() in low:
                        found_end = page.search(end_marker, case=False, regex=False)
                        if found_end:
                            end_y = found_end[0]['top']
                            logger.info("Найден end_marker на странице %d, top y = %f", page_num, end_y)
                            if top > end_y:
                                logger.info("Таблица bbox %s пропущена — ниже end_marker", meta.bbox)
                                continue
                            # после этого — можем завершить диапазон
                            in_range = False

                    # добавляем в список кандидатов (временно)
                # если есть candidate — нужно фильтровать вложенные
                # проходимся по candidate-таблицам, отделяя top-level и nested
                for outer in page_tables_buffer:
                    is_nested = False
                    nested = []
                    for inner in page_tables_buffer:
                        if outer is inner:
                            continue
                        # если bbox inner полностью внутри bbox outer — считаем nested
                        x0o, topo, x1o, boto = outer.bbox
                        x0i, topi, x1i, boti = inner.bbox
                        if (x0i >= x0o and x1i <= x1o and topi >= topo and boti <= boto):
                            is_nested = True
                            nested.append(inner.to_serializable())
                    if not is_nested:
                        # top-level table — сохраняем
                        top_tables.append({
                            "page": outer.page_num,
                            "bbox": outer.bbox,
                            "rows": outer.rows,
                            "nested": nested  # список вложенных таблиц (в виде серилизованных dict)
                        })
                    else:
                        logger.info("Таблица на странице %d bbox %s — вложенная, пропускаем как top-level",
                                    outer.page_num, outer.bbox)
            if (not in_range) and (page_with_start is not None):
                logger.info("Диапазон закончился на странице %d — прекращаем", page_num)
                break

    if not top_tables:
        logger.warning("Не найдено top-level таблиц между маркерами")
    else:
        logger.info("Найдено top-level таблиц: %d", len(top_tables))

    return top_tables


if __name__ == "__main__":
    path = "your_document.pdf"
    result = extract_top_level_tables(path)
    for idx, tbl in enumerate(result, start=1):
        df = pd.DataFrame(tbl["rows"])
        print(f"--- Table #{idx} (page {tbl['page']}), shape {df.shape}, nested count = {len(tbl['nested'])} ---")
        print(df.head())
        if tbl["nested"]:
            print(" Nested tables json:", json.dumps(tbl["nested"], ensure_ascii=False))
