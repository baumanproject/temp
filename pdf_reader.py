import pdfplumber
import pandas as pd
from typing import List, Optional, Tuple

def extract_top_level_tables(
    pdf_path: str,
    start_marker: str = "паспорт продукта",
    end_marker: str = "приложение номер 1"
) -> List[pd.DataFrame]:
    tables_out: List[pd.DataFrame] = []

    with pdfplumber.open(pdf_path) as pdf:
        in_range = False
        start_y: Optional[float] = None
        page_with_start = None

        for page in pdf.pages:
            page_num = page.page_number
            text = page.extract_text() or ""
            low = text.lower()

            # если находим маркер начала — отмечаем старт
            if not in_range and start_marker.lower() in low:
                in_range = True
                page_with_start = page_num
                # пытаемся определить y-координату маркера, если pdfplumber поддерживает .search()
                found = page.search(start_marker, case=False, regex=False) if hasattr(page, "search") else None
                if found:
                    start_y = found[0]["top"]

            if in_range:
                # найдём все таблицы на странице
                try:
                    tbls = page.find_tables()
                except Exception:
                    tbls = []

                # буфер — таблицы на этой странице
                page_tables: List[Tuple[pdfplumber.table_table.Table, Tuple[float,float,float,float]]] = []
                for tbl in tbls:
                    try:
                        bbox = tbl.bbox  # (x0, top, x1, bottom)
                    except Exception:
                        # если bbox недоступен — принимаем таблицу без bbox
                        bbox = None
                    page_tables.append((tbl, bbox))

                # фильтр вложенных: если bbox одной таблицы содержится в bbox другой — считаем вложенной
                top_level = []
                for i, (outer_tbl, outer_bbox) in enumerate(page_tables):
                    is_nested = False
                    if outer_bbox is None:
                        # если нет bbox — пропускаем вложенность фильтр (берём как top-level)
                        top_level.append(outer_tbl)
                    else:
                        x0o, top_o, x1o, bot_o = outer_bbox
                        for j, (inner_tbl, inner_bbox) in enumerate(page_tables):
                            if i == j or inner_bbox is None:
                                continue
                            x0i, top_i, x1i, bot_i = inner_bbox
                            # если inner полностью внутри outer — treat inner as nested
                            if x0i >= x0o and x1i <= x1o and top_i >= top_o and bot_i <= bot_o:
                                is_nested = True
                                break
                        if not is_nested:
                            top_level.append(outer_tbl)

                for tbl in top_level:
                    # опционально: фильтрация по bbox относительно start_marker на странице начала
                    if page_num == page_with_start and start_y is not None:
                        bbox = tbl.bbox if hasattr(tbl, "bbox") else None
                        if bbox is not None:
                            # если таблица выше start_marker — пропускаем
                            if bbox[3] < start_y:
                                continue

                    # если на этой странице есть end_marker — можно прекратить после этого
                    if end_marker.lower() in low:
                        # но перед этим — можно фильтровать таблицы ниже end_marker
                        found_end = page.search(end_marker, case=False, regex=False) if hasattr(page, "search") else None
                        if found_end:
                            end_y = found_end[0]["top"]
                            bbox = tbl.bbox if hasattr(tbl, "bbox") else None
                            if bbox is not None and bbox[1] > end_y:
                                continue
                            # после обработки — завершаем
                        in_range = False

                    # извлекаем строки таблицы
                    try:
                        raw = tbl.extract()
                        df = pd.DataFrame(raw)
                        tables_out.append(df)
                    except Exception:
                        continue

            # если диапазон закончился — выходим
            if not in_range and page_with_start is not None:
                break

    return tables_out


if __name__ == "__main__":
    pdf_path = "your_file.pdf"
    tables = extract_top_level_tables(pdf_path)
    for idx, df in enumerate(tables, start=1):
        print(f"Table {idx}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(df.head())
