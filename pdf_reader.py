import pdfplumber
import pandas as pd
from typing import Optional, Tuple

def extract_tables_between_markers_with_bbox(
    pdf_path: str,
    start_marker: str = "паспорт продукта",
    end_marker: str = "приложение номер 1"
) -> Optional[pd.DataFrame]:
    in_range = False
    collected_rows = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_num = page.page_number
            text = page.extract_text() or ""
            low = text.lower()

            # ищем маркер начала на странице
            if (not in_range) and (start_marker.lower() in low):
                # нашли — считаем, что дальше на этой и следующих стр. таблицы могут идти
                in_range = True
                # найдём y-координату этого маркера (примерно)
                # используем page.search
                found = page.search(start_marker, case=False, regex=False)
                if found:
                    # берем первую найденную — get top
                    start_y = found[0]['top']
                else:
                    start_y = None
            # если уже в диапазоне — ищем таблицы
            if in_range:
                # перегоняем все таблицы на странице
                for table_obj in page.find_tables():
                    bbox: Tuple[float, float, float, float] = table_obj.bbox  # x0, top, x1, bottom
                    top_y = bbox[1]
                    bottom_y = bbox[3]

                    # логика фильтрации:
                    # — если мы на той же странице, что маркер начала: таблица должна быть **ниже** начала
                    if 'start_y' in locals() and start_y is not None and page_num == page.page_number:
                        if bottom_y < start_y:
                            # таблица выше маркера — пропускаем
                            continue
                    # — если маркер конца найден на этой странице — таблицы ниже конца пропускаем
                    if end_marker.lower() in low:
                        # найдём y-координату маркера конца
                        found_end = page.search(end_marker, case=False, regex=False)
                        if found_end:
                            end_y = found_end[0]['top']
                            # таблица должна быть **выше** маркера конца
                            if top_y > end_y:
                                # таблица ниже конца — пропускаем
                                continue
                        # после обработки маркера конца — можно завершить всё
                        in_range = False

                    # если таблица прошла фильтры — извлекаем текст
                    raw = table_obj.extract()
                    for row in raw:
                        collected_rows.append(row)

    if collected_rows:
        return pd.DataFrame(collected_rows)
    else:
        return None
