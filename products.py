# pip install python-docx pandas openpyxl
import os
import json
from collections import Counter

import pandas as pd
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml.ns import qn


def iter_block_items(document):
    """
    Итерация по блокам документа (параграфы и таблицы) в порядке появления.
    """
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _normalize_row_cells(row):
    """
    Убираем дубли ячеек в строке (одинаковый XML подряд) — часто из-за merge’ов.
    """
    normalized = []
    prev_xml = None
    for cell in row.cells:
        xml = cell._tc.xml
        if xml == prev_xml:
            continue
        prev_xml = xml
        normalized.append(cell)
    return normalized


def _extract_cell_json(cell):
    """
    Извлекает содержимое ячейки: 
    - текст → поле "text"
    - вложенные таблицы → поле "table" (список)
    Возвращает JSON-строку.
    """
    texts = [p.text.strip() for p in cell.paragraphs if p.text.strip()]
    text = "\n".join(texts) if texts else ""

    nested_tables = list(cell.tables)
    if nested_tables:
        tables_data = []
        for t in nested_tables:
            tables_data.append(_table_to_dicts(t))
        table_field = tables_data
    else:
        table_field = ""

    obj = {
        "text": text,
        "table": table_field
    }
    return json.dumps(obj, ensure_ascii=False)


def _cells_to_inner_table(cells):
    """
    Превращает список ячеек в «внутреннюю таблицу» одной строки.
    Возвращает список из одного словаря: {col_1: json, col_2: json, …}
    """
    if not cells:
        return []
    max_cols = len(cells)
    headers = [f"col_{i+1}" for i in range(max_cols)]
    row_obj = {}
    for i, cell in enumerate(cells):
        row_obj[headers[i]] = _extract_cell_json(cell)
    return [row_obj]


def _determine_main_cols(norm_rows):
    """
    Определяем число основных колонок:
    - Берём до первых 3 строк таблицы (после нормализации).
    - Если первая строка содержит 1 ячейку → считаем, что это подпись/текст и игнорируем.
    - Среди оставшихся строк (2-я и 3-я) смотрим число ячеек, выбираем наиболее частое.
    - Если нет явного большинства — выбираем меньшее из встреченных.
    """
    counts = [len(r) for r in norm_rows[:3]]
    if counts and counts[0] == 1:
        counts = counts[1:]
    if not counts:
        return 0
    ctr = Counter(counts)
    most_common, freq = ctr.most_common(1)[0]
    if freq > 1:
        return most_common
    return min(counts)


def _table_to_dicts(table):
    """
    Преобразует объект Table в список словарей (строки),
    где ключи — col_1, col_2, …, а значения — JSON-строки ячеек.
    Обрабатывает случаи, когда в строке больше ячеек, чем main_cols (распад).
    """
    rows = list(table.rows)
    if not rows:
        return []

    norm_rows = [_normalize_row_cells(r) for r in rows]

    main_cols = _determine_main_cols(norm_rows)
    if main_cols <= 0:
        # fallback: если не удалось определить, берём по первой строке
        main_cols = len(norm_rows[0])

    headers = [f"col_{i+1}" for i in range(main_cols)]

    result = []
    for row_cells in norm_rows:
        row_obj = {h: None for h in headers}
        cell_count = len(row_cells)

        if cell_count <= main_cols:
            for idx in range(cell_count):
                row_obj[headers[idx]] = _extract_cell_json(row_cells[idx])
            for idx in range(cell_count, main_cols):
                row_obj[headers[idx]] = json.dumps({"text": "", "table": ""}, ensure_ascii=False)
        else:
            # есть «распад»: больше ячеек, чем основных колонок
            for idx in range(main_cols):
                if idx < len(row_cells):
                    row_obj[headers[idx]] = _extract_cell_json(row_cells[idx])
                else:
                    row_obj[headers[idx]] = json.dumps({"text": "", "table": ""}, ensure_ascii=False)

            # определяем, к какой колонке относятся лишние ячейки
            target_idx = None
            for idx in range(main_cols):
                val = json.loads(row_obj[headers[idx]])
                if val["text"] == "" and val["table"] == "":
                    target_idx = idx
                    break
            if target_idx is None:
                target_idx = main_cols - 1

            extra_cells = row_cells[main_cols:]
            inner_cells = [row_cells[target_idx]] + extra_cells
            inner_table = _cells_to_inner_table(inner_cells)
            row_obj[headers[target_idx]] = json.dumps({"text": "", "table": inner_table}, ensure_ascii=False)

        result.append(row_obj)

    return result


def extract_tables_with_text(docx_path):
    """
    Основная функция: извлекает из .docx все таблицы с предшествующим текстом.
    Возвращает список:
      [
        {
          "text_before": <str или None>,
          "df": <pd.DataFrame>
        },
        ...
      ]
    """
    doc = Document(docx_path)
    results = []
    text_buffer = []

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            txt = block.text.strip()
            if txt:
                text_buffer.append(txt)
        elif isinstance(block, Table):
            caption = "\n".join(text_buffer).strip() or None
            text_buffer = []

            table_data = _table_to_dicts(block)
            df = pd.DataFrame(table_data) if table_data else pd.DataFrame()

            results.append({
                "text_before": caption,
                "df": df
            })

    return results


def save_tables_to_csv(tables, base_filename="output"):
    """
    Сохраняет каждую таблицу в CSV и Excel.
    """
    for idx, item in enumerate(tables, start=1):
        fname_csv = f"{base_filename}_table_{idx}.csv"
        fname_excel = f"{base_filename}_table_{idx}.xlsx"
        item["df"].to_csv(fname_csv, index=False, encoding="utf-8", quoting=pd.io.common.csv.QUOTE_ALL)
        item["df"].to_excel(fname_excel, index=False, engine="openpyxl")


def load_table_from_csv(fname_csv):
    """
    Загружает CSV и парсит JSON-ячейки обратно в объекты Python.
    Возвращает pd.DataFrame, где каждая ячейка — dict {"text":…, "table":…}
    """
    df = pd.read_csv(fname_csv, dtype=str, encoding="utf-8")
    for col in df.columns:
        df[col] = df[col].apply(lambda v: json.loads(v) if pd.notna(v) else {"text":"", "table":""})
    return df


def process_folder(folder_path):
    """
    Обрабатывает все .docx-файлы в указанной папке и возвращает список результатов.
    """
    results = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".docx"):
            full_path = os.path.join(folder_path, fname)
            try:
                tables = extract_tables_with_text(full_path)
                serialized = []
                for tbl in tables:
                    serialized.append({
                        "text_before": tbl["text_before"],
                        "table": tbl["df"].to_dict(orient="records")
                    })
                results.append({
                    "filename": fname,
                    "tables": serialized
                })
            except Exception as e:
                print(f"Ошибка обработки {full_path}: {e}")
    return results


def main():
    base_dir = "/data/products"
    categories = ["ИСЖ", "ДСЖ", "НСЖ"]
    all_results = {}

    for cat in categories:
        folder = os.path.join(base_dir, cat)
        if not os.path.isdir(folder):
            print(f"Папка не найдена: {folder}")
            continue
        all_results[cat] = process_folder(folder)

    # Сохраняем результаты в JSON-файлы
    for cat, data in all_results.items():
        out_path = os.path.join(base_dir, f"{cat}_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return all_results


if __name__ == "__main__":
    results = main()
    for cat, arr in results.items():
        print(f"Категория {cat}: обработано {len(arr)} файлов")
