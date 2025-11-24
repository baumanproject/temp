# pip install python-docx pandas openpyxl
import os
import json
import csv
import re
from collections import Counter

import pandas as pd
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml.ns import qn


# ================== базовые утилиты ================== #

def iter_block_items(document):
    """
    Идём по документу в порядке: параграфы / таблицы.
    """
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _normalize_row_cells(row):
    """
    Убираем дублирующиеся подряд ячейки (одинаковый XML) — артефакт merge’ов.
    Возвращаем список ячеек.
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


# ================== извлечение содержимого ячейки ================== #

def _extract_cell_json(cell):
    """
    Ячейка -> JSON-строка {"text": str, "table": ...}

    text  — весь текст ячейки (с переносами).
    table — "" либо список вложенных таблиц (каждая — list[dict]).
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

    obj = {"text": text, "table": table_field}
    return json.dumps(obj, ensure_ascii=False)


def _cells_to_inner_table(cells):
    """
    Список ячеек -> "внутренняя таблица" из одной строки.
    Возвращает list[dict] с ключами col_1..col_k,
    значения — JSON-ячейки.
    """
    if not cells:
        return []
    max_cols = len(cells)
    headers = [f"col_{i+1}" for i in range(max_cols)]
    row_obj = {}
    for i, cell in enumerate(cells):
        row_obj[headers[i]] = _extract_cell_json(cell)
    return [row_obj]


# ================== определение количества колонок ================== #

_index_re = re.compile(r"^\s*\d+([.)])?\s*$")

def _is_index_column(rows, idx, max_rows=5):
    """
    Проверяет, является ли колонка с индексом idx (0-based) индекс-колонкой:
    по первым max_rows строкам значение text соответствует цифре-индексу
    и поле table пусто.
    """
    score = 0
    n = min(len(rows), max_rows)
    header = f"col_{idx+1}"
    for row in rows[:n]:
        if header not in row:
            continue
        val_json = row.get(header)
        if not isinstance(val_json, str):
            continue
        try:
            obj = json.loads(val_json)
        except Exception:
            continue
        text = obj.get("text", "").strip()
        table_field = obj.get("table", "")
        if _index_re.match(text) and (table_field == "" or table_field == []):
            score += 1
    return score >= max(2, int(n * 0.6))


def _determine_main_cols(norm_rows):
    """
    Определяем число основных колонок и учитываем возможность
    индексной первой колонки.
    Возвращает tuple (main_cols, dropped_index_flag)
    """
    counts = [len(r) for r in norm_rows[:5]]
    if counts and counts[0] == 1:
        counts = counts[1:]
    if not counts:
        return 0, False

    ctr = Counter(counts)
    most_common, freq = ctr.most_common(1)[0]
    if freq > 1:
        cols = most_common
    else:
        cols = min(counts)

    headers = [f"col_{i+1}" for i in range(cols)]
    sample_rows = []
    for r in norm_rows[:5]:
        row_obj = {}
        for i in range(cols):
            if i < len(r):
                row_obj[f"col_{i+1}"] = _extract_cell_json(r[i])
            else:
                row_obj[f"col_{i+1}"] = json.dumps({"text":"", "table":""}, ensure_ascii=False)
        sample_rows.append(row_obj)

    if _is_index_column(sample_rows, 0):
        return cols - 1, True

    return cols, False


# ================== конвертация таблицы ================== #

def _table_to_dicts(table):
    """
    Table -> list[dict], ключи col_1..col_N, значения — JSON-строки ячеек.
    Учитывает:
      - эвристику по числу колонок;
      - "распад" ячеек;
      - отбрасывание индекс-колонки.
    """
    rows = list(table.rows)
    if not rows:
        return []

    norm_rows = [_normalize_row_cells(r) for r in rows]

    main_cols, dropped_index = _determine_main_cols(norm_rows)
    if main_cols <= 0:
        main_cols = len(norm_rows[0])

    drop_offset = 1 if dropped_index else 0
    headers = [f"col_{i+1}" for i in range(main_cols)]

    result = []
    empty_json = json.dumps({"text": "", "table": ""}, ensure_ascii=False)

    for row_cells in norm_rows:
        row_obj = {h: empty_json for h in headers}
        cell_count = len(row_cells)

        if cell_count - drop_offset <= main_cols:
            for idx in range(main_cols):
                src_idx = idx + drop_offset
                if src_idx < cell_count:
                    row_obj[headers[idx]] = _extract_cell_json(row_cells[src_idx])
                else:
                    row_obj[headers[idx]] = empty_json
        else:
            for idx in range(main_cols):
                src_idx = idx + drop_offset
                if src_idx < cell_count:
                    row_obj[headers[idx]] = _extract_cell_json(row_cells[src_idx])
                else:
                    row_obj[headers[idx]] = empty_json

            extra_cells = row_cells[drop_offset + main_cols:]
            target_idx = None
            for idx in range(main_cols):
                obj = json.loads(row_obj[headers[idx]])
                if obj["text"] == "" and obj["table"] == "":
                    target_idx = idx
                    break
            if target_idx is None:
                target_idx = main_cols - 1

            inner_cells = [row_cells[drop_offset + target_idx]] + extra_cells
            inner_table = _cells_to_inner_table(inner_cells)
            row_obj[headers[target_idx]] = json.dumps(
                {"text": "", "table": inner_table},
                ensure_ascii=False
            )

        result.append(row_obj)

    return result


# ================== работа с docx ================== #

def extract_tables_with_text(docx_path):
    """
    Возвращает список:
    [
      {
        "text_before": <str или None>,
        "df": <pd.DataFrame с колонками col_i, в ячейках JSON-строки>
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
                "df": df,
            })

    return results


# ================== сохранение / загрузка ================== #

def save_tables_to_csv(tables, base_filename="output"):
    """
    Сохраняет для одного docx все таблицы в CSV+XLSX:
    output_table_1.csv, output_table_1.xlsx, ...
    """
    for idx, item in enumerate(tables, start=1):
        fname_csv = f"{base_filename}_table_{idx}.csv"
        fname_xlsx = f"{base_filename}_table_{idx}.xlsx"
        item["df"].to_csv(
            fname_csv,
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL
        )
        item["df"].to_excel(
            fname_xlsx,
            index=False,
            engine="openpyxl"
        )


def load_table_from_csv(fname_csv):
    """
    Чтение CSV назад: каждая ячейка -> dict {"text":..., "table":...}.
    """
    df = pd.read_csv(fname_csv, dtype=str, encoding="utf-8")
    for col in df.columns:
        df[col] = df[col].apply(
            lambda v: json.loads(v) if isinstance(v, str) and v.strip() else {"text": "", "table": ""}
        )
    return df


# ================== обход папок ИСЖ / ДСЖ / НСЖ ================== #

def process_folder(folder_path):
    """
    Обрабатывает все .docx в папке.
    Возвращает список:
    [
      {
        "filename": "...docx",
        "tables": [
           {"text_before": ..., "table": [ {row}, {row}, ... ]},
           ...
        ]
      },
      ...
    ]
    """
    results = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(".docx"):
            continue
        full_path = os.path.join(folder_path, fname)
        try:
            tables = extract_tables_with_text(full_path)
            serialized_tables = []
            for tbl in tables:
                serialized_tables.append({
                    "text_before": tbl["text_before"],
                    "table": tbl["df"].to_dict(orient="records")
                })
            results.append({
                "filename": fname,
                "tables": serialized_tables
            })
        except Exception as exc:
            print(f"Ошибка обработки {full_path}: {exc}")
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

    # сохраняем по категории
    for cat, data in all_results.items():
        out_path = os.path.join(base_dir, f"{cat}_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    return all_results


if __name__ == "__main__":
    results = main()
    for cat, arr in results.items():
        print(f"{cat}: обработано {len(arr)} файлов")
