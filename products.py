# pip install python-docx pandas openpyxl
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml.ns import qn
import pandas as pd
import json


def iter_block_items(document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _normalize_row_cells(row):
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
    Возвращает JSON-строку: {"text": ..., "table": ...}
    text — строка или ""
    table — либо "" или список (json) представление вложенной таблицы
    """
    # извлекаем текст
    texts = [p.text.strip() for p in cell.paragraphs if p.text.strip()]
    text = "\n".join(texts) if texts else ""

    # проверяем вложенные таблицы
    nested_tables = list(cell.tables)
    if nested_tables:
        # представим каждую вложенную таблицу как список dict
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
    if not cells:
        return []
    max_cols = len(cells)
    headers = [f"col_{i+1}" for i in range(max_cols)]
    row_obj = {}
    for i, cell in enumerate(cells):
        row_obj[headers[i]] = _extract_cell_json(cell)
    return [row_obj]


def _table_to_dicts(table):
    rows = list(table.rows)
    if not rows:
        return []

    norm_rows = [_normalize_row_cells(r) for r in rows]
    first_row = norm_rows[0]
    main_cols = len(first_row)
    headers = [f"col_{i+1}" for i in range(main_cols)]
    result = []

    for row_cells in norm_rows:
        row_obj = {h: None for h in headers}
        cell_count = len(row_cells)
        if cell_count <= main_cols:
            for idx in range(cell_count):
                row_obj[headers[idx]] = _extract_cell_json(row_cells[idx])
            for idx in range(cell_count, main_cols):
                row_obj[headers[idx]] = _extract_cell_json(row_cells[idx]) if idx < len(row_cells) else json.dumps({"text":"", "table":""}, ensure_ascii=False)
        else:
            # есть распад сверх main_cols
            for idx in range(main_cols):
                if idx < len(row_cells):
                    row_obj[headers[idx]] = _extract_cell_json(row_cells[idx])
                else:
                    row_obj[headers[idx]] = json.dumps({"text":"", "table":""}, ensure_ascii=False)

            # определить, к какой колонке относятся лишние ячейки
            target_idx = None
            for idx in range(main_cols):
                val_json = row_obj[headers[idx]]
                val = json.loads(val_json)
                if val["text"] == "" and val["table"] == "":
                    target_idx = idx
                    break
            if target_idx is None:
                target_idx = main_cols - 1

            extra_cells = row_cells[main_cols:]
            inner_cells = [row_cells[target_idx]] + extra_cells
            inner_table = _cells_to_inner_table(inner_cells)
            # заменяем содержимое target колонке
            row_obj[headers[target_idx]] = json.dumps({"text": "", "table": inner_table}, ensure_ascii=False)

        result.append(row_obj)

    return result


def extract_tables_with_text(docx_path):
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
        item["df"].to_csv(fname_csv, index=False, encoding="utf-8")
        item["df"].to_excel(fname_excel, index=False, engine="openpyxl")


def load_table_from_csv(fname_csv):
    """
    Загружает CSV и парсит JSON-ячейки обратно.
    Возвращает pd.DataFrame с обычными строками {text, table}.
    """
    df = pd.read_csv(fname_csv, dtype=str, encoding="utf-8")
    # парсим каждую ячейку
    for col in df.columns:
        df[col] = df[col].apply(lambda v: json.loads(v) if pd.notna(v) else {"text":"", "table":""})
    return df


if __name__ == "__main__":
    tables = extract_tables_with_text("example.docx")
    save_tables_to_csv(tables, base_filename="mydoc")
    # пример загрузки:
    df0 = load_table_from_csv("mydoc_table_1.csv")
    print(df0.iloc[0,0]["text"], df0.iloc[0,0]["table"])
