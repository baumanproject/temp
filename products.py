# pip install python-docx pandas
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


def _extract_cell_value(cell):
    texts = [p.text.strip() for p in cell.paragraphs if p.text.strip()]
    text = "\n".join(texts) if texts else ""
    nested_tables = list(cell.tables)
    nested_data = []
    for t in nested_tables:
        nested_data.append(_table_to_dicts(t))
    if nested_tables and text:
        value = {"text": text, "tables": nested_data}
        return json.dumps(value, ensure_ascii=False)
    if nested_tables and not text:
        return json.dumps(nested_data, ensure_ascii=False)
    return text


def _cells_to_inner_table(cells):
    if not cells:
        return []
    max_cols = len(cells)
    headers = [f"col_{i+1}" for i in range(max_cols)]
    row_obj = {headers[i]: _extract_cell_value(cells[i]) for i in range(max_cols)}
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
            # обычный случай: нет "распада"
            for idx in range(cell_count):
                row_obj[headers[idx]] = _extract_cell_value(row_cells[idx])
        else:
            # есть "распад" — надо выяснить к какой колонке
            #  логика: сравниваем содержание первых main_cols-1 ячеек,
            #  и оставшиеся считать "распадом" той колонки,
            #  для которой ячейка (в первой main_cols) пустая или дублируется.
            #  Простой эвристический подход:
            for idx in range(main_cols):
                if idx < len(row_cells):
                    val = _extract_cell_value(row_cells[idx])
                else:
                    val = None
                row_obj[headers[idx]] = val
            # ищем колонку-индекс куда отнести лишние ячейки:
            # например, если первые k ячеек похожи на первая строку,
            # а далее идут лишние, то лишние относим к колонке k+1.
            # Простой вариант: отнести лишние к последней колонке, у которой нет контента в этой строке
            target_idx = None
            for idx in range(main_cols):
                if row_obj[headers[idx]] in (None, ""):
                    target_idx = idx
                    break
            if target_idx is None:
                target_idx = main_cols - 1
            # собираем лишние ячейки
            extra_cells = row_cells[main_cols:]
            inner_cells = [row_cells[target_idx]] + extra_cells
            inner_table = _cells_to_inner_table(inner_cells)
            row_obj[headers[target_idx]] = json.dumps(inner_table, ensure_ascii=False)
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
            results.append({"text_before": caption, "df": df})
    return results


if __name__ == "__main__":
    tables = extract_tables_with_text("example.docx")
    for i, item in enumerate(tables, start=1):
        print(f"=== Таблица {i} ===")
        print("Текст перед таблицей:", item["text_before"])
        print(item["df"])
        print("-" * 50)
