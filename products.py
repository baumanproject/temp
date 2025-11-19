# pip install python-docx pandas
from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.oxml.ns import qn
import pandas as pd
import json


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
    Убираем дубли ячеек в строке (одинаковый XML подряд).
    Частая проблема кривых merge'ов.
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


def _extract_cell_value(cell):
    """
    Значение ячейки:
    - вложенные таблицы → JSON
    - текст → строка
    - текст + таблицы → JSON {text, tables}
    """
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


def _table_to_dicts(table):
    """
    Table → список словарей (строки).
    Имена колонок: col_1, col_2, ...
    Все строки считаются данными, никаких заголовков.
    """
    rows = list(table.rows)
    if not rows:
        return []

    # нормализуем строки (убираем дубли ячеек)
    norm_rows = [_normalize_row_cells(r) for r in rows]

    # определяем максимальное число колонок
    max_cols = max(len(r) for r in norm_rows) if norm_rows else 0
    headers = [f"col_{i+1}" for i in range(max_cols)]

    result = []

    for row_cells in norm_rows:
        row_obj = {}
        for idx in range(max_cols):
            col_name = headers[idx]
            cell_value = None
            if idx < len(row_cells):
                cell_value = _extract_cell_value(row_cells[idx])
            row_obj[col_name] = cell_value
        result.append(row_obj)

    return result


def extract_tables_with_text(docx_path):
    """
    На выходе список:
    [
        {
            "text_before": <str или None>,
            "df": <pd.DataFrame с колонками col_i>
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

            results.append(
                {
                    "text_before": caption,
                    "df": df,
                }
            )

    return results


if __name__ == "__main__":
    path = "example.docx"
    tables = extract_tables_with_text(path)

    for i, item in enumerate(tables, start=1):
        print(f"=== Таблица {i} ===")
        print("Текст перед таблицей:")
        print(item["text_before"])
        print("\nDataFrame:")
        print(item["df"])
        print("\n" + "-" * 80 + "\n")
