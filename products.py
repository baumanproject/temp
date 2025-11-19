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
    Убираем дубли ячеек в строке.
    Частая проблема кривых merge'ов — соседние ячейки полностью дублируют друг друга.
    Считаем дубликатами подряд идущие ячейки с одинаковым XML.
    """
    normalized = []
    prev_xml = None
    for cell in row.cells:
        xml = cell._tc.xml
        if xml == prev_xml:
            # пропускаем дубль
            continue
        prev_xml = xml
        normalized.append(cell)
    return normalized


def _safe_header_names(raw_headers):
    """
    Делает имена колонок уникальными и не пустыми.
    """
    headers = []
    counter = {}

    for i, h in enumerate(raw_headers):
        name = h.strip()
        if not name:
            name = f"col_{i+1}"

        base = name
        if name in counter:
            counter[base] += 1
            name = f"{base}_{counter[base]}"
        else:
            counter[base] = 0

        headers.append(name)
    return headers


def _extract_cell_value(cell):
    """
    Достаём значение ячейки:
    - если есть вложенные таблицы → JSON (dict / list)
    - если только текст → строка
    - если и текст, и таблицы → JSON с полями text + tables
    """
    # обычный текст
    texts = [p.text.strip() for p in cell.paragraphs if p.text.strip()]
    text = "\n".join(texts) if texts else ""

    # вложенные таблицы
    nested_tables = list(cell.tables)
    nested_data = []

    for t in nested_tables:
        nested_data.append(_table_to_dicts(t))

    if nested_tables and text:
        value = {"text": text, "tables": nested_data}
        return json.dumps(value, ensure_ascii=False)

    if nested_tables and not text:
        # только таблицы
        return json.dumps(nested_data, ensure_ascii=False)

    # только текст
    return text


def _table_to_dicts(table):
    """
    Превращает Table → список словарей (строки).
    Вложенные таблицы в ячейках уже сериализуются в JSON.
    """
    rows = list(table.rows)
    if not rows:
        return []

    # нормализуем строки (убираем дубли ячеек)
    norm_rows = [_normalize_row_cells(r) for r in rows]

    # заголовок
    header_cells = norm_rows[0]
    raw_headers = [_extract_cell_value(c) for c in header_cells]
    headers = _safe_header_names(raw_headers)

    result = []

    for row_cells in norm_rows[1:]:
        row_obj = {}
        # гарантируем, что по каждому заголовку есть ключ
        for idx, col_name in enumerate(headers):
            cell_value = None
            if idx < len(row_cells):
                cell_value = _extract_cell_value(row_cells[idx])
            row_obj[col_name] = cell_value
        result.append(row_obj)

    return result


def extract_tables_with_text(docx_path):
    """
    Основная функция.

    На выходе:
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

    # буфер текста между предыдущей таблицей и текущей
    text_buffer = []

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            txt = block.text.strip()
            if txt:
                text_buffer.append(txt)

        elif isinstance(block, Table):
            # текст перед таблицей (если он был)
            caption = "\n".join(text_buffer).strip() or None
            text_buffer = []

            # конвертируем таблицу → список dict → DataFrame
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
    # пример использования
    path = "example.docx"
    tables = extract_tables_with_text(path)

    for i, item in enumerate(tables, start=1):
        print(f"=== Таблица {i} ===")
        print("Текст перед таблицей:")
        print(item["text_before"])
        print("\nDataFrame:")
        print(item["df"])
        print("\n" + "-" * 80 + "\n")
