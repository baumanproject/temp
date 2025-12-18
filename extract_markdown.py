import re

from docx import Document
from rapidfuzz import fuzz


_WS = re.compile(r"\s+")
_JUNK = re.compile(r"[^0-9a-zA-Zа-яА-ЯёЁ%№\s]+")


def _norm(s: str) -> str:
    s = (s or "").strip().casefold().replace("ё", "е")
    s = _JUNK.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _unique_row_cells(row) -> list:
    """
    row.cells может содержать дубликаты для merged-cells.  [oai_citation:6‡Stack Overflow](https://stackoverflow.com/questions/48090922/python-docx-row-cells-return-a-merged-cell-multiple-times?utm_source=chatgpt.com)
    Дедуп по identity xml-ячейки (_tc).
    """
    uniq = []
    seen = set()
    for c in row.cells:
        tc = getattr(c, "_tc", None)
        key = id(tc) if tc is not None else id(c)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _table_to_text(table, depth: int = 0, max_depth: int = 5) -> str:
    """
    Сериализация таблицы в простой текст.
    Вложенные таблицы тоже печатаем (но ограничиваем глубину на всякий случай).
    """
    if depth > max_depth:
        return ""

    lines = []
    for row in table.rows:
        cells = _unique_row_cells(row)
        cell_texts = []
        for cell in cells:
            cell_texts.append(_cell_to_text(cell, depth=depth + 1, max_depth=max_depth))
        # как “рядом” — через разделитель
        row_line = " | ".join([t for t in cell_texts if t.strip() != ""]).strip()
        if row_line:
            lines.append(row_line)

    return "\n".join(lines).strip()


def _cell_to_text(cell, depth: int = 0, max_depth: int = 5) -> str:
    """
    Берём весь текст ячейки:
    - параграфы
    - вложенные таблицы (как текст подряд)  [oai_citation:7‡GitHub](https://github.com/python-openxml/python-docx/issues/769?utm_source=chatgpt.com)
    """
    parts = []

    # параграфы
    for p in cell.paragraphs:
        t = (p.text or "").strip()
        if t:
            parts.append(t)

    # вложенные таблицы
    for t in cell.tables:
        tt = _table_to_text(t, depth=depth + 1, max_depth=max_depth)
        if tt:
            parts.append(tt)

    return "\n".join(parts).strip()


def _iter_tables_recursive(tables) -> list:
    """
    Возвращает список всех таблиц в порядке обхода:
    - текущие tables
    - затем вложенные из каждой ячейки (рекурсивно)
    """
    all_tables = []
    for table in tables:
        all_tables.append(table)
        for row in table.rows:
            for cell in _unique_row_cells(row):
                nested = cell.tables  # nested tables живут отдельно  [oai_citation:8‡GitHub](https://github.com/python-openxml/python-docx/issues/769?utm_source=chatgpt.com)
                if nested:
                    all_tables.extend(_iter_tables_recursive(nested))
    return all_tables


def extract_mapping_from_docx(
    docx_path: str,
    attributes: list[str],
    threshold: float = 85.0,
) -> tuple[dict, float, list]:
    """
    mapping: attr -> value|None
    recall: found/total
    details: список (attr, score, table_idx, row_idx, matched_cell_idx)
    """

    doc = Document(docx_path)
    tables = _iter_tables_recursive(doc.tables)

    mapping = {a: None for a in attributes}
    details = []

    found = 0

    # Чтобы было “самое раннее”, сканируем таблицы/строки сверху вниз
    # и для каждого attr берём ПЕРВУЮ строку, которая проходит threshold.
    for attr in attributes:
        attr_n = _norm(attr)

        matched = False
        best_seen_score = 0.0
        best_seen_info = None

        for ti, table in enumerate(tables):
            for ri, row in enumerate(table.rows):
                cells = _unique_row_cells(row)
                cell_texts = [_cell_to_text(c) for c in cells]

                # найдём лучшую ячейку в строке по score
                best_score = 0.0
                best_ci = None

                for ci, txt in enumerate(cell_texts):
                    txt_n = _norm(txt)
                    if not txt_n:
                        continue
                    score = float(fuzz.WRatio(attr_n, txt_n))  # общий скорер  [oai_citation:9‡RapidFuzz](https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html?utm_source=chatgpt.com)
                    if score > best_score:
                        best_score = score
                        best_ci = ci

                if best_score > best_seen_score:
                    best_seen_score = best_score
                    best_seen_info = (ti, ri, best_ci)

                if best_ci is None:
                    continue

                if best_score >= threshold:
                    # Значение = все остальные ячейки в строке (кроме matched cell)
                    value_parts = []
                    for ci, txt in enumerate(cell_texts):
                        if ci == best_ci:
                            continue
                        tt = (txt or "").strip()
                        if tt:
                            value_parts.append(tt)

                    value = "\n".join(value_parts).strip()
                    if not value:
                        # пустое значение = считаем не найдено
                        mapping[attr] = None
                        details.append((attr, best_score, ti, ri, best_ci, "EMPTY_VALUE"))
                    else:
                        mapping[attr] = value
                        found += 1
                        details.append((attr, best_score, ti, ri, best_ci, "OK"))

                    matched = True
                    break

            if matched:
                break

        if not matched:
            # чтобы ты мог крутить threshold — отдаём лучший score, который вообще встречался
            if best_seen_info is None:
                details.append((attr, 0.0, None, None, None, "NO_MATCH"))
            else:
                ti, ri, ci = best_seen_info
                details.append((attr, best_seen_score, ti, ri, ci, "BELOW_THRESHOLD"))

    recall = found / max(1, len(attributes))
    return mapping, recall, details
