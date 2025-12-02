import logging
from typing import List

from docx import Document
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def extract_docx_tables_between_markers(
    docx_path: str,
    start_marker: str = "паспорт продукта",
    end_marker: str = "приложение номер 1",
) -> List[pd.DataFrame]:
    """
    Извлекает ТОЛЬКО внешние (top-level) таблицы из DOCX между абзацами,
    содержащими start_marker и end_marker.

    Возвращает список pd.DataFrame (без имён колонок).
    """
    doc = Document(docx_path)
    logger.info("Открыт DOCX '%s'", docx_path)

    start_marker_l = start_marker.lower()
    end_marker_l = end_marker.lower()

    in_range = False
    found_start = False
    found_end = False

    result_tables: List[pd.DataFrame] = []

    # iter_inner_content() — даёт Paragraph | Table в порядке документа
    # и не лезет внутрь ячеек (nested-таблицы сами не появятся).
    # см. доку python-docx: Document.iter_inner_content
    for idx, block in enumerate(doc.iter_inner_content()):
        # --- абзац ---
        if block.__class__.__name__ == "Paragraph":
            text = (block.text or "").strip()
            low = text.lower()

            if not in_range and start_marker_l in low:
                in_range = True
                found_start = True
                logger.info(
                    "Найден start_marker '%s' в абзаце #%d: %r",
                    start_marker, idx, text,
                )
                continue

            if in_range and end_marker_l in low:
                found_end = True
                in_range = False
                logger.info(
                    "Найден end_marker '%s' в абзаце #%d: %r. Диапазон завершён.",
                    end_marker, idx, text,
                )
                break  # дальше нам уже не интересны таблицы

        # --- таблица ---
        elif block.__class__.__name__ == "Table":
            # Таблицы нас интересуют только внутри диапазона
            if not in_range:
                continue

            # Это top-level таблица (nested сюда не попадает, мы не ходим в cell.tables)
            try:
                rows = []
                for row in block.rows:
                    cells_text = [cell.text for cell in row.cells]
                    rows.append(cells_text)

                df = pd.DataFrame(rows)
                result_tables.append(df)

                logger.info(
                    "Добавлена таблица из блока #%d: shape=%s",
                    idx, df.shape,
                )
            except Exception as e:
                logger.warning(
                    "Ошибка при извлечении таблицы из блока #%d: %s",
                    idx, e,
                )
                continue

        else:
            # На всякий случай логируем неожиданный тип
            logger.debug("Неизвестный тип блока #%d: %s", idx, type(block))

    if not found_start:
        logger.warning("start_marker '%s' не найден в документе", start_marker)
    if not found_end:
        logger.warning("end_marker '%s' не найден в документе", end_marker)

    logger.info("Всего найдено внешних таблиц между маркерами: %d", len(result_tables))
    return result_tables


if __name__ == "__main__":
    path = "your_document.docx"
    tables = extract_docx_tables_between_markers(path)

    for i, df in enumerate(tables, start=1):
        print(f"\n--- TABLE #{i} ---")
        print(df.head())
        print("shape:", df.shape)
