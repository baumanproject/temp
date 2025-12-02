import pdfplumber
import pandas as pd
import logging
from typing import List, Optional, Tuple, Set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def _bbox_inside(inner: Tuple[float, float, float, float],
                 outer: Tuple[float, float, float, float],
                 eps: float = 0.5) -> bool:
    """
    Возвращает True, если bbox `inner` целиком лежит внутри bbox `outer`
    (с маленьким допуском eps на погрешности float).
    bbox = (x0, top, x1, bottom)
    """
    x0i, topi, x1i, bottomi = inner
    x0o, topo, x1o, bottomo = outer
    return (
        x0i >= x0o - eps and
        x1i <= x1o + eps and
        topi >= topo - eps and
        bottomi <= bottomo + eps
    )


def extract_top_level_tables(
    pdf_path: str,
    start_marker: str = "паспорт продукта",
    end_marker: str = "приложение номер 1"
) -> List[pd.DataFrame]:
    """
    Возвращает список DataFrame (без заголовков) только для
    ВНЕШНИХ (top-level) таблиц, лежащих между start_marker и end_marker.

    - Границы по тексту (страницы и вертикальное положение).
    - Вложенные таблицы по bbox выкидываются.
    """
    result_tables: List[pd.DataFrame] = []

    with pdfplumber.open(pdf_path) as pdf:
        in_range = False
        start_y: Optional[float] = None
        page_with_start: Optional[int] = None

        logger.info("Открыт PDF '%s', страниц: %d", pdf_path, len(pdf.pages))

        for page in pdf.pages:
            page_num = page.page_number

            try:
                text = page.extract_text() or ""
            except Exception as e:
                logger.warning("Не удалось извлечь текст на странице %d: %s", page_num, e)
                text = ""
            low = text.lower()

            # --- детектируем старт ---
            if (not in_range) and (start_marker.lower() in low):
                in_range = True
                page_with_start = page_num
                logger.info("Найден start_marker '%s' на странице %d", start_marker, page_num)

                try:
                    found = page.search(start_marker, case=False, regex=False)
                except Exception as e:
                    logger.warning("Ошибка page.search для start_marker на странице %d: %s", page_num, e)
                    found = None

                if found:
                    start_y = found[0]["top"]
                    logger.info(" start_marker top y = %f", start_y)
                else:
                    start_y = None
                    logger.info(" Не удалось определить bbox start_marker (page.search ничего не вернул)")

            if not in_range:
                continue

            # --- ищем таблицы на странице ---
            try:
                tables = page.find_tables()
            except Exception as e:
                logger.warning("Ошибка find_tables на странице %d: %s", page_num, e)
                tables = []

            if not tables:
                logger.debug("Страница %d: таблиц не найдено", page_num)
            else:
                logger.info("Страница %d: найдено таблиц: %d", page_num, len(tables))

            # Собираем (tbl, bbox)
            tbls_meta: List[Tuple[object, Optional[Tuple[float, float, float, float]]]] = []
            for tbl in tables:
                bbox = getattr(tbl, "bbox", None)
                if bbox is None:
                    logger.debug("Таблица без bbox на странице %d, принимаем как top-level по умолчанию", page_num)
                else:
                    logger.debug("Таблица bbox=%s на странице %d", bbox, page_num)
                tbls_meta.append((tbl, bbox))

            # --- вычисляем вложенные таблицы по bbox ---
            nested_indices: Set[int] = set()
            for i, (tbl_i, bbox_i) in enumerate(tbls_meta):
                if bbox_i is None:
                    continue
                for j, (tbl_j, bbox_j) in enumerate(tbls_meta):
                    if i == j or bbox_j is None:
                        continue
                    # если bbox_i лежит внутри bbox_j → i - вложенная таблица
                    if _bbox_inside(bbox_i, bbox_j):
                        nested_indices.add(i)
                        logger.info(
                            "Таблица #%d (bbox=%s) на странице %d помечена как ВЛОЖЕННАЯ внутрь #%d (bbox=%s)",
                            i, bbox_i, page_num, j, bbox_j
                        )
                        break

            top_level_indices = [i for i in range(len(tbls_meta)) if i not in nested_indices]
            logger.info(
                "Страница %d: top-level таблиц: %d (indices=%s)",
                page_num, len(top_level_indices), top_level_indices
            )

            # --- фильтруем по вертикальным границам start/end ---
            # end_marker может быть на этой же странице
            end_y: Optional[float] = None
            if end_marker.lower() in low:
                try:
                    found_end = page.search(end_marker, case=False, regex=False)
                except Exception as e:
                    logger.warning("Ошибка page.search для end_marker на странице %d: %s", page_num, e)
                    found_end = None

                if found_end:
                    end_y = found_end[0]["top"]
                    logger.info("Найден end_marker '%s' на странице %d, top y = %f", end_marker, page_num, end_y)
                else:
                    logger.info("Не удалось определить bbox end_marker (page.search ничего не вернул)")

            for idx in top_level_indices:
                tbl, bbox = tbls_meta[idx]

                # если это страница старта → требуем, чтобы таблица была НИЖЕ start_marker
                if page_num == page_with_start and start_y is not None and bbox is not None:
                    _, _, _, bottom = bbox
                    if bottom < start_y:
                        logger.info(
                            "Страница %d: таблица idx=%d bbox=%s отфильтрована (выше start_marker)",
                            page_num, idx, bbox
                        )
                        continue

                # если на странице есть end_marker → таблица должна быть ВЫШЕ end_marker
                if end_y is not None and bbox is not None:
                    _, top, _, _ = bbox
                    if top > end_y:
                        logger.info(
                            "Страница %d: таблица idx=%d bbox=%s отфильтрована (ниже end_marker)",
                            page_num, idx, bbox
                        )
                        continue

                # --- извлекаем содержимое таблицы ---
                try:
                    rows = tbl.extract()
                except Exception as e:
                    logger.warning(
                        "Ошибка extract() таблицы idx=%d на странице %d: %s", idx, page_num, e
                    )
                    continue

                df = pd.DataFrame(rows)
                logger.info(
                    "Принята таблица idx=%d на странице %d, shape=%s",
                    idx, page_num, df.shape
                )
                result_tables.append(df)

            # если мы увидели end_marker на этой странице — дальше уже не идём
            if end_y is not None:
                in_range = False
                logger.info("Диапазон завершён на странице %d (по end_marker)", page_num)
                break

        logger.info("Всего внешних таблиц между маркерами: %d", len(result_tables))

    return result_tables


if __name__ == "__main__":
    pdf_path = "your.pdf"
    tables = extract_top_level_tables(pdf_path)
    for i, df in enumerate(tables, start=1):
        print(f"\n--- TABLE #{i} ---")
        print(df.head())
        print("shape:", df.shape)
