import re
from dataclasses import dataclass
from typing import Iterable

from rapidfuzz import fuzz, process


# -------------------------
# Настройки/результаты
# -------------------------

@dataclass(frozen=True)
class MatchInfo:
    attribute: str
    matched_key: str | None
    score: float
    line_index: int | None
    value: str | None


@dataclass(frozen=True)
class Candidate:
    line_index: int
    raw_line: str
    key: str
    value: str
    key_norm: str


# -------------------------
# Нормализация строк
# -------------------------

_WS_RE = re.compile(r"\s+")
_JUNK_RE = re.compile(r"[^0-9a-zA-Zа-яА-ЯёЁ%№\s]+")  # оставляем буквы/цифры/пробел/несколько символов

def normalize_key(s: str) -> str:
    """
    Нормализация только для сравнения (fuzzy).
    Сами атрибуты и ключи НЕ меняем — возвращаем оригиналы, а сравниваем normalized.
    """
    t = s.strip().casefold()
    t = t.replace("ё", "е")
    t = _JUNK_RE.sub(" ", t)          # пунктуацию -> пробел
    t = _WS_RE.sub(" ", t).strip()    # схлопываем пробелы
    return t


# -------------------------
# Детектор "табличной" строки и разбор ячеек
# -------------------------

# Типовая строка-разделитель markdown таблицы: | --- | ---: | :--- |
# (может быть с пробелами). Мы такие пропускаем.
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*[:\- ]+\s*(\|\s*[:\- ]+\s*)+\|?\s*$")

def is_candidate_table_line(line: str) -> bool:
    if "|" not in line:
        return False
    # слишком мало пайпов — скорее не таблица
    if line.count("|") < 2:
        return False
    # разделитель/шапка таблицы
    if _TABLE_SEPARATOR_RE.match(line):
        return False
    return True


def split_pipe_cells(line: str) -> list[str]:
    """
    Толерантный сплит по '|'.
    Убираем внешние '|' если есть, но НЕ требуем идеальной таблицы.
    """
    s = line.strip()
    # убрать крайние пайпы (часто они есть, но не всегда)
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    cells = [c.strip() for c in s.split("|")]
    return cells


def extract_key_value_from_cells(cells: list[str]) -> tuple[str, str] | None:
    """
    Делает (key, value) из набора ячеек с разным количеством колонок.
    Правило: берём ПЕРВУЮ непустую ячейку как key, всё что справа — value.
    """
    # найти индекс первого непустого
    key_idx = None
    for i, c in enumerate(cells):
        if c:
            key_idx = i
            break
    if key_idx is None:
        return None

    key = cells[key_idx].strip()
    # value = всё справа от key_idx, включая пустые, но лучше убрать крайние пустоты
    right = [c.strip() for c in cells[key_idx + 1 :]]
    # если справа всё пустое, пробуем распознать формат "Ключ: значение" в самом key
    if not any(right):
        if ":" in key:
            k, v = key.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k:
                return (k, v)
        return None

    # собираем value
    value = " | ".join([c for c in right if c != ""]).strip()
    if not value:
        return None

    # если key содержит двоеточие, иногда это "Ключ: ..." и справа тоже что-то — оставим как есть
    return (key, value)


def extract_candidates_from_markdown(markdown_text: str) -> list[Candidate]:
    """
    1) Идём построчно.
    2) Игнорируем code blocks ```...```.
    3) Берём все строки с '|' (кроме разделителей).
    4) Для каждой строки пытаемся извлечь (key,value).
    """
    candidates: list[Candidate] = []
    in_code_block = False

    for idx, raw_line in enumerate(markdown_text.splitlines()):
        line = raw_line.rstrip("\n")

        # toggling code fences
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        if not is_candidate_table_line(line):
            continue

        cells = split_pipe_cells(line)
        if len(cells) < 2:
            continue

        kv = extract_key_value_from_cells(cells)
        if kv is None:
            continue
        key, value = kv

        candidates.append(
            Candidate(
                line_index=idx,
                raw_line=line,
                key=key,
                value=value,
                key_norm=normalize_key(key),
            )
        )

    return candidates


# -------------------------
# Матчинг атрибутов по кандидатам
# -------------------------

def match_attributes_from_markdown(
    markdown_text: str,
    attributes: list[str],
    threshold: float = 85.0,
) -> tuple[dict[str, str | None], float, list[MatchInfo]]:
    """
    Возвращает:
      1) mapping: dict[attr -> value|None]
      2) recall: найдено / всего
      3) details: список MatchInfo (для отладки/подбора threshold)
    """

    candidates = extract_candidates_from_markdown(markdown_text)

    # если кандидатов нет — всё None
    if not candidates:
        mapping = {a: None for a in attributes}
        details = [MatchInfo(a, None, 0.0, None, None) for a in attributes]
        return mapping, 0.0, details

    # список normalized ключей (для RapidFuzz)
    keys_norm = [c.key_norm for c in candidates]

    mapping: dict[str, str | None] = {}
    details: list[MatchInfo] = []
    found = 0

    for attr in attributes:
        attr_norm = normalize_key(attr)

        # extractOne вернёт лучший матч (match, score, index)  [oai_citation:2‡GitHub](https://github.com/rapidfuzz/RapidFuzz?utm_source=chatgpt.com)
        best = process.extractOne(
            attr_norm,
            keys_norm,
            scorer=fuzz.token_set_ratio,   # устойчиво к перестановкам/лишним словам  [oai_citation:3‡datacamp.com](https://www.datacamp.com/tutorial/fuzzy-string-python?utm_source=chatgpt.com)
            score_cutoff=threshold,
        )

        if best is None:
            mapping[attr] = None
            details.append(MatchInfo(attr, None, 0.0, None, None))
            continue

        matched_norm, score, cand_idx = best
        cand = candidates[cand_idx]

        mapping[attr] = cand.value
        details.append(MatchInfo(attr, cand.key, float(score), cand.line_index, cand.value))
        found += 1

    recall = found / max(1, len(attributes))
    return mapping, recall, details
