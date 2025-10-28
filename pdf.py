from __future__ import annotations
import logging
from pathlib import Path

import pandas as pd
import pdfplumber
import tabula
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("passport_simple")

class ExtractConfig(BaseModel):
    pdf_path: str
    header_text: str = "ПАСПОРТ ПРОДУКТА"
    max_follow_pages: int = 6
    header_margin: float = 6.0
    side_margins: tuple = Field(default=(12.0, 12.0))
    bottom_margin: float = 12.0
    use_lattice: bool = True
    value_joiner: str = " ⏐ "

class PassportSimpleExtractor:
    def __init__(self, cfg: ExtractConfig):
        self.cfg = cfg

    def _find_header_bbox(self, page) -> tuple | None:
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
        if not words:
            return None
        target = self.cfg.header_text.strip().lower()
        for i in range(len(words)):
            acc = []
            x0 = words[i]["x0"]; top = words[i]["top"]
            x1 = words[i]["x1"]; bottom = words[i]["bottom"]
            for j in range(i, len(words)):
                acc.append(words[j]["text"])
                x0 = min(x0, words[j]["x0"])
                top = min(top, words[j]["top"])
                x1 = max(x1, words[j]["x1"])
                bottom = max(bottom, words[j]["bottom"])
                joined = " ".join(acc).replace(" "," ").strip().lower()
                if joined == target:
                    return (x0, top, x1, bottom)
                if words[j]["top"] - bottom > 2.0 and not target.startswith(joined):
                    break
        return None

    def _area_below_header(self, page) -> list | None:
        bbox = self._find_header_bbox(page)
        if not bbox:
            return None
        x0, top, x1, bottom = bbox
        left_m, right_m = self.cfg.side_margins
        top_area = bottom + self.cfg.header_margin
        area = [
            max(0.0, top_area),
            max(0.0, left_m),
            page.height - self.cfg.bottom_margin,
            page.width - right_m,
        ]
        if area[2] - area[0] < 20 or area[3] - area[1] < 20:
            return None
        return area

    def _read_area(self, pdf_path: str, page_num: int, area: list) -> list[pd.DataFrame]:
        opts = {
            "pages": page_num,
            "area": [area],
            "multiple_tables": True,
            "guess": False,
            "pandas_options": {"dtype": str},
        }
        if self.cfg.use_lattice:
            opts["lattice"] = True
        else:
            opts["stream"] = True
        return tabula.read_pdf(pdf_path, **opts)

    def extract(self) -> pd.DataFrame:
        pdf_path = str(Path(self.cfg.pdf_path).resolve())
        dfs = []
        with pdfplumber.open(pdf_path) as pdf:
            start_page = None
            area_ref = None
            for idx, page in enumerate(pdf.pages, start=1):
                area = self._area_below_header(page)
                if area:
                    start_page = idx
                    area_ref = area
                    log.info(f"Found header on page {idx}, area: {area_ref}")
                    break
        if start_page is None or area_ref is None:
            raise RuntimeError(f"Header '{self.cfg.header_text}' not found")

        for p in range(start_page, start_page + self.cfg.max_follow_pages):
            dfs_page = self._read_area(pdf_path, p, area_ref)
            if not dfs_page:
                continue
            df_page = pd.concat(dfs_page, ignore_index=True)
            if df_page.dropna(how="all").shape[0] == 0:
                continue
            dfs.append(df_page)

        if not dfs:
            raise RuntimeError("No table extracted after header")

        raw_df = pd.concat(dfs, ignore_index=True).astype("string")
        return self._two_column_norm(raw_df)

    def _two_column_norm(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(axis=1, how="all")
        if df.shape[1] > 2:
            left = df.iloc[:, 0]
            right = df.iloc[:, 1:].apply(lambda r: " ".join([x for x in r if pd.notna(x)]).strip(), axis=1)
            df2 = pd.DataFrame({"Ключ": left.str.strip(), "Значение": right.str.strip()})
        elif df.shape[1] == 2:
            df2 = df.copy()
            df2.columns = ["Ключ", "Значение"]
            df2["Ключ"] = df2["Ключ"].str.strip()
            df2["Значение"] = df2["Значение"].str.strip()
        else:
            df.columns = ["raw"]
            split = df["raw"].str.split(":", n=1, expand=True)
            if split.shape[1] == 2:
                df2 = pd.DataFrame({"Ключ": split[0].str.strip(), "Значение": split[1].str.strip()})
            else:
                df2 = pd.DataFrame({"Ключ": df["raw"].str.strip(), "Значение": pd.NA})

        # Склейка продолжений: если Ключ пустой → объединяем Значение с предыдущим
        rows = []
        for _, row in df2.iterrows():
            key = row["Ключ"] if pd.notna(row["Ключ"]) else ""
            val = row["Значение"] if pd.notna(row["Значение"]) else ""
            if key:
                rows.append([key, val])
            else:
                if rows:
                    rows[-1][1] = f"{rows[-1][1]}{self.cfg.value_joiner}{val}"
                else:
                    rows.append([None, val])
        out = pd.DataFrame(rows, columns=["Ключ", "Значение"])
        return out.dropna(how="all").reset_index(drop=True)

if __name__ == "__main__":
    cfg = ExtractConfig(
        pdf_path="path/to/your.pdf",
        header_text="ПАСПОРТ ПРОДУКТА",
        use_lattice=True,
        value_joiner=" ⏐ ",
    )
    df = PassportSimpleExtractor(cfg).extract()
    print(df)
    # df.to_csv("passport.csv", index=False)
