from __future__ import annotations
import logging
import pandas as pd
from pydantic import BaseModel, Field
import tabula  # убедитесь, что установлена tabula-py

logger = logging.getLogger(__name__)

class LoaderConfig(BaseModel):
    pdf_path: str
    header_marker: str = "ПАСПОРТ ПРОДУКТА"
    pages_after: int = 10  # сколько страниц за заголовком максимум
    tabula_options: dict = Field(default_factory=lambda: {
        "pages": "all", 
        "multiple_tables": True,
        # возможно lattice/stream варианты:
        # "lattice": True
    })

class PDFTableExtractor:
    def __init__(self, cfg: LoaderConfig):
        self.cfg = cfg

    def extract(self) -> pd.DataFrame:
        # найдём страницу с заголовком
        import PyPDF2
        reader = PyPDF2.PdfReader(self.cfg.pdf_path)
        start_page: int | None = None
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if self.cfg.header_marker in text:
                start_page = idx
                logger.info(f"Found header marker on page {idx}")
                break
        if start_page is None:
            raise ValueError(f"Header marker '{self.cfg.header_marker}' not found")

        # зададим диапазон страниц для извлечения таблиц
        end_page = start_page + self.cfg.pages_after - 1
        pages_arg = f"{start_page}-{end_page}"
        options = dict(self.cfg.tabula_options)
        options["pages"] = pages_arg
        logger.info(f"Extracting tables from pages {pages_arg}")

        dfs: list[pd.DataFrame] = tabula.read_pdf(self.cfg.pdf_path, **options)  # возвращает list
        if not dfs:
            raise RuntimeError("No tables found by tabula")
        logger.info(f"tabula extracted {len(dfs)} tables")
        # объединим все
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined DataFrame shape: {combined.shape}")
        return combined

class PostProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def process(self) -> pd.DataFrame:
        # допустим: если первая строка — заголовки
        self.df.columns = self.df.iloc[0]
        df2 = self.df.drop(index=0).reset_index(drop=True)
        # если две колонки — переименуем
        if len(df2.columns) == 2:
            df2.columns = ["Key", "Value"]
        # избавимся от строк полностью NaN
        df3 = df2.dropna(how="all").reset_index(drop=True)
        return df3

# Пример запуска
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = LoaderConfig(pdf_path="path/to/your.pdf")
    extractor = PDFTableExtractor(cfg)
    raw_df = extractor.extract()
    processor = PostProcessor(raw_df)
    final_df = processor.process()
    print(final_df.head())
