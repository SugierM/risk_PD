import fitz
from pathlib import Path
from config import DOCS_DIR


def render_page_with_highlight(
    source_file: str,
    page_number: int,
    search_text: str | None = None,
    zoom: float = 2.0,
) -> bytes | None:
    """
    Render a PDF page as PNG, optionally highlighting text.

    :param source_file: Description
    :type source_file: str
    :param page_number: Description
    :type page_number: int
    :param search_text: Description
    :type search_text: str | None
    :param zoom: Description
    :type zoom: float
    :return: Description
    :rtype: bytes | None
    """
    pdf_path = Path(DOCS_DIR) / source_file
    if not pdf_path.exists():
        return None

    doc = fitz.open(str(pdf_path))
    if page_number < 1 or page_number > len(doc):
        doc.close()
        return None

    page = doc[page_number - 1]

    
    if search_text:
        query = search_text.strip()[:80]
        instances = page.search_for(query)
        for inst in instances:
            annot = page.add_highlight_annot(inst)
            annot.set_colors(stroke=(1, 0.9, 0))  # yellow
            annot.update()

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes


def get_page_count(source_file: str) -> int:
    """
    Docstring for get_page_count
    
    :param source_file: Description
    :type source_file: str
    :return: Description
    :rtype: int
    """
    pdf_path = Path(DOCS_DIR) / source_file
    if not pdf_path.exists():
        return 0
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count