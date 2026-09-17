# api/pdf_export.py

import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _stringify_cell(value: Any) -> str:
    """Convierte cualquier valor (incluyendo dict/list anidados) a texto para la tabla."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def generate_pdf_report(
    title: str,
    data: List[Dict[str, Any]],
    subtitle: Optional[str] = None,
    columns: Optional[List[str]] = None
) -> bytes:
    """
    Genera un PDF tabular a partir de una lista de registros (dicts).

    Args:
        title: Título del reporte.
        data: Lista de registros a exportar (p. ej. el campo `data` de un APIResponse).
        subtitle: Texto opcional bajo el título (contexto de la predicción).
        columns: Orden/subconjunto de columnas a incluir. Por defecto usa las
            llaves del primer registro.

    Returns:
        Bytes del PDF generado.
    """
    if not data:
        raise ValueError("No hay datos para exportar")

    if columns is None:
        columns = list(data[0].keys())

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(title, styles["Title"]))
    if subtitle:
        elements.append(Paragraph(subtitle, styles["Normal"]))
    elements.append(Paragraph(
        f"Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 0.5 * cm))

    header_style = ParagraphStyle(
        "header", fontSize=8, leading=10, textColor=colors.white, fontName="Helvetica-Bold"
    )
    cell_style = ParagraphStyle("cell", fontSize=7, leading=9)

    table_data = [[Paragraph(str(col), header_style) for col in columns]]
    for row in data:
        table_data.append([Paragraph(_stringify_cell(row.get(col)), cell_style) for col in columns])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    elements.append(table)
    doc.build(elements)

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
