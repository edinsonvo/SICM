"""Generación de reportes PDF del SICM v2.0.

Usa ReportLab para producir un informe profesional con título, resumen
ejecutivo, análisis del modelo, resultados de simulación y conclusiones.
"""

import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable)

# Paleta corporativa
PRIMARY = colors.HexColor("#1f3a5f")
ACCENT = colors.HexColor("#2ca02c")
LIGHT = colors.HexColor("#eef2f7")


def _build_styles():
    """Estilos de párrafo y tabla para el documento."""
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "TitleSICM", parent=base["Title"], textColor=PRIMARY,
            fontSize=20, spaceAfter=4),
        "subtitle": ParagraphStyle(
            "SubSICM", parent=base["BodyText"], textColor=colors.HexColor("#5a6b82"),
            fontSize=11, leading=15, spaceAfter=10),
        "h2": ParagraphStyle(
            "H2SICM", parent=base["Heading2"], textColor=PRIMARY,
            fontSize=13, spaceBefore=14, spaceAfter=6),
        "body": ParagraphStyle(
            "BodySICM", parent=base["BodyText"], fontSize=10,
            leading=15, spaceAfter=8),
        "small": ParagraphStyle(
            "SmallSICM", parent=base["BodyText"], fontSize=8.5,
            textColor=colors.HexColor("#555555")),
    }
    return styles


def _section_title(styles, text):
    """Devuelve el flujo del título de sección con una línea separadora."""
    return [Paragraph(text, styles["h2"]),
            HRFlowable(width="100%", thickness=0.8,
                       color=colors.HexColor("#c7d3e0"))]


def _param_table(params, styles):
    """Tabla de parámetros del modelo."""
    rows = [["Parámetro", "Valor"]]
    for key, value in (params or {}).items():
        if isinstance(value, float):
            rows.append([key, f"{value:.4f}"])
        else:
            rows.append([key, str(value)])
    return _styled_table(rows, styles, widths=[90 * mm, 60 * mm])


def _styled_table(rows, styles, widths=None):
    """Crea una tabla con estilo profesional."""
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7d3e0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _eq_table(before, after, deltas, styles, model_name):
    """Tabla comparativa de equilibrios antes/después del choque."""
    if model_name == "IS-LM":
        rows = [["Variable", "Antes", "Después", "Cambio"],
                ["Producción (Y)", f"{before.get('Y', 0):.2f}",
                 f"{after.get('Y', 0):.2f}", f"{deltas.get('Y', 0):+.2f}"],
                ["Tasa de interés (r)", f"{before.get('r', 0) * 100:.2f}%",
                 f"{after.get('r', 0) * 100:.2f}%",
                 f"{deltas.get('r', 0) * 100:+.2f} p.p."],
                ["Consumo (C)", f"{before.get('C', 0):.2f}",
                 f"{after.get('C', 0):.2f}", f"{deltas.get('C', 0):+.2f}"],
                ["Inversión (I)", f"{before.get('I', 0):.2f}",
                 f"{after.get('I', 0):.2f}", f"{deltas.get('I', 0):+.2f}"],
                ["Gasto público (G)", f"{before.get('G', 0):.2f}",
                 f"{after.get('G', 0):.2f}", f"{deltas.get('G', 0):+.2f}"]]
    elif model_name == "Mundell-Fleming":
        rows = [["Variable", "Antes", "Después", "Cambio"],
                ["Producción (Y)", f"{before.get('Y', 0):.2f}",
                 f"{after.get('Y', 0):.2f}", f"{deltas.get('Y', 0):+.2f}"],
                ["Tipo de cambio (e)", f"{before.get('e', 0):.2f}",
                 f"{after.get('e', 0):.2f}", f"{deltas.get('e', 0):+.2f}"],
                ["Oferta monetaria (M)", f"{before.get('M', 0):.2f}",
                 f"{after.get('M', 0):.2f}", f"{deltas.get('M', 0):+.2f}"],
                ["Exportaciones netas (NX)", f"{before.get('NX', 0):.2f}",
                 f"{after.get('NX', 0):.2f}", f"{deltas.get('NX', 0):+.2f}"]]
    else:
        rows = [["Variable", "Antes", "Después", "Cambio"],
                ["Producción (Y)", f"{before.get('Y', 0):.2f}",
                 f"{after.get('Y', 0):.2f}", f"{deltas.get('Y', 0):+.2f}"],
                ["Nivel de precios (P)", f"{before.get('P', 0):.2f}",
                 f"{after.get('P', 0):.2f}", f"{deltas.get('P', 0):+.2f}"],
                ["Brecha (% del PIB)", f"{before.get('gap', 0):.2f}",
                 f"{after.get('gap', 0):.2f}", f"{deltas.get('gap', 0):+.2f}"]]
    return _styled_table(rows, styles, widths=[50 * mm, 40 * mm, 40 * mm, 40 * mm])


def generate_pdf_report(report_data, output_path):
    """Genera el reporte PDF a partir de un diccionario de resultados.

    Parámetros
    ----------
    report_data : dict
        Con claves: title, subtitle, date, model, params_before,
        equilibrium_before, shock_label, magnitude, equilibrium_after,
        deltas, mechanism, policy_type, executive_summary, conclusions.
    output_path : str
        Ruta del archivo PDF de salida.

    Devuelve
    --------
    str
        Ruta del PDF generado.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    styles = _build_styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title=report_data.get("title", "Reporte SICM v2.0"),
        author="SICM v2.0")

    flow = []
    # --- Encabezado ----------------------------------------------------
    flow.append(Paragraph("Simulador Integral de Choques Macroeconómicos",
                          styles["small"]))
    flow.append(Spacer(1, 2))
    flow.append(Paragraph(report_data.get("title", "Reporte de simulación"),
                          styles["title"]))
    flow.append(Paragraph(report_data.get("subtitle", ""), styles["subtitle"]))
    flow.append(Paragraph(
        f"Fecha de generación: {report_data.get('date', datetime.now().strftime('%d/%m/%Y %H:%M'))}",
        styles["small"]))
    flow.append(Spacer(1, 6))

    # --- Resumen ejecutivo ---------------------------------------------
    flow += _section_title(styles, "1. Resumen ejecutivo")
    flow.append(Paragraph(
        report_data.get("executive_summary") or
        "Simulación de un choque macroeconómico sobre un modelo "
        "computacional de equilibrio. El documento resume los parámetros, "
        "el equilibrio previo, el choque aplicado, el equilibrio posterior "
        "y las conclusiones analíticas.",
        styles["body"]))

    # --- Análisis del modelo -------------------------------------------
    flow += _section_title(styles, "2. Análisis del modelo")
    flow.append(Paragraph(
        f"Modelo base: <b>{report_data.get('model', '—')}</b>. "
        f"Parámetros utilizados:",
        styles["body"]))
    flow.append(_param_table(report_data.get("params_before", {}), styles))

    # --- Simulación -----------------------------------------------------
    flow += _section_title(styles, "3. Resultados de la simulación")
    flow.append(Paragraph(
        f"Choque aplicado: <b>{report_data.get('shock_label', '—')}</b> "
        f"con magnitud de {report_data.get('magnitude', 0) * 100:.0f} %. "
        f"Política asociada: <b>{report_data.get('policy_type', '—')}</b>.",
        styles["body"]))
    flow.append(_eq_table(
        report_data.get("equilibrium_before", {}),
        report_data.get("equilibrium_after", {}),
        report_data.get("deltas", {}),
        styles, report_data.get("model", "IS-LM")))

    flow.append(Spacer(1, 6))
    flow.append(Paragraph("Mecanismo de transmisión:", styles["body"]))
    flow.append(Paragraph(
        report_data.get("mechanism") or "Sin descripción disponible.",
        styles["small"]))

    # --- Conclusiones ---------------------------------------------------
    flow += _section_title(styles, "4. Conclusiones")
    flow.append(Paragraph(
        report_data.get("conclusions") or
        "Se recomienda contrastar los resultados con series reales y "
        "analizar la robustez del equilibrio ante cambios de parámetros.",
        styles["body"]))

    flow.append(Spacer(1, 12))
    flow.append(HRFlowable(width="100%", thickness=0.6,
                           color=colors.HexColor("#c7d3e0")))
    flow.append(Paragraph(
        "Reporte generado automáticamente por el Simulador Integral de "
        "Choques Macroeconómicos (SICM) v2.0.",
        styles["small"]))

    doc.build(flow)
    return output_path
