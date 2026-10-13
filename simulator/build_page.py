"""Inlines simulator/model.json into the page so it works as one file.
Run after build_model.py:   python simulator/build_page.py"""
from pathlib import Path
model = Path("simulator/model.json").read_text()
page = Path("simulator/template.html").read_text().replace("__MODEL__", model)
Path("simulator/index.html").write_text(page)
print("Saved simulator/index.html")
