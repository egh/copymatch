import os
import itertools
import fitz
from typing import Tuple
from copymatch import make_state, match_text, extract_pdf_words, merge_word_rects
import argparse

COLORS = [
    0x7DE198,
    0x81E3E1,
    0x8CFF32,
    0x95C8F3,
    0xABFF32,
    0xAEB5FF,
    0xB3E561,
    0xD4FF32,
    0xDEACF9,
    0xE9FF32,
    0xF3A6C8,
    0xFBAC87,
    0xFDFF32,
    0xFF8C87,
    0xFFDC74,
]


def convert_color(rgb: int) -> Tuple[float, float, float]:
    return (((rgb >> 16) & 255) / 255, ((rgb >> 8) & 255) / 255, (rgb & 255) / 255)


def main():
    parser = argparse.ArgumentParser(description="Find and annotate similar texts")
    parser.add_argument("analysis_text", type=str, help="Text to analyze.")
    parser.add_argument("source_texts", nargs="+", type=str, help="Source texts.")
    args = parser.parse_args()
    original_doc = fitz.open(args.analysis_text)
    words = extract_pdf_words(original_doc)
    state = make_state(words)
    color_no = 0
    for path in args.source_texts:
        if os.path.splitext(path)[-1].lower() != ".pdf":
            continue
        doc = fitz.open(path)
        title = doc.metadata["title"]
        author = doc.metadata["author"]
        for page_no, words in itertools.groupby(
            match_text(state, extract_pdf_words(doc)), lambda word: word.page_no
        ):
            page = original_doc[page_no]
            annot = page.add_highlight_annot(quads=merge_word_rects(words))
            annot.set_colors(stroke=convert_color(COLORS[color_no]))
            annot.set_info(
                title=author,
                subject=f"{title} ({os.path.basename(path)})",
                content=f"{author} / {title} ({os.path.basename(path)})",
            )
            annot.update()
        color_no = (color_no + 1) % len(COLORS)
    original_doc.save("output.pdf")
