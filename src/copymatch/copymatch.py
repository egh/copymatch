import os
import itertools
import string
import fitz
from typing import Optional, List, Tuple
from copymatch import make_state, match_text, PDFWord, normalize, filter_token
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


def extract_words(path) -> List[PDFWord]:
    raw_words = [
        (page_no, word)
        for (page_no, page) in enumerate(fitz.open(path))
        for word in page.get_text("words", sort=True, delimiters=string.punctuation)
        if filter_token(word[4])
    ]

    return [
        PDFWord(
            token=normalize(word[4]),
            pos=pos,
            rect=fitz.Rect(word[0:4]),
            page_no=page_no,
            block_no=word[5],
            line_no=word[6],
            word_no=word[7],
        )
        for (pos, (page_no, word)) in enumerate(raw_words)
    ]


def merge_word_rects(words: List[PDFWord]):
    retval: List[fitz.Rect] = []
    last_word: Optional[PDFWord] = None
    for word in words:
        if len(retval) == 0:
            retval.append(word.rect)
        else:
            if (
                (last_word is not None)
                and (last_word.block_no == word.block_no)
                and (last_word.line_no == word.line_no)
                and ((last_word.pos + 1) == word.pos)
            ):
                retval[-1].include_rect(word.rect)
            else:
                retval.append(word.rect)
            last_word = word
    return retval


def main():
    parser = argparse.ArgumentParser(description="Find and annotate similar texts")
    parser.add_argument("analysis_text", type=str, help="Text to analyze.")
    parser.add_argument("source_texts", nargs="+", type=str, help="Source texts.")
    args = parser.parse_args()
    words = extract_words(args.analysis_text)
    state = make_state(words)
    original_doc = fitz.open(args.analysis_text)
    color_no = 0
    for path in args.source_texts:
        doc = fitz.open(path)
        title = doc.metadata["title"]
        author = doc.metadata["author"]
        for page_no, words in itertools.groupby(
            match_text(state, extract_words(path)), lambda word: word.page_no
        ):
            page = original_doc[page_no]
            annot = page.add_highlight_annot(quads=merge_word_rects(words))
            annot.set_colors(stroke=convert_color(COLORS[color_no]))
            annot.set_info(title=f"{author}, {title} ({os.path.basename(path)})")
            annot.update()
        color_no = (color_no + 1) % len(COLORS)
    original_doc.save("output.pdf")
