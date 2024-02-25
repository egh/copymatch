import os
import itertools
import fitz
from typing import Tuple
from copymatch import (
    extract_pdf_words,
    extract_pdf_words_parsr,
    make_state,
    match_text,
    merge_word_rects,
)
import argparse
import Levenshtein

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


def mk_checker(distance):
    def checker(token, state):
        if token in state:
            return state[token]
        else:
            for to_test in state:
                if (
                    Levenshtein.distance(token, to_test, score_cutoff=distance)
                    <= distance
                ):
                    return state[to_test]
        return None

    return checker


def main():
    parser = argparse.ArgumentParser(description="Find and annotate similar texts")
    parser.add_argument("analysis_text", type=str, help="Text to analyze.")
    parser.add_argument("source_texts", nargs="+", type=str, help="Source texts.")
    parser.add_argument(
        "-d",
        "--distance",
        type=int,
        default=0,
        help="Maximum edit distance between word that is considered a match (default is an exact match).",
    )
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        default=8,
        help="Minimum number of required tokens matched to mark text (default is 8)",
    )
    parser.add_argument(
        "-p",
        "--parsr",
        action="store_true",
        help="Use parsr server for processing PDFs.",
    )

    args = parser.parse_args()
    if args.parsr:
        extract_words_func = extract_pdf_words_parsr
    else:
        extract_words_func = extract_pdf_words
    original_doc = fitz.open(args.analysis_text)
    words = extract_words_func(args.analysis_text)
    state = make_state(words, ngram_size=args.length)
    color_no = 0
    for path in args.source_texts:
        if os.path.splitext(path)[-1].lower() != ".pdf":
            continue
        doc = fitz.open(path)
        title = doc.metadata["title"]
        author = doc.metadata["author"]
        if args.distance == 0:
            checker = None
        else:
            checker = mk_checker(args.distance)
        for page_no, words in itertools.groupby(
            match_text(state, extract_words_func(path), checker=checker),
            lambda word: word.page_no,
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
