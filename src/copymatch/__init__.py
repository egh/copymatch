import hashlib
import os
import pickle
import sqlite3
import sys
import unicodedata
import zlib
from collections import deque
from collections.abc import Container, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import fitz
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from sqlitedict import SqliteDict

from copymatch.parsr import ParsrClient

PUNCT_TBL = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)


@dataclass(eq=True, frozen=True)
class Word:
    token: str
    pos: int
    ended_in_hyphen: bool


@dataclass(eq=True, frozen=True)
class PDFWord(Word):
    rects: Tuple[fitz.Rect, Optional[fitz.Rect]]
    page_no: int
    block_no: int
    line_no: int
    word_no: int


WORDS = set(brown.words())


@dataclass
class State(Container[str], Iterable[str]):
    transitions: Dict[str, "State"] = field(default_factory=dict)
    startpoints: List[int] = field(default_factory=list)
    end_state: bool = False
    length: int = 0
    rect: Optional[fitz.Rect] = None
    prev_state: Optional["State"] = None
    words: Optional[List[Word]] = None

    def __contains__(self, term: Any):
        return term in self.transitions

    def __iter__(self):
        return self.transitions.__iter__()

    def __getitem__(self, index: str):
        return self.transitions[index]

    def __setitem__(self, index: str, item: "State"):
        self.transitions[index] = item

    def __len__(self):
        return self.length


def cache_file() -> Path:
    return (
        Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        / "copymatch.db"
    )


def parse_page_range(page_range: str) -> Generator[int, str, None]:
    for part in page_range.split(","):
        nums = [int(x) for x in part.split("-")]
        if len(nums) == 1:
            yield nums[0]
        else:
            for num in range(nums[0], nums[1] + 1):
                yield num


def normalize(token: str):
    return unicodedata.normalize("NFKD", token).casefold().translate(PUNCT_TBL)


# Assumption: Original text does not contain repeated bigrams, and if
# so, which is marked as a match is not defined.
def make_state(lst: List[Word], ngram_size=8):
    base = State()
    ptrs: List[Tuple[State, List[Word]]] = [(base, [])]
    for word in lst:
        next_ptrs: List[Tuple[State, List[Word]]] = [(base, [])]
        for ptr, words in ptrs:
            length = 1 + len(ptr)
            end_state = length == ngram_size
            if word.token not in ptr:
                ptr[word.token] = State(length=length, end_state=end_state, words=[])
            words.append(word)
            if end_state:
                ptr[word.token].words.extend(words)
            else:
                next_ptrs.append((ptr[word.token], words))
        ptrs = next_ptrs
    return base


def match_text(base: State, text: List[Word], checker=None):
    next_states = [base]
    retval: List[Word] = []
    # Go through the text
    for idx, word in enumerate(text):
        # In our current states...
        new_next_states = [base]
        for state in next_states:
            # Is this a permissible next term?
            if checker is None:
                if word.token in state:
                    next_state = state[word.token]
                else:
                    next_state = None
            else:
                next_state = checker(word.token, state)
            if next_state is not None:
                if next_state.end_state:
                    retval.extend(next_state.words)
                else:
                    new_next_states.append(next_state)
        next_states = new_next_states
    return sorted(set(retval), key=lambda word: word.pos)


def tokenize(text: str):
    return [
        Word(token=normalize(w), pos=pos, ended_in_hyphen=(w[-1] == "-"))
        for (pos, w) in enumerate(word_tokenize(text))
    ]


def merge_words(a: PDFWord, b: PDFWord):
    return PDFWord(
        token=(a.token + b.token),
        pos=a.pos,
        rects=(a.rects[0], b.rects[0]),
        line_no=a.line_no,
        page_no=a.page_no,
        block_no=a.block_no,
        word_no=a.word_no,
        ended_in_hyphen=False,
    )


def merge_hyphenated(words: List[PDFWord]) -> List[PDFWord]:
    todo = deque(words)
    retval = []
    last = None
    while len(todo) > 0:
        item = todo.popleft()
        if last is not None:
            if (last.token + item.token) in WORDS:
                retval[-1] = merge_words(last, item)
                last = None
                continue
        retval.append(item)
        if item.ended_in_hyphen:
            last = item
        else:
            last = None
    return retval


def cache_encode(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))


def cache_decode(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))


def hash_path(path: str) -> str:
    hash = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(4096):
            hash.update(block)
    return hash.hexdigest()


def parsr(path: str):
    sum = hash_path(path)
    with SqliteDict(
        cache_file(), encode=cache_encode, decode=cache_decode, autocommit=True
    ) as db:
        if sum not in db:
            parsr = ParsrClient("localhost:3001")
            resultid = parsr.send_document(
                file_path=path,
                config_path="defaultConfig.json",
                wait_till_finished=True,
            )["server_response"]
            db[sum] = parsr.get_json(resultid)
        return db[sum]


# https://raw.githubusercontent.com/pd3f/dehyphen/master/dehyphen/scorer.py


# TODO try https://github.com/pd3f/dehyphen/blob/master/dehyphen/format.py
def extract_pdf_words_parsr(path: str) -> List[PDFWord]:
    def word_filter(word):
        if word["type"] != "word":
            return False
        if "isFooter" in word["properties"] and word["properties"]["isFooter"]:
            return False
        return True

    def mk_rect(box):
        return fitz.Rect(box["l"], box["t"], box["l"] + box["w"], box["t"] + box["h"])

    j = parsr(path)

    words = [
        PDFWord(
            token=normalize(word["content"]),
            rects=(mk_rect(word["box"]), None),
            pos=word["properties"]["order"],
            word_no=word["properties"]["order"],
            page_no=page["pageNumber"] - 1,
            line_no=line["properties"]["order"],
            block_no=paragraph["properties"]["order"],
            ended_in_hyphen=(word["content"][-1] == "-"),
        )
        for page in j["pages"]
        for paragraph in page["elements"]
        if paragraph is not None and paragraph["type"] == "paragraph"
        for line in paragraph["content"]
        if line["type"] == "line"
        for word in line["content"]
        if word_filter(word)
    ]
    return merge_hyphenated(words)


def extract_pdf_words(path: str) -> List[PDFWord]:
    doc = fitz.open(path)
    raw_words = [
        (page_no, word)
        for (page_no, page) in enumerate(doc)
        for word in page.get_text("words", sort=True)
    ]
    words = [
        PDFWord(
            token=normalize(word[4]),
            pos=pos,
            rects=(fitz.Rect(word[0:4]), None),
            page_no=page_no,
            block_no=word[5],
            line_no=word[6],
            word_no=word[7],
            ended_in_hyphen=(word[4][-1] == "-"),
        )
        for (pos, (page_no, word)) in enumerate(raw_words)
    ]
    return merge_hyphenated([word for word in words if word.token != ""])


def merge_word_rects(words: List[PDFWord]):
    retval: List[fitz.Rect] = []
    last_word: Optional[PDFWord] = None
    for word in words:
        if len(retval) == 0:
            retval.append(word.rects[0])
            if word.rects[1] is not None:
                retval.append(word.rects[1])
        else:
            if (
                (last_word is not None)
                and (last_word.rects[1] is None)
                and (last_word.block_no == word.block_no)
                and (last_word.line_no == word.line_no)
                and ((last_word.pos + 1) == word.pos)
            ):
                retval[-1].include_rect(word.rects[0])
            else:
                retval.append(word.rects[0])
                if word.rects[1] is not None:
                    retval.append(word.rects[1])
            last_word = word
    return retval
