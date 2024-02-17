import sys
import string
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections.abc import Container, Iterable
from nltk.tokenize import word_tokenize
import fitz
import unicodedata

PUNCT_TBL = dict.fromkeys(
    i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith("P")
)


@dataclass(eq=True, frozen=True)
class Word:
    token: str
    pos: int


@dataclass(eq=True, frozen=True)
class PDFWord(Word):
    rect: fitz.Rect
    page_no: int
    block_no: int
    line_no: int
    word_no: int


@dataclass
class State(Container[str], Iterable[str]):
    next_tokens: Dict[str, "State"] = field(default_factory=dict)
    startpoints: List[int] = field(default_factory=list)
    end_state: bool = False
    length: int = 0
    rect: Optional[fitz.Rect] = None
    prev_state: Optional["State"] = None
    words: Optional[List[Word]] = None

    def __contains__(self, term: Any):
        return term in self.next_tokens

    def __iter__(self):
        return self.next_tokens.__iter__()

    def __getitem__(self, index: str):
        return self.next_tokens[index]

    def __setitem__(self, index: str, item: "State"):
        self.next_tokens[index] = item

    def __len__(self):
        return self.length


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


def match_text(base: State, text: List[Word]):
    next_states = [base]
    retval: List[Word] = []
    # Go through the text
    for idx, word in enumerate(text):
        # In our current states...
        new_next_states = [base]
        for state in next_states:
            # Is this a permissible next term?
            if word.token in state:
                next_state = state[word.token]
                if next_state.end_state:
                    retval.extend(next_state.words)
                else:
                    new_next_states.append(state[word.token])
        next_states = new_next_states
    return sorted(set(retval), key=lambda word: word.pos)


def tokenize(text: str):
    tokens = [token for token in word_tokenize(text)]
    return [Word(token=normalize(w), pos=pos) for (pos, w) in enumerate(tokens)]


def extract_pdf_words(doc: fitz.Document) -> List[PDFWord]:
    raw_words = [
        (page_no, word)
        for (page_no, page) in enumerate(doc)
        for word in page.get_text("words", sort=True, delimiters=string.punctuation)
    ]
    words = [
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
    return [word for word in words if word.token != ""]


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
