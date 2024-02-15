from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections.abc import Container, Iterable
from nltk.tokenize import word_tokenize
import fitz
import unicodedata


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


def normalize_text(word: str):
    return unicodedata.normalize("NFKD", word).casefold()


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
    tokens = [token for token in word_tokenize(text) if token.isalpha()]
    return [Word(token=normalize_text(w), pos=pos) for (pos, w) in enumerate(tokens)]
