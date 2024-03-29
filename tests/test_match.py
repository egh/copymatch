from copymatch import (
    State,
    make_state,
    match_text,
    normalize,
    parse_page_range,
    tokenize,
)


def test_match():
    base = make_state(tokenize("hello world and goodbye"), 2)
    results = match_text(base, tokenize("this is the world and goodbye"))
    assert [result.pos for result in results] == [1, 2, 3]


def test_make_state():
    fsa = make_state(tokenize("hello world and goodbye"), 2)
    assert set(fsa.transitions.keys()) == {"hello", "world", "and", "goodbye"}
    assert fsa["goodbye"].transitions == {}
    assert set(fsa["hello"].transitions.keys()) == {"world"}
    assert fsa["hello"]["world"].end_state == True
    assert fsa["hello"]["world"].words[0].token == "hello"


def test_make_state_large():
    t = tokenize(
        """Lorem ipsum dolor sit amet, consectetur adipiscing
    elit. Donec eu ornare turpis, elementum finibus arcu. Sed leo
    neque, facilisis ac ipsum at, congue dignissim elit. Integer nec
    erat accumsan, tristique dolor venenatis, vestibulum risus.
    Phasellus ornare purus non nisl interdum, eget dictum leo lacinia.
    Praesent ac libero orci. Sed dictum nulla at ante porttitor
    vehicula. Curabitur tempus, nunc ac consectetur sodales, elit ex
    mollis velit, eu auctor elit ipsum pharetra orci. Praesent et
    eleifend eros. Donec elementum vitae tortor eget aliquam. Duis
    accumsan viverra volutpat. Pellentesque ac ligula finibus,
    elementum ipsum a, tempor massa. Quisque maximus orci et vehicula
    rutrum. Donec consectetur dui nec libero egestas, ut mattis magna
    porttitor. Maecenas porta erat sapien, ut hendrerit eros sagittis
    ultrices. In felis metus, placerat nec ipsum vel, laoreet pulvinar
    ipsum. Praesent nec velit eu ex condimentum bibendum."""
    )
    base = make_state(t, ngram_size=6)
    assert len(base.transitions) == 93
    matches = match_text(
        base, tokenize("lorem rutrum donec consectetur dui nec libero egestas")
    )
    assert matches[0].pos == 125


def test_normalize():
    assert normalize("fulﬁll") == "fulfill"
    assert normalize("FULFILL") == "fulfill"
    assert normalize("Straße") == "strasse"
    assert normalize("Congreſs") == "congress"
    assert normalize("hello.") == "hello"
    assert normalize(".&-") == ""
    assert normalize("‘hello’") == "hello"


def test_page_range():
    assert list(parse_page_range("1-2")) == [1, 2]
    assert list(parse_page_range("1,2")) == [1, 2]
    assert list(parse_page_range("1,2-5,8")) == [1, 2, 3, 4, 5, 8]
