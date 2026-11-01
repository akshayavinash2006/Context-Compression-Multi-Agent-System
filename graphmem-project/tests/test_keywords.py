import pytest


@pytest.mark.parametrize("text", ["CNC 3 spindle is loud", "CNC-3 spindle is loud", "cnc3 spindle is loud"])
def test_machine_id_variants(mem, text):
    terms = {t.canonical: t.kind for t in mem.extractor.extract(text)}
    assert terms.get("cnc-3") == "machine"


def test_variants_resolve_to_one_node(mem):
    ids = set()
    for text in ["CNC 3 vibration", "CNC-3 vibration", "cnc3 vibration"]:
        for t in mem.extractor.extract(text):
            if t.kind == "machine":
                ids.add(mem.index.resolve(t.canonical, t.kind)[0])
    assert ids == {"K:cnc-3"}


def test_digit_guard(mem):
    a, _ = mem.index.resolve("cnc-3", "machine")
    b, _ = mem.index.resolve("cnc-5", "machine")
    assert a != b
    # fuzzy and embedding matches must not cross digits either
    assert mem.index.resolve("cnc 5", "machine", create=False)[0] == b


def test_plural_lemma(mem):
    assert mem.extractor.normalize("spindle bearings") == "spindle bearing"
    terms = {t.canonical for t in mem.extractor.extract("Two spindle bearings were replaced on CNC-3.")}
    assert "spindle bearing" in terms


def test_aliases(mem):
    terms = {t.canonical for t in mem.extractor.extract("Why is CNC-3 vibrating?")}
    assert {"cnc-3", "vibration"} <= terms


def test_no_false_machine_ids(mem):
    terms = {t.canonical for t in mem.extractor.extract("vibration hit 7.2 mm/s at 14:05")}
    assert not any(t.startswith(("at-", "hit-")) for t in terms)
