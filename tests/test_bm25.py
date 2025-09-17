from vifact.modules.retrieval import BM25Index


def test_bm25_simple():
    docs = [
        ("d1", "Hà Nội là thủ đô của Việt Nam"),
        ("d2", "Thành phố Hồ Chí Minh là đô thị lớn"),
        ("d3", "Đà Nẵng là thành phố biển"),
    ]
    idx = BM25Index()
    idx.fit(docs)
    res = idx.search("thủ đô Việt Nam", top_k=2)
    assert res and res[0][0] in {"d1"}

