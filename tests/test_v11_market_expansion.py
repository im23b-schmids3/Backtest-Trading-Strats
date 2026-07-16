from fib_backtester.research.v11_market_expansion import _catalog, _portfolio_sets


def test_catalog_contains_required_alpha_markets_and_source_metadata():
    rows = {row["symbol"]: row for row in _catalog()}
    for symbol in ("MET", "MNQ", "MES", "MCL", "MGC", "SIL", "M6E", "ZN"):
        assert rows[symbol]["official_alpha_supported"] is True
        assert rows[symbol]["official_source"].endswith("alpha-futures.com/assets")


def test_portfolio_sets_are_progressive_and_keep_baseline():
    ranking = __import__("pandas").DataFrame({"market": ["MNQ", "MGC", "M6E", "ZN"], "admitted": [True, True, True, True], "official_alpha_supported": [True] * 4, "robustness_score": [4, 3, 2, 1]})
    portfolios = _portfolio_sets(["MNQ", "MGC"], ranking)
    assert portfolios["Portfolio A - current baseline"]
    assert len(portfolios["Portfolio C - top 3 admitted"]) >= len(portfolios["Portfolio A - current baseline"])
    assert len(portfolios["Portfolio E - all admitted"]) >= len(portfolios["Portfolio D - top 5 admitted"])
