# Plateau selection

Selection scores combine expectancy, capped profit factor, drawdown and fee
penalties, and trade support. The reviewer prefers a neighboring stable region
to a single maximum. An isolated maximum with strong expectancy is vetoed and
cannot be frozen. A round with no stable region stops with insufficient
evidence. The tolerances are configured in `research_defaults.yaml`.
