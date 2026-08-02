from pydantic import ValidationError

from ..schemas.strategy_spec import StrategySpec


def validate_strategy_spec(value: StrategySpec | dict) -> tuple[bool, list[str]]:
    try:
        StrategySpec.model_validate(value)
    except ValidationError as exc:
        return False, [error["msg"] for error in exc.errors()]
    return True, []

