from .roll import _Number, Expression, DiceRollError
import typing

class _Percentage(float):
    def __repr__(self) -> str:
        return f"{_Number(self*100)}%"


class FnOp(Expression):
    @classmethod
    def name(cls) -> str:
        raise NotImplementedError

    @classmethod
    def description(cls) -> str:
        return ""

    @classmethod
    def help(cls) -> str:
        return "No help text available for this function."

    def __init__(self, arg: Expression):
        self.arg = arg

    def __repr__(self):
        return "%s %s" % (self.name(), self.arg)


class ProbabilityOf(FnOp):
    def __init__(self, arg: Expression):
        self.arg = arg

    def roll(self):
        return _Percentage(self.arg.probability())

    @classmethod
    def name(cls):
        return "p"

    @classmethod
    def description(cls) -> str:
        return "probability of an event"

    @classmethod
    def help(cls) -> str:
        return """p <event>

Arguments:
    event - A boolean expression.

Result:
    Returns a number between 0 and 1 represening the
    probability of the event occuring.

Examples:
    !roll p true
    !roll p(1d20 == 1)
    !roll p(1d6 <= 2 or 1d6 >= 5)
"""


class Mean(FnOp):
    def __init__(self, arg: Expression):
        self.arg = arg

    def roll(self):
        return _Number(self.arg.mean())

    @classmethod
    def name(cls):
        return "mean"

    @classmethod
    def description(cls) -> str:
        return "average of a list"

    @classmethod
    def help(cls) -> str:
        return """mean <seq>

Arguments:
    seq - A sequence

Result:
    Returns the mean (aka average) of the given
    sequence.

Examples:
    !roll mean 1d20
    !roll mean(3d6-1)
    !roll mean(1,2,3,4)
"""


class Min(FnOp):
    def __init__(self, arg: Expression):
        self.arg = arg

    def roll(self):
        return _Number(self.arg.min())

    @classmethod
    def name(cls):
        return "min"

    @classmethod
    def description(cls) -> str:
        return "minimum of a list"

    @classmethod
    def help(cls) -> str:
        return """min <seq>

Arguments:
    seq - A sequence

Result:
    Returns the smallest possible value
    in the given sequence.

Examples:
    !roll min 1d20
    !roll min(3d6-1)
    !roll min(1,2,3,4)
"""


class Max(FnOp):
    def __init__(self, arg: Expression):
        self.arg = arg

    def roll(self):
        return _Number(self.arg.max())

    @classmethod
    def name(cls):
        return "max"

    @classmethod
    def description(cls) -> str:
        return "maximum of a list"

    @classmethod
    def help(cls) -> str:
        return """max <seq>

Arguments:
    seq - A sequence

Result:
    Returns the largest possible value
    in the given sequence.

Examples:
    !roll max 1d20
    !roll max(3d6-1)
    !roll max(1,2,3,4)
"""


class ProbTab(FnOp):
    def __init__(self, arg: Expression):
        self.arg = arg

    def roll(self):
        return self.arg.probability_table()

    @classmethod
    def name(cls):
        return "probtab"

    @classmethod
    def description(cls) -> str:
        return "print probability table"

    @classmethod
    def help(cls) -> str:
        return """probtab <x>

Arguments:
    x - A value of any type

Result:
    Prints the probability table of a given
    value. Useful mainly for debugging.

Examples:
    !roll probtab 3d6
"""


NAMES_TO_FUNCTIONS: typing.Dict[str, typing.Type[FnOp]] = {
    fn.name(): fn for fn in (ProbabilityOf, Mean, Min, Max, ProbTab)
}


def resolve_function_call(name: str, arg: Expression) -> FnOp:
    name = name.lower()
    if name in NAMES_TO_FUNCTIONS:
        return NAMES_TO_FUNCTIONS[name](arg)
    else:
        raise DiceRollError("unknown function %s" % name)
