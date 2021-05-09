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


class ExpandableFnOp(FnOp):
    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_arg, arg_expanded = self.arg.expand()
        return self.__class__(expanded_arg), arg_expanded


class SeqFnOp(FnOp):
    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_arg, arg_expanded = self.arg.as_sequence().expand()
        return (
            self.__class__(expanded_arg),
            arg_expanded,
        )


class ProbabilityOf(FnOp):
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
    def roll(self):
        return _Number(self.arg.mean())

    @classmethod
    def name(cls):
        return "mean"

    @classmethod
    def description(cls) -> str:
        return "average of a sequence"

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
    def roll(self):
        return _Number(self.arg.min())

    @classmethod
    def name(cls):
        return "min"

    @classmethod
    def description(cls) -> str:
        return "minimum of a sequence"

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
    def roll(self):
        return _Number(self.arg.max())

    @classmethod
    def name(cls):
        return "max"

    @classmethod
    def description(cls) -> str:
        return "maximum of a sequence"

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


class Seq(SeqFnOp):
    def roll(self):
        return tuple(self.arg.as_sequence().roll())

    @classmethod
    def name(cls):
        return "seq"

    @classmethod
    def description(cls) -> str:
        return "convert to a sequence"

    @classmethod
    def help(cls) -> str:
        return """seq <xdy>

Arguments:
    xdy - An XdY expression

Result:
    Converts an XdY expression to a sequence.
    Normally, XdY expressions are usable wherever
    sequences can be used, but this function forces
    the expression to be presented as a sequence.

Examples:
    !roll seq 3d4
"""


class Sum(SeqFnOp):
    def roll(self):
        return _Number(sum(self.arg.as_sequence().roll()))

    @classmethod
    def name(cls):
        return "sum"

    @classmethod
    def description(cls) -> str:
        return "sum of a sequence"

    @classmethod
    def help(cls) -> str:
        return """sum <seq>

Arguments:
    seq - A sequence

Result:
    Adds together all elements in a sequence.

Examples:
    !roll sum 3d4
    !roll sum(1,2,3)
"""


class Product(SeqFnOp):
    def roll(self):
        result = 1.0
        for value in self.arg.as_sequence().roll():
            result *= value
        return _Number(result)

    @classmethod
    def name(cls):
        return "product"

    @classmethod
    def description(cls) -> str:
        return "product of a sequence"

    @classmethod
    def help(cls) -> str:
        return """product <seq>

Arguments:
    seq - A sequence

Result:
    Multiplies together all elements in a sequence.

Examples:
    !roll product 3d4
    !roll product(1,2,3)
"""


NAMES_TO_FUNCTIONS: typing.Dict[str, typing.Type[FnOp]] = {
    fn.name(): fn for fn in (ProbabilityOf, Mean, Min, Max, ProbTab, Seq, Sum, Product)
}


def resolve_function_call(name: str, arg: Expression) -> FnOp:
    name = name.lower()
    if name in NAMES_TO_FUNCTIONS:
        return NAMES_TO_FUNCTIONS[name](arg)
    else:
        raise DiceRollError("unknown function %s" % name)
