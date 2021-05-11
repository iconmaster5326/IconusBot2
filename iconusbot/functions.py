import io
from .roll import (
    ImageResult,
    Sequence,
    Unpack,
    _Number,
    Expression,
    DiceRollError,
    Tuple,
)
import typing
import plotly.express as px
import pandas


def product(xs: typing.Iterable):
    result = 1
    for x in xs:
        result *= x
    return result


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
        return "(%s %s)" % (self.name(), self.arg)


class NonSeqFnOp(FnOp):
    def op(self, arg):
        raise NotImplementedError

    def constant(self) -> bool:
        return self.arg.constant()

    def roll(self):
        return self.op(self.arg.roll())

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        result = {}
        for k, v in self.arg.probability_table().items():
            new_key = self.op(k)
            result.setdefault(new_key, 0.0)
            result[new_key] += v
        return result

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_arg, arg_expanded = self.arg.expand()
        return self.__class__(expanded_arg), arg_expanded


class SeqFnOp(FnOp):
    def op(self, arg: tuple):
        raise NotImplementedError

    def constant(self) -> bool:
        return self.arg.constant()

    def roll(self):
        return self.op(self.arg.as_sequence().roll())

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        result = {}
        for k, v in self.arg.as_sequence().probability_table().items():
            new_key = self.op(k)
            result.setdefault(new_key, 0.0)
            result[new_key] += v
        return result

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


class Seq(SeqFnOp, Sequence):
    def op(self, arg: tuple):
        return arg

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
    def op(self, arg: tuple):
        return _Number(sum(arg))

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
    def op(self, arg: tuple):
        return _Number(product(arg))

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


class Plot(FnOp):
    def roll(self):
        KEY_VALUE = " value "

        unparsed_data = self.arg.as_sequence()
        raw_data = {KEY_VALUE: {}}
        if not isinstance(unparsed_data, Tuple):
            raise DiceRollError("plot must operate on a tuple")
        for expr in unparsed_data.args:
            if isinstance(expr, Unpack):
                raise DiceRollError("Unpacks not yet supported in plot")
            else:
                probtab = expr.probability_table()
                raw_data[KEY_VALUE].update({k: k for k in probtab})
                raw_data[str(expr)] = probtab

        data = pandas.DataFrame.from_dict(
            {
                data_key: [
                    data_value.get(table_key, 0.0) for table_key in raw_data[KEY_VALUE]
                ]
                for data_key, data_value in raw_data.items()
            }
        )
        fig = px.bar(data, x=KEY_VALUE, y=data.columns[1:], barmode="overlay")
        fig.update_yaxes(title_text="probability", tickformat="%")
        stream = io.BytesIO()
        fig.write_image(file=stream, format="png")
        return ImageResult(stream.getvalue())

    @classmethod
    def name(cls):
        return "plot"

    @classmethod
    def description(cls) -> str:
        return "produce a probability graph"

    @classmethod
    def help(cls) -> str:
        return """plot <seq>

Arguments:
    seq - A sequence of dice rolls

Result:
    Produces a graph comparing the probability
    distrobutions of all the given dice rolls.

Examples:
    !roll plot(2d6,)
    !roll plot(1d20,3d6)
"""


class Any(SeqFnOp):
    def op(self, arg: tuple):
        return any(arg)

    def probability(self) -> float:
        return sum(
            v for k, v in self.arg.as_sequence().probability_table().items() if any(k)
        )

    @classmethod
    def name(cls):
        return "any"

    @classmethod
    def description(cls) -> str:
        return "find if any items in sequence are true"

    @classmethod
    def help(cls) -> str:
        return """any <seq>

Arguments:
    seq - A sequence

Result:
    Returns true if any of the items in the given sequence
    are true, and false otherwise.

Examples:
    !roll any()
    !roll any(true,false,true)
    !roll any(seq(3d4) == 1)
    !roll any(seq(3d4) < seq(3d6))
"""


class All(SeqFnOp):
    def op(self, arg: tuple):
        return all(arg)

    def probability(self) -> float:
        return sum(
            v for k, v in self.arg.as_sequence().probability_table().items() if all(k)
        )

    @classmethod
    def name(cls):
        return "all"

    @classmethod
    def description(cls) -> str:
        return "find if all items in sequence are true"

    @classmethod
    def help(cls) -> str:
        return """all <seq>

Arguments:
    seq - A sequence

Result:
    Returns true if all of the items in the given sequence
    are true, and false otherwise.

Examples:
    !roll all()
    !roll all(true,false,true)
    !roll all(seq(3d4) == 1)
    !roll all(seq(3d4) < seq(3d6))
"""


NAMES_TO_FUNCTIONS: typing.Dict[str, typing.Type[FnOp]] = {
    fn.name(): fn
    for fn in (
        ProbabilityOf,
        Mean,
        Min,
        Max,
        ProbTab,
        Seq,
        Sum,
        Product,
        Plot,
        Any,
        All,
    )
}


def resolve_function_call(name: str, arg: Expression) -> FnOp:
    name = name.lower()
    if name in NAMES_TO_FUNCTIONS:
        return NAMES_TO_FUNCTIONS[name](arg)
    else:
        raise DiceRollError("unknown function %s" % name)
