import io
from .roll import (
    Constant,
    ImageResult,
    NotASequenceError,
    Number,
    Sequence,
    Unpack,
    _Number,
    Expression,
    DiceRollError,
    Tuple,
    _mean,
)
import typing
import plotly.express as px
import pandas
import math


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


class SeqIfArgIsSeqFnOp(NonSeqFnOp):
    def as_sequence(self) -> "Sequence":
        try:
            self.arg.as_sequence()
        except NotASequenceError:
            return super().as_sequence()
        else:
            this = self

            class Passthrough(Sequence):
                def roll(self):
                    return this.op(this.arg.as_sequence().roll())

                def constant(self) -> bool:
                    return this.constant()

                def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
                    return this.probability_table_impl()

                def __repr__(self) -> str:
                    return str(this)

            return Passthrough()


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
        KEY_VALUE = object()
        unparsed_data = self.arg.as_sequence()
        probability_tables: typing.Dict[str, typing.Dict[typing.Any, float]] = {}

        for expr in self.arg.as_sequence().unevaluated_items():
            probability_tables[str(expr)] = expr.probability_table()

        possible_values = set()

        for probtab in probability_tables.values():
            possible_values.update(probtab.keys())

        possible_values = sorted(possible_values)
        records = [tuple(str(x) for x in possible_values)]
        record_labels = [KEY_VALUE]

        for label, probtab in probability_tables.items():
            record = []
            for value in possible_values:
                record.append(probtab.get(value, 0.0))
            records.append(tuple(record))
            record_labels.append(label)

        data = pandas.DataFrame.from_records(zip(*records), columns=record_labels)
        fig = px.bar(data, x=data.columns[0], y=data.columns[1:], barmode="overlay")
        fig.update_xaxes(title_text="value")
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


class Abs(SeqIfArgIsSeqFnOp):
    def op(self, arg):
        if isinstance(arg, tuple):
            return tuple(_Number(abs(x)) for x in arg)
        return _Number(abs(arg))

    @classmethod
    def name(cls):
        return "abs"

    @classmethod
    def description(cls) -> str:
        return "absolute value"

    @classmethod
    def help(cls) -> str:
        return """abs <n>

Arguments:
    seq - A number or sequence of numbers

Result:
    Returns the absolute value of n.

Examples:
    !roll abs(1)
    !roll abs(-1)
    !roll abs(-1,0,1)
    !roll abs(seq(5d{-1,0,1}))
"""


class Explode(FnOp):
    EPSILON = 0.000001

    def roll(self):
        if self.arg.constant():
            return math.inf
        arg = self.arg.roll()
        if arg == self.arg.max():
            return _Number(arg + self.roll())
        else:
            return arg

    def constant(self) -> bool:
        return self.arg.constant()

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        result = {}
        prob_factor = 1.0
        running_total = 0.0
        probtab = self.arg.probability_table()
        maxval = self.arg.max()
        n_values = len(probtab)

        while True:
            for key, value in probtab.items():
                if key == maxval:
                    continue

                new_key = _Number(running_total + key)
                result.setdefault(new_key, 0.0)
                result[new_key] += value * prob_factor

            running_total += maxval
            prob_factor /= n_values

            if prob_factor < self.EPSILON:
                return result

    def min(self) -> float:
        if self.arg.constant():
            return math.inf
        else:
            return self.arg.min()

    def max(self) -> float:
        return math.inf

    @classmethod
    def name(cls):
        return "explode"

    @classmethod
    def description(cls) -> str:
        return "explode a die"

    @classmethod
    def help(cls) -> str:
        return (
            """explode <dn>

Arguments:
    dn - A die

Result:
    Explodes a die - that is, when the die rolls
    its maximum possible value, roll an additional
    die and add the results together. This process
    may continue indefinitely as long as the new die
    rolled is its maximum value.

    Since the result of this has no maximum value,
    the probability table generated from this function
    is cut off beyond a certain epsilon (configured
    by the server owner to be %s).

Examples:
    !roll explode d10
    !roll explode d{2,4,6,8}
"""
            % f"{cls.EPSILON*100:f}%"
        )


class Eval(FnOp, Sequence):
    def __init__(self, arg: Expression):
        FnOp.__init__(self, arg)
        self.evaluated_arg = arg.roll()

    def roll(self):
        return self.evaluated_arg

    def constant(self) -> bool:
        return True

    def expand(self) -> typing.Tuple["Expression", bool]:
        return Constant(self.evaluated_arg), True

    def as_sequence(self) -> "Sequence":
        if isinstance(self.evaluated_arg, tuple):
            return self
        else:
            return FnOp.as_sequence(self)

    def unevaluated_items(self) -> typing.Iterable[Expression]:
        if isinstance(self.evaluated_arg, tuple):
            return (Constant(x) for x in self.evaluated_arg)
        else:
            return Sequence.unevaluated_items(self)

    def mean(self) -> float:
        if isinstance(self.evaluated_arg, tuple):
            return _mean(self.evaluated_arg)
        else:
            return self.evaluated_arg

    def min(self) -> float:
        if isinstance(self.evaluated_arg, tuple):
            return min(self.evaluated_arg)
        else:
            return self.evaluated_arg

    def max(self) -> float:
        if isinstance(self.evaluated_arg, tuple):
            return max(self.evaluated_arg)
        else:
            return self.evaluated_arg

    @classmethod
    def name(cls):
        return "eval"

    @classmethod
    def description(cls) -> str:
        return "eagerly evaluate something"

    @classmethod
    def help(cls) -> str:
        return """eval <x>

Arguments:
    x - Any expression

Result:
    Evaluates a value immediately. This is useful
    in functions like `p`, `mean`, `plot`, and
    others, where arguments are not usually
    evaluated.

Examples:
    !roll eval d6
    !roll plot(eval d6, d6)
"""


class Rep(FnOp, Sequence):
    def __init__(self, arg: Expression):
        super().__init__(arg)
        self.unevaled_args = list(arg.as_sequence().unevaluated_items())
        if len(self.unevaled_args) != 2:
            raise DiceRollError(
                "'%s' expected %s arguments, got %s"
                % (self, 2, len(self.unevaled_args))
            )

    def roll(self):
        return tuple(
            self.unevaled_args[0].roll()
            for _ in range(int(self.unevaled_args[1].roll()))
        )

    def constant(self) -> bool:
        if not self.unevaled_args[1].constant():
            return False
        return self.unevaled_args[1].roll() <= 0 or self.unevaled_args[0].constant()

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_arg1, arg1_expanded = self.unevaled_args[1].expand()
        if arg1_expanded:
            return (self.__class__(Tuple(self.unevaled_args[0], expanded_arg1)), True)
        else:
            return (
                Tuple(*((self.unevaled_args[0],) * int(expanded_arg1.roll()))),
                True,
            )

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        result = {}
        for key_1, value_1 in self.unevaled_args[1].probability_table().items():
            for key_tail, value_tail in (
                self.__class__(Tuple(self.unevaled_args[0], Number(key_1 - 1)))
                .probability_table()
                .items()
            ):
                for key_0, value_0 in self.unevaled_args[0].probability_table().items():
                    new_key = (key_0,) + key_tail
                    result.setdefault(new_key, 0.0)
                    result[new_key] += value_1 * value_tail * value_0
        return result

    def mean(self) -> float:
        return self.unevaled_args[0].mean()

    def min(self) -> float:
        return self.unevaled_args[0].min()

    def max(self) -> float:
        return self.unevaled_args[0].max()

    def unevaluated_items(self) -> typing.Iterable[Expression]:
        if not self.unevaled_args[1].constant():
            return super().unevaluated_items()
        for _ in range(int(self.unevaled_args[1].roll())):
            yield self.unevaled_args[0]

    @classmethod
    def name(cls):
        return "rep"

    @classmethod
    def description(cls) -> str:
        return "repeat a value"

    @classmethod
    def help(cls) -> str:
        return """rep (<x>, <n>)

Arguments:
    x - The expression to repeat
    n - The number of times to repeat

Result:
    Returns a sequence of length n,
    in which every element is the
    reuslt of rolling x. x itself
    is not eagerly evaluated.

Examples:
    !roll rep(d6, 4)
    !roll rep(eval(d20), 6)
    !roll rep(4d6 drop worst 1, 6)
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
        Abs,
        Explode,
        Eval,
        Rep,
    )
}


def resolve_function_call(name: str, arg: Expression) -> FnOp:
    name = name.lower()
    if name in NAMES_TO_FUNCTIONS:
        return NAMES_TO_FUNCTIONS[name](arg)
    else:
        raise DiceRollError("unknown function %s" % name)
