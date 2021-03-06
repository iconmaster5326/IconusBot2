import random
import typing
import itertools


class _Number(float):
    def __repr__(self) -> str:
        result = f"{float(self):.2f}"
        if result.endswith(".00"):
            result = result[:-3]
        return result


class ImageResult:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def __repr__(self) -> str:
        return "📊"


class DiceRollError(ValueError):
    pass


class NotASequenceError(DiceRollError):
    pass


class Expression:
    def roll(self):
        raise NotImplementedError

    def probability(self) -> float:
        raise DiceRollError("Probability of '%s' cannot be computed" % self)

    def mean(self) -> float:
        try:
            if self.constant():
                return self.roll()
            else:
                result = 0.0
                for key, value in self.probability_table().items():
                    result += key * value
                return result
        except DiceRollError:
            raise DiceRollError("Mean of '%s' cannot be computed" % self)

    def min(self) -> float:
        try:
            if self.constant():
                return self.roll()
            else:
                return min(self.probability_table())
        except DiceRollError:
            raise DiceRollError("Minimum of '%s' cannot be computed" % self)

    def max(self) -> float:
        try:
            if self.constant():
                return self.roll()
            else:
                return max(self.probability_table())
        except DiceRollError:
            raise DiceRollError("Maximum of '%s' cannot be computed" % self)

    def constant(self) -> bool:
        return True

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        raise DiceRollError("Probability table of '%s' cannot be computed" % self)

    def probability_table(self) -> typing.Dict[typing.Any, float]:
        if self.constant():
            return {self.roll(): 1.0}
        else:
            return self.probability_table_impl()

    def as_sequence(self) -> "Sequence":
        raise NotASequenceError("'%s' is not a sequence" % self)

    def expand(self) -> typing.Tuple["Expression", bool]:
        return self, False


def _mean(xs: typing.Iterable[float]) -> float:
    total = 0.0
    n = 0
    for x in xs:
        total += x
        n += 1
    return total / n


class Sequence(Expression):
    def unevaluated_items(self) -> typing.Iterable[Expression]:
        raise DiceRollError("sequence '%s' does not have fixed contents" % self)

    def as_sequence(self) -> "Sequence":
        return self

    def mean(self) -> float:
        try:
            if self.constant():
                return _mean(self.roll())
            else:
                result = 0.0
                for key, value in self.probability_table().items():
                    result += _mean(key) * value
                return result
        except DiceRollError:
            raise DiceRollError("Mean of '%s' cannot be computed" % self)

    def min(self) -> float:
        try:
            if self.constant():
                return min(self.roll())
            else:
                return min(min(k) for k in self.probability_table())
        except DiceRollError:
            raise DiceRollError("Minimum of '%s' cannot be computed" % self)

    def max(self) -> float:
        try:
            if self.constant():
                return max(self.roll())
            else:
                return max(max(k) for k in self.probability_table())
        except DiceRollError:
            raise DiceRollError("Maximum of '%s' cannot be computed" % self)


class Constant(Expression):
    def __init__(self, value):
        self.value = value

    def roll(self):
        return self.value

    def __repr__(self):
        return str(self.value)


class Number(Expression):
    def __init__(self, value):
        self.value = float(value)

    def roll(self):
        return _Number(self.value)

    def __repr__(self):
        return str(self.roll())


class TrueValue(Expression):
    def roll(self):
        return True

    def probability(self) -> float:
        return 1.0

    def __repr__(self):
        return "true"


class FalseValue(Expression):
    def roll(self):
        return False

    def probability(self) -> float:
        return 0.0

    def __repr__(self):
        return "false"


class Unpack:
    def __init__(self, value: Expression) -> None:
        self.value = value

    def __repr__(self) -> str:
        return "\\*%s" % self.value

    def constant(self) -> bool:
        return self.value.constant()

    def expand(self) -> typing.Tuple[Expression, bool]:
        return self.value.as_sequence().expand()


class Tuple(Sequence):
    def __init__(self, *args: typing.Union[Expression, Unpack]) -> None:
        self.args = tuple(args)

    def _roll_arg(self, arg: typing.Union[Expression, Unpack]) -> typing.Tuple:
        if isinstance(arg, Unpack):
            return arg.value.as_sequence().roll()
        else:
            return (arg.roll(),)

    def roll(self):
        return tuple(
            itertools.chain.from_iterable(self._roll_arg(arg) for arg in self.args)
        )

    def __repr__(self):
        return "(" + ", ".join(str(arg) for arg in self.args) + ")"

    def constant(self) -> bool:
        return all(arg.constant() for arg in self.args)

    def expand(self) -> typing.Tuple["Tuple", bool]:
        expanded_args = []
        args_expanded = False
        for arg in self.args:
            expanded_arg, arg_expanded = arg.expand()
            if arg_expanded:
                args_expanded = True
            if isinstance(arg, Unpack):
                expanded_args.append(Unpack(expanded_arg))
            else:
                expanded_args.append(expanded_arg)

        return self.__class__(*expanded_args), args_expanded

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        if len(self.args) == 0:
            return {(): 1.0}

        head = self.args[0]
        if isinstance(head, Unpack):
            probtab_head = head.value.as_sequence().probability_table()
        else:
            probtab_head = head.probability_table()

        probtab_tail = Tuple(*self.args[1:]).probability_table()
        result = {}

        for key_head, value_head in probtab_head.items():
            for key_tail, value_tail in probtab_tail.items():
                if isinstance(head, Unpack):
                    new_key = (*key_head, *key_tail)
                else:
                    new_key = (key_head, *key_tail)
                result.setdefault(new_key, 0.0)
                result[new_key] += value_head * value_tail
        return result

    def unevaluated_items(self) -> typing.Iterable[Expression]:
        for arg in self.args:
            if isinstance(arg, Unpack):
                yield from arg.value.as_sequence().unevaluated_items()
            else:
                yield arg


class BiMathOp(Expression):
    def op(
        self,
        lhs: typing.Union[typing.Tuple, float],
        rhs: typing.Union[typing.Tuple, float],
    ) -> typing.Union[typing.Tuple, float]:
        if isinstance(lhs, tuple) and isinstance(rhs, tuple):
            if len(lhs) != len(rhs):
                raise DiceRollError("sequences in '%s' are not the same length" % self)
            return tuple(self.op(a, b) for a, b in zip(lhs, rhs))
        elif isinstance(lhs, tuple):
            return tuple(self.op(x, rhs) for x in lhs)
        elif isinstance(rhs, tuple):
            return tuple(self.op(lhs, x) for x in rhs)
        else:
            return _Number(self.op_impl(lhs, rhs))

    def op_impl(self, lhs: float, rhs: float) -> float:
        raise NotImplementedError

    def __init__(self, lhs: Expression, rhs: Expression):
        self.lhs = lhs
        self.rhs = rhs

    def roll(self):
        return self.op(self.lhs.roll(), self.rhs.roll())

    def constant(self) -> bool:
        return self.lhs.constant() and self.rhs.constant()

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_lhs, lhs_expanded = self.lhs.expand()
        expanded_rhs, rhs_expanded = self.rhs.expand()
        return self.__class__(expanded_lhs, expanded_rhs), lhs_expanded or rhs_expanded

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        table1 = self.lhs.probability_table()
        table2 = self.rhs.probability_table()
        result: typing.Dict[typing.Any, float] = {}
        for key1, value1 in table1.items():
            for key2, value2 in table2.items():
                new_key = self.op(key1, key2)
                result.setdefault(new_key, 0)
                result[new_key] += value1 * value2
        return result

    def mean(self) -> float:
        return self.op_impl(self.lhs.mean(), self.rhs.mean())

    def min(self) -> float:
        return self.op_impl(self.lhs.min(), self.rhs.min())

    def max(self) -> float:
        return self.op_impl(self.lhs.max(), self.rhs.max())

    def as_sequence(self) -> "Sequence":
        this = self
        try:
            lhs = self.lhs.as_sequence()
            lhs_seq = True
        except NotASequenceError:
            lhs_seq = False
        try:
            rhs = self.rhs.as_sequence()
            rhs_seq = True
        except NotASequenceError:
            rhs_seq = False

        if not lhs_seq and not rhs_seq:
            return super().as_sequence()
        if lhs_seq and rhs_seq:

            class BinMathOpDualSeq(Sequence):
                def roll(self):
                    return this.op(lhs.roll(), rhs.roll())

                def constant(self) -> bool:
                    return this.constant()

                def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
                    table1 = lhs.probability_table()
                    table2 = rhs.probability_table()
                    result: typing.Dict[typing.Any, float] = {}
                    for key1, value1 in table1.items():
                        for key2, value2 in table2.items():
                            new_key = this.op(key1, key2)
                            result.setdefault(new_key, 0)
                            result[new_key] += value1 * value2
                    return result

                def __repr__(self) -> str:
                    return this.__repr__()

                def unevaluated_items(self) -> typing.Iterable[Expression]:
                    for arg1, arg2 in zip(
                        lhs.unevaluated_items(), rhs.unevaluated_items()
                    ):
                        yield this.__class__(arg1, arg2)

            return BinMathOpDualSeq()
        else:

            class BinMathOpOneSeq(Sequence):
                def roll(self):
                    if lhs_seq:
                        lhs_rolled = lhs.roll()
                    else:
                        lhs_rolled = this.lhs.roll()
                    if rhs_seq:
                        rhs_rolled = rhs.roll()
                    else:
                        rhs_rolled = this.rhs.roll()

                    return this.op(lhs_rolled, rhs_rolled)

                def constant(self) -> bool:
                    return this.constant()

                def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
                    if lhs_seq:
                        table1 = lhs.probability_table()
                    else:
                        table1 = this.lhs.probability_table()
                    if rhs_seq:
                        table2 = rhs.probability_table()
                    else:
                        table2 = this.rhs.probability_table()

                    result: typing.Dict[typing.Any, float] = {}
                    for key1, value1 in table1.items():
                        for key2, value2 in table2.items():
                            new_key = this.op(key1, key2)
                            result.setdefault(new_key, 0)
                            result[new_key] += value1 * value2
                    return result

                def __repr__(self) -> str:
                    return this.__repr__()

                def unevaluated_items(self) -> typing.Iterable[Expression]:
                    if lhs_seq:
                        lhs_value = lhs.unevaluated_items()
                    else:
                        lhs_value = [this.lhs] * len(tuple(rhs.unevaluated_items()))
                    if rhs_seq:
                        rhs_value = rhs.unevaluated_items()
                    else:
                        rhs_value = [this.rhs] * len(tuple(lhs.unevaluated_items()))

                    for arg1, arg2 in zip(lhs_value, rhs_value):
                        yield this.__class__(arg1, arg2)

            return BinMathOpOneSeq()


class Add(BiMathOp):
    def op_impl(self, lhs: float, rhs: float) -> float:
        return lhs + rhs

    def __repr__(self):
        return "(%s + %s)" % (self.lhs, self.rhs)


class Sub(BiMathOp):
    def op_impl(self, lhs: float, rhs: float) -> float:
        return lhs - rhs

    def __repr__(self):
        return "(%s - %s)" % (self.lhs, self.rhs)


class Mul(BiMathOp):
    def op_impl(self, lhs: float, rhs: float) -> float:
        return lhs * rhs

    def __repr__(self):
        return "(%s * %s)" % (self.lhs, self.rhs)


class Div(BiMathOp):
    def op_impl(self, lhs: float, rhs: float) -> float:
        return lhs / rhs

    def __repr__(self):
        return "(%s / %s)" % (self.lhs, self.rhs)


class Neg(Expression):
    def __init__(self, lhs: Expression):
        self.lhs = lhs

    def roll(self):
        return _Number(-self.lhs.roll())

    def constant(self) -> bool:
        return self.lhs.constant()

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_lhs, lhs_expanded = self.lhs.expand()
        return Neg(expanded_lhs), lhs_expanded

    def __repr__(self):
        return "-%s" % self.lhs

    def mean(self) -> float:
        return -self.lhs.mean()

    def min(self) -> float:
        return -self.lhs.min()

    def max(self) -> float:
        return -self.lhs.max()


class Count(Expression):
    def __init__(self, lhs: Expression):
        self.lhs = lhs

    def roll(self):
        return _Number(len(self.lhs.as_sequence().roll()))

    def constant(self) -> bool:
        return self.lhs.constant()

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_lhs, lhs_expanded = self.lhs.as_sequence().expand()
        return Count(expanded_lhs), lhs_expanded

    def __repr__(self):
        return "#%s" % self.lhs

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        result = {}
        for key, value in self.lhs.as_sequence().probability_table().items():
            new_key = len(key)
            result.setdefault(new_key, 0.0)
            result[new_key] += value
        return result


class Range(Sequence):
    def __init__(
        self,
        from_: Expression,
        to: Expression,
        step: typing.Optional[Expression] = None,
    ) -> None:
        self.from_ = from_
        self.to = to
        self.step = step

    def roll(self):
        step = 1 if self.step is None else int(self.step.roll())
        return tuple(
            range(
                int(self.from_.roll()),
                int(self.to.roll()) + (-1 if step < 0 else 1),
                int(step),
            )
        )

    def constant(self) -> bool:
        return (
            self.from_.constant()
            and self.to.constant()
            and (True if self.step is None else self.step.constant())
        )

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_from, from_expanded = self.from_.expand()
        expanded_to, to_expanded = self.to.expand()
        expanded_step, step_expanded = (
            (None, False) if self.step is None else self.step.expand()
        )

        return (
            self.__class__(expanded_from, expanded_to, expanded_step),
            from_expanded or to_expanded or step_expanded,
        )

    def __repr__(self) -> str:
        return "(%s to %s)" % (self.from_, self.to) + (
            "" if self.step is None else " by %s" % self.step
        )

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        result = {}
        for from_key, from_value in self.from_.probability_table().items():
            for to_key, to_value in self.to.probability_table().items():
                for step_key, step_value in (
                    (Number(1) if self.step is None else self.step)
                    .probability_table()
                    .items()
                ):
                    new_key = tuple(
                        range(
                            int(from_key),
                            int(to_key) + (-1 if step_key < 0 else 1),
                            int(step_key),
                        )
                    )
                    result.setdefault(new_key, 0.0)
                    result[new_key] += from_value * to_value * step_value
        return result

    def unevaluated_items(self) -> typing.Iterable[Expression]:
        if not self.constant():
            return super().unevaluated_items()
        else:
            return (Number(x) for x in self.roll())


class Die(Expression):
    pass


class DieSequence(Die):
    def __init__(self, sequence: Tuple) -> None:
        self.sequence = sequence

    def roll(self):
        xs = self.sequence.as_sequence().roll()
        if len(xs) == 0:
            raise DiceRollError("attempted to roll d{}")
        return random.choice(xs)

    def constant(self) -> bool:
        return self.sequence.constant() and len(self.sequence.as_sequence().roll()) <= 1

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_seq, seq_expanded = self.sequence.expand()
        if seq_expanded:
            return self.__class__(expanded_seq), True
        else:
            return Constant(self.roll()), True

    def __repr__(self) -> str:
        return "d{%s}" % ", ".join(str(x) for x in self.sequence.args)

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        result = {}
        for key, value in self.sequence.probability_table().items():
            for new_key in key:
                result.setdefault(new_key, 0.0)
                result[new_key] += value * 1 / len(key)
        return result

    def min(self) -> float:
        return self.sequence.min()

    def max(self) -> float:
        return self.sequence.max()

    def mean(self) -> float:
        return self.sequence.mean()


class DieNumber(Die):
    def __init__(self, number: Expression) -> None:
        self.number = number

    def constant(self) -> bool:
        return self.number.constant() and self.number.roll() <= 1

    def roll(self):
        n = self.number.roll()
        if not isinstance(n, float) and not isinstance(n, int):
            raise DiceRollError("number expected, got %s" % n)
        if n < 1:
            raise DiceRollError("attempted to roll a die with %s faces" % n)
        return random.randint(1, int(n))

    def __repr__(self) -> str:
        return "d%s" % self.number

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_seq, seq_expanded = self.number.expand()
        if seq_expanded:
            return self.__class__(expanded_seq), True
        else:
            return Constant(self.roll()), True

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        if self.number.constant():
            n = self.number.roll()
            return {i: 1 / n for i in range(1, int(n) + 1)}
        else:
            result = {}
            for key, value in self.number.probability_table().items():
                for i in range(1, int(key) + 1):
                    result.setdefault(i, 0.0)
                    result[i] += value * 1 / key
            return result

    def min(self) -> float:
        return 1.0

    def max(self) -> float:
        return self.number.max()


class Dice(Expression):
    def __init__(self, n_dice: Expression, dice: Die) -> None:
        self.n_dice = n_dice
        self.dice = dice

    def roll(self):
        n_dice = self.n_dice.roll()
        if not isinstance(n_dice, int) and not isinstance(n_dice, float):
            raise DiceRollError("attempted to roll %s dice" % (n_dice,))
        return sum(self.dice.roll() for _ in range(int(n_dice)))

    def constant(self) -> bool:
        return self.n_dice.constant() and self.dice.constant()

    def expand(self) -> typing.Tuple["Expression", bool]:
        class ExpandedDice(Tuple):
            def roll(self):
                return _Number(sum(super().roll()))

            def __repr__(self) -> str:
                return (
                    "0"
                    if len(self.args) == 0
                    else "(%s)" % " + ".join(str(x) for x in self.args)
                )

            def as_sequence(self) -> Sequence:
                return Tuple(*self.args)

        expanded_lhs, lhs_expanded = self.n_dice.expand()
        if lhs_expanded:
            return self.__class__(expanded_lhs, self.dice), True
        else:
            n_dice = expanded_lhs.roll()
            if not isinstance(n_dice, int) and not isinstance(n_dice, float):
                raise DiceRollError("attempted to roll %s dice" % (n_dice))
            if n_dice == 1:
                return Constant(self.dice.roll()), True
            else:
                return (
                    ExpandedDice(
                        *(Constant(x.roll()) for x in [self.dice] * int(n_dice))
                    ),
                    True,
                )

    def __repr__(self) -> str:
        return "(%s%s)" % (self.n_dice, self.dice)

    def as_sequence(self) -> Sequence:
        if self.n_dice.constant():
            return Tuple(*([self.dice] * int(self.n_dice.roll())))
        else:
            dice = self

            class DiceSequence(Sequence):
                def roll(self):
                    n_dice = dice.n_dice.roll()
                    if not isinstance(n_dice, int) and not isinstance(n_dice, float):
                        raise DiceRollError("attempted to roll %s dice" % (n_dice,))
                    return tuple(dice.dice.roll() for _ in range(int(n_dice)))

                def constant(self) -> bool:
                    return dice.constant()

                def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
                    result = {}
                    for n_key, n_value in dice.n_dice.probability_table().items():
                        for seq_key, seq_value in (
                            Dice(Number(n_key), dice.dice)
                            .as_sequence()
                            .probability_table_impl()
                            .items()
                        ):
                            result.setdefault(seq_key, 0.0)
                            result[seq_key] += n_value * seq_value
                    return result

            return DiceSequence()

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        result = {}
        for key, value in self.as_sequence().probability_table().items():
            new_key = sum(key)
            result.setdefault(new_key, 0.0)
            result[new_key] += value
        return result

    def min(self) -> float:
        return self.n_dice.min() * self.dice.min()

    def max(self) -> float:
        return self.n_dice.max() * self.dice.max()


class BinCompOp(Expression):
    def op_impl(self, lhs, rhs) -> bool:
        raise NotImplementedError

    def op(
        self,
        lhs: typing.Union[typing.Tuple, float],
        rhs: typing.Union[typing.Tuple, float],
    ) -> typing.Union[typing.Tuple, bool]:
        if isinstance(lhs, tuple) and isinstance(rhs, tuple):
            if len(lhs) != len(rhs):
                raise DiceRollError("sequences in '%s' are not the same length" % self)
            return tuple(self.op(a, b) for a, b in zip(lhs, rhs))
        elif isinstance(lhs, tuple):
            return tuple(self.op(x, rhs) for x in lhs)
        elif isinstance(rhs, tuple):
            return tuple(self.op(lhs, x) for x in rhs)
        else:
            return self.op_impl(lhs, rhs)

    def __init__(self, lhs: Expression, rhs: Expression):
        self.lhs = lhs
        self.rhs = rhs

    def roll(self):
        return self.op(self.lhs.roll(), self.rhs.roll())

    def probability(self) -> float:
        if self.constant():
            return 1.0 if self.op(self.lhs.roll(), self.rhs.roll()) else 0.0
        else:
            table1 = self.lhs.probability_table()
            table2 = self.rhs.probability_table()
            result = 0
            for key1 in table1:
                for key2 in table2:
                    if self.op(key1, key2):
                        result += table1[key1] * table2[key2]
            return result

    def constant(self) -> bool:
        return self.lhs.constant() and self.rhs.constant()

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_lhs, lhs_expanded = self.lhs.expand()
        expanded_rhs, rhs_expanded = self.rhs.expand()
        return self.__class__(expanded_lhs, expanded_rhs), lhs_expanded or rhs_expanded

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        p = self.probability()
        return {False: 1.0 - p, True: p}

    def mean(self) -> float:
        raise DiceRollError("Mean of '%s' cannot be computed" % self)

    def min(self) -> float:
        raise DiceRollError("Minimum of '%s' cannot be computed" % self)

    def max(self) -> float:
        raise DiceRollError("Maximum of '%s' cannot be computed" % self)

    def as_sequence(self) -> "Sequence":
        this = self
        try:
            lhs = self.lhs.as_sequence()
            lhs_seq = True
        except NotASequenceError:
            lhs_seq = False
        try:
            rhs = self.rhs.as_sequence()
            rhs_seq = True
        except NotASequenceError:
            rhs_seq = False

        if not lhs_seq and not rhs_seq:
            return super().as_sequence()
        if lhs_seq and rhs_seq:

            class BinCompOpDualSeq(Sequence):
                def roll(self):
                    return this.op(lhs.roll(), rhs.roll())

                def constant(self) -> bool:
                    return this.constant()

                def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
                    table1 = lhs.probability_table()
                    table2 = rhs.probability_table()
                    result: typing.Dict[typing.Any, float] = {}
                    for key1, value1 in table1.items():
                        for key2, value2 in table2.items():
                            new_key = this.op(key1, key2)
                            result.setdefault(new_key, 0)
                            result[new_key] += value1 * value2
                    return result

                def __repr__(self) -> str:
                    return this.__repr__()

                def unevaluated_items(self) -> typing.Iterable[Expression]:
                    for arg1, arg2 in zip(
                        lhs.unevaluated_items(), rhs.unevaluated_items()
                    ):
                        yield this.__class__(arg1, arg2)

            return BinCompOpDualSeq()
        else:

            class BinCompOpOneSeq(Sequence):
                def roll(self):
                    if lhs_seq:
                        lhs_rolled = lhs.roll()
                    else:
                        lhs_rolled = this.lhs.roll()
                    if rhs_seq:
                        rhs_rolled = rhs.roll()
                    else:
                        rhs_rolled = this.rhs.roll()

                    return this.op(lhs_rolled, rhs_rolled)

                def constant(self) -> bool:
                    return this.constant()

                def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
                    if lhs_seq:
                        table1 = lhs.probability_table()
                    else:
                        table1 = this.lhs.probability_table()
                    if rhs_seq:
                        table2 = rhs.probability_table()
                    else:
                        table2 = this.rhs.probability_table()

                    result: typing.Dict[typing.Any, float] = {}
                    for key1, value1 in table1.items():
                        for key2, value2 in table2.items():
                            new_key = this.op(key1, key2)
                            result.setdefault(new_key, 0)
                            result[new_key] += value1 * value2
                    return result

                def __repr__(self) -> str:
                    return this.__repr__()

                def unevaluated_items(self) -> typing.Iterable[Expression]:
                    if lhs_seq:
                        lhs_value = lhs.unevaluated_items()
                    else:
                        lhs_value = [this.lhs] * len(tuple(rhs.unevaluated_items()))
                    if rhs_seq:
                        rhs_value = rhs.unevaluated_items()
                    else:
                        rhs_value = [this.rhs] * len(tuple(lhs.unevaluated_items()))

                    for arg1, arg2 in zip(lhs_value, rhs_value):
                        yield this.__class__(arg1, arg2)

            return BinCompOpOneSeq()


class Eq(BinCompOp):
    def op_impl(self, lhs, rhs) -> bool:
        return lhs == rhs

    def __repr__(self):
        return "(%s == %s)" % (self.lhs, self.rhs)

    def probability(self) -> float:
        if self.constant():
            return 1.0 if self.op(self.lhs.roll(), self.rhs.roll()) else 0.0
        elif self.lhs.constant():
            return self.rhs.probability_table().get(self.lhs.roll(), 0.0)
        elif self.rhs.constant():
            return self.lhs.probability_table().get(self.rhs.roll(), 0.0)
        else:
            table1 = self.lhs.probability_table()
            table2 = self.rhs.probability_table()
            result = 0
            for key, value1 in table1.items():
                if key in table2:
                    value2 = table2[key]
                    result += value1 * value2
            return result


class Ne(BinCompOp):
    def op_impl(self, lhs, rhs) -> bool:
        return lhs != rhs

    def __repr__(self):
        return "(%s != %s)" % (self.lhs, self.rhs)


class Le(BinCompOp):
    def op_impl(self, lhs, rhs) -> bool:
        return lhs <= rhs

    def __repr__(self):
        return "(%s <= %s)" % (self.lhs, self.rhs)


class Ge(BinCompOp):
    def op_impl(self, lhs, rhs) -> bool:
        return lhs >= rhs

    def __repr__(self):
        return "(%s >= %s)" % (self.lhs, self.rhs)


class Lt(BinCompOp):
    def op_impl(self, lhs, rhs) -> bool:
        return lhs < rhs

    def __repr__(self):
        return "(%s < %s)" % (self.lhs, self.rhs)


class Gt(BinCompOp):
    def op_impl(self, lhs, rhs) -> bool:
        return lhs > rhs

    def __repr__(self):
        return "(%s > %s)" % (self.lhs, self.rhs)


class EvaluatedSequence(Sequence):
    def __init__(self, *args):
        self.args = tuple(args)

    def roll(self):
        return self.args

    def __repr__(self):
        return "(" + ", ".join(str(arg) for arg in self.args) + ")"

    def unevaluated_items(self) -> typing.Iterable[Expression]:
        return (Constant(x) for x in self.args)


class DropKeepOp(Expression):
    def op(self, lhs: list, rhs: int) -> list:
        raise NotImplementedError

    def roll(self):
        return _Number(sum(self.as_sequence().roll()))

    def __init__(self, lhs: Expression, rhs: Expression):
        self.lhs = lhs
        self.rhs = rhs

    def constant(self) -> bool:
        return self.lhs.constant() and self.rhs.constant()

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_lhs, lhs_expanded = self.lhs.as_sequence().expand()
        expanded_rhs, rhs_expanded = self.rhs.expand()
        return (
            self.__class__(expanded_lhs, expanded_rhs),
            lhs_expanded or rhs_expanded,
        )

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        result = {}
        for key, value in self.as_sequence().probability_table().items():
            new_key = sum(key)
            result.setdefault(new_key, 0.0)
            result[new_key] += value
        return result

    def as_sequence(self) -> Sequence:
        this = self

        class DropKeepSeq(Sequence):
            def roll(self):
                values = list(this.lhs.as_sequence().roll())
                values.sort()
                n_to_drop = int(this.rhs.roll())
                return tuple(this.op(values, n_to_drop))

            def constant(self) -> bool:
                return this.constant()

            def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
                result = {}
                for lhs_key, lhs_value in (
                    this.lhs.as_sequence().probability_table().items()
                ):
                    for rhs_key, rhs_value in this.rhs.probability_table().items():
                        xs = list(lhs_key)
                        xs.sort()
                        new_key = tuple(this.op(xs, int(rhs_key)))
                        result.setdefault(new_key, 0.0)
                        result[new_key] += lhs_value * rhs_value
                return result

        return DropKeepSeq()


class DropWorst(DropKeepOp):
    def op(self, lhs: list, rhs: float) -> list:
        return lhs[rhs:]

    def __repr__(self):
        return "(%s drop worst %s)" % (self.lhs, self.rhs)


class DropBest(DropKeepOp):
    def op(self, lhs: list, rhs: float) -> list:
        return lhs[:-rhs]

    def __repr__(self):
        return "(%s drop best %s)" % (self.lhs, self.rhs)


class KeepWorst(DropKeepOp):
    def op(self, lhs: list, rhs: float) -> list:
        return lhs[:rhs]

    def __repr__(self):
        return "(%s keep worst %s)" % (self.lhs, self.rhs)


class KeepBest(DropKeepOp):
    def op(self, lhs: list, rhs: float) -> list:
        return lhs[-rhs:]

    def __repr__(self):
        return "(%s keep best %s)" % (self.lhs, self.rhs)


class And(Expression):
    def __init__(self, lhs: Expression, rhs: Expression):
        self.lhs = lhs
        self.rhs = rhs

    def roll(self):
        return self.lhs.roll() and self.rhs.roll()

    def probability(self) -> float:
        return self.lhs.probability() * self.rhs.probability()

    def constant(self) -> bool:
        return self.lhs.constant() and self.rhs.constant()

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_lhs, lhs_expanded = self.lhs.expand()
        expanded_rhs, rhs_expanded = self.rhs.expand()
        return self.__class__(expanded_lhs, expanded_rhs), lhs_expanded or rhs_expanded

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        p = self.probability()
        return {False: 1.0 - p, True: p}

    def mean(self) -> float:
        raise DiceRollError("Mean of '%s' cannot be computed" % self)

    def min(self) -> float:
        raise DiceRollError("Minimum of '%s' cannot be computed" % self)

    def max(self) -> float:
        raise DiceRollError("Maximum of '%s' cannot be computed" % self)

    def __repr__(self) -> str:
        return "(%s and %s)" % (self.lhs, self.rhs)


class Or(Expression):
    def __init__(self, lhs: Expression, rhs: Expression):
        self.lhs = lhs
        self.rhs = rhs

    def roll(self):
        return self.lhs.roll() or self.rhs.roll()

    def probability(self) -> float:
        return self.lhs.probability() + self.rhs.probability()

    def constant(self) -> bool:
        return self.lhs.constant() and self.rhs.constant()

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_lhs, lhs_expanded = self.lhs.expand()
        expanded_rhs, rhs_expanded = self.rhs.expand()
        return self.__class__(expanded_lhs, expanded_rhs), lhs_expanded or rhs_expanded

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        p = self.probability()
        return {False: 1.0 - p, True: p}

    def mean(self) -> float:
        raise DiceRollError("Mean of '%s' cannot be computed" % self)

    def min(self) -> float:
        raise DiceRollError("Minimum of '%s' cannot be computed" % self)

    def max(self) -> float:
        raise DiceRollError("Maximum of '%s' cannot be computed" % self)

    def __repr__(self) -> str:
        return "(%s or %s)" % (self.lhs, self.rhs)


class Not(Expression):
    def __init__(self, lhs: Expression):
        self.lhs = lhs

    def roll(self):
        return not self.lhs.roll()

    def probability(self) -> float:
        return 1.0 - self.lhs.probability()

    def constant(self) -> bool:
        return self.lhs.constant()

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_lhs, lhs_expanded = self.lhs.expand()
        return self.__class__(expanded_lhs), lhs_expanded

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        p = self.probability()
        return {False: 1.0 - p, True: p}

    def mean(self) -> float:
        raise DiceRollError("Mean of '%s' cannot be computed" % self)

    def min(self) -> float:
        raise DiceRollError("Minimum of '%s' cannot be computed" % self)

    def max(self) -> float:
        raise DiceRollError("Maximum of '%s' cannot be computed" % self)

    def __repr__(self) -> str:
        return "not (%s)" % self.lhs


class IfThenElse(Expression):
    def __init__(self, if_: Expression, then: Expression, else_: Expression):
        self.if_ = if_
        self.then = then
        self.else_ = else_

    def roll(self):
        return self.then.roll() if self.if_.roll() else self.else_.roll()

    def probability(self) -> float:
        cond_prob = self.if_.probability()
        return (
            cond_prob * self.then.probability()
            + (1 - cond_prob) * self.else_.probability()
        )

    def constant(self) -> bool:
        if self.if_.constant():
            if self.if_.roll():
                return self.then.constant()
            else:
                return self.else_.constant()
        return False

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        cond_prob = self.if_.probability()
        result = {}
        for key, value in self.then.probability_table().items():
            result.setdefault(key, 0.0)
            result[key] += cond_prob * value
        for key, value in self.else_.probability_table().items():
            result.setdefault(key, 0.0)
            result[key] += (1 - cond_prob) * value
        return result

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_cond, cond_expanded = self.if_.expand()
        expanded_then, then_expanded = self.then.expand()
        expanded_else, else_expanded = self.else_.expand()
        return (
            self.__class__(expanded_cond, expanded_then, expanded_else),
            cond_expanded or then_expanded or else_expanded,
        )

    def __repr__(self) -> str:
        return "(if %s then %s else %s)" % (self.if_, self.then, self.else_)

    def as_sequence(self) -> Sequence:
        ifte = self

        class IFTESeq(Sequence):
            def roll(self):
                if ifte.if_.roll():
                    return ifte.then.as_sequence().roll()
                else:
                    return ifte.else_.as_sequence().roll()

            def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
                cond_prob = ifte.if_.probability()
                result = {}
                for key, value in ifte.then.as_sequence().probability_table().items():
                    result.setdefault(key, 0.0)
                    result[key] += cond_prob * value
                for key, value in ifte.else_.as_sequence().probability_table().items():
                    result.setdefault(key, 0.0)
                    result[key] += (1 - cond_prob) * value
                return result

            def constant(self) -> bool:
                return ifte.constant()

            def unevaluated_items(self) -> typing.Iterable[Expression]:
                if not ifte.if_.constant():
                    return super().unevaluated_items()
                return (
                    ifte.then.as_sequence().unevaluated_items()
                    if ifte.if_.roll()
                    else ifte.else_.as_sequence().unevaluated_items()
                )

        return IFTESeq()


class Let(Expression):
    def __init__(self, name: str, value: Expression, body: Expression) -> None:
        self.name = name
        self.value = value
        self.body = body
        self._cached_value = None

    def roll(self):
        self._cached_value = self.value.roll()
        result = self.body.roll()
        self._cached_value = None
        return result

    def constant(self) -> bool:
        return self.body.constant()

    def probability(self) -> float:
        result = 0.0
        for key, value in self.value.probability_table().items():
            self._cached_value = key
            result += self.body.probability() * value
        self._cached_value = None
        return result

    def mean(self) -> float:
        return self.body.mean()

    def min(self) -> float:
        return self.body.min()

    def max(self) -> float:
        return self.body.max()

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        result = {}
        for key, value in self.value.probability_table().items():
            self._cached_value = key
            for evaluated_key, evaluated_value in self.body.probability_table().items():
                result.setdefault(evaluated_key, 0.0)
                result[evaluated_key] += evaluated_value * value
        self._cached_value = None
        return result

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_value, value_expanded = self.value.expand()
        if value_expanded:
            self.value = expanded_value
            self.body, _ = self.body.expand()
            return self, True
        else:
            self._cached_value = expanded_value.roll()
            result = self.body.expand()
            self._cached_value = None
            return result

    def as_sequence(self) -> Sequence:
        let = self

        class LetSeq(Sequence):
            def roll(self):
                let._cached_value = let.value.roll()
                result = let.body.as_sequence().roll()
                let._cached_value = None
                return result

            def constant(self) -> bool:
                return let.constant()

            def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
                result = {}
                for key, value in let.value.probability_table().items():
                    let._cached_value = key
                    for evaluated_key, evaluated_value in (
                        let.body.as_sequence().probability_table().items()
                    ):
                        result.setdefault(evaluated_key, 0.0)
                        result[evaluated_key] += evaluated_value * value
                let._cached_value = None
                return result

            def unevaluated_items(self) -> typing.Iterable[Expression]:
                return let.body.as_sequence().unevaluated_items()

        return LetSeq()

    def __repr__(self) -> str:
        return "(let %s = %s in %s)" % (self.name, self.value, self.body)


class Var(Expression):
    def __init__(
        self, *, name: typing.Optional[str] = None, let: typing.Optional[Let] = None
    ) -> None:
        if name is None and let is not None:
            self.name = let.name
        else:
            self.name = name
        self.let = let

    def constant(self) -> bool:
        if self.let is None:
            raise DiceRollError("Unknown variable %s" % self.name)
        if self.let._cached_value is None:
            return self.let.value.constant()
        return True

    def roll(self):
        if self.let is None:
            raise DiceRollError("Unknown variable %s" % self.name)
        if self.let._cached_value is None:
            return self.let.value.roll()
        return self.let._cached_value

    def probability(self) -> float:
        if self.let is None:
            raise DiceRollError("Unknown variable %s" % self.name)
        if self.let._cached_value is None:
            return self.let.value.probability()
        if isinstance(self.let._cached_value, bool):
            return 1.0 if self.let._cached_value else 0.0
        return super().probability()

    def mean(self) -> float:
        if self.let is None:
            raise DiceRollError("Unknown variable %s" % self.name)
        if self.let._cached_value is None:
            return self.let.value.mean()
        return self.let._cached_value

    def min(self) -> float:
        if self.let is None:
            raise DiceRollError("Unknown variable %s" % self.name)
        if self.let._cached_value is None:
            return self.let.value.min()
        return self.let._cached_value

    def max(self) -> float:
        if self.let is None:
            raise DiceRollError("Unknown variable %s" % self.name)
        if self.let._cached_value is None:
            return self.let.value.max()
        return self.let._cached_value

    def probability_table_impl(self):
        if self.let is None:
            raise DiceRollError("Unknown variable %s" % self.name)
        elif self.let._cached_value is None:
            return self.let.value.probability_table()
        else:
            return {self.let._cached_value: 1.0}

    def expand(self) -> typing.Tuple["Expression", bool]:
        if self.let is None:
            raise DiceRollError("Unknown variable %s" % self.name)
        if self.let._cached_value is None:
            return self, False
        return Constant(self.let._cached_value), True

    def as_sequence(self) -> "Sequence":
        if self.let is None:
            raise DiceRollError("Unknown variable %s" % self.name)
        if self.let._cached_value is None:
            return self.let.value.as_sequence()
        if not isinstance(self.let._cached_value, tuple):
            return super().as_sequence()
        return EvaluatedSequence(*self.let._cached_value)

    def __repr__(self) -> str:
        return str(self.name)


class For(Let, Sequence):
    def __init__(
        self,
        name: str,
        values: Expression,
        where: typing.Optional[Expression] = None,
        body: typing.Optional[Expression] = None,
    ) -> None:
        this = self

        class ForValue(Expression):
            def constant(self) -> bool:
                return this.values.constant()

            def mean(self) -> float:
                return this.values.mean()

            def min(self) -> float:
                return this.values.min()

            def max(self) -> float:
                return this.values.max()

            def __repr__(self) -> str:
                return str(this.values)

        if body is None:
            body = Var(name=name, let=self)
        Let.__init__(self, name, ForValue(), body)
        self.values = values
        self.where = where

    def roll(self):
        result = []
        for item in self.value.roll():
            self._cached_value = item
            if self.where is not None and not self.where.roll():
                continue
            result.append(self.body.roll())
        self._cached_value = None
        return tuple(result)

    def as_sequence(self) -> Sequence:
        return self

    def mean(self) -> float:
        return self.body.mean()

    def min(self) -> float:
        return self.body.min()

    def max(self) -> float:
        return self.body.max()

    def probability(self) -> float:
        return Expression.probability(self)

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        def combine_probtab_list(
            items: typing.List[typing.Dict[typing.Any, float]]
        ) -> typing.Dict[tuple, float]:
            if len(items) == 0:
                return {(): 1.0}
            else:
                head = items[0]
                tail = combine_probtab_list(items[1:])
                result = {}
                for head_k, head_v in head.items():
                    for tail_k, tail_v in tail.items():
                        new_key = (head_k,) + tail_k
                        result.setdefault(new_key, 0.0)
                        result[new_key] += head_v * tail_v
                return result

        result = {}

        for values_key, values_value in (
            self.values.as_sequence().probability_table().items()
        ):
            result_probtabs: typing.List[typing.Dict[typing.Any, float]] = []
            for item in values_key:
                self._cached_value = item
                if self.where is not None and not self.where.roll():
                    continue
                result_probtabs.append(self.body.probability_table())
            for body_key, body_value in combine_probtab_list(result_probtabs).items():
                result.setdefault(body_key, 0.0)
                result[body_key] += values_value * body_value

        self._cached_value = None
        return result

    def unevaluated_items(self) -> typing.Iterable[Expression]:
        if not self.constant():
            return super().unevaluated_items()
        else:
            return EvaluatedSequence(self.roll()).unevaluated_items()

    def expand(self) -> typing.Tuple["Expression", bool]:
        self.values, value_expanded = self.values.as_sequence().expand()
        if value_expanded:
            return self, True
        else:
            items = []
            for value in self.values.roll():
                self._cached_value = value
                if self.where is not None and not self.where.roll():
                    continue
                items.append(self.body.expand()[0])
            self._cached_value = None
            return Tuple(*items), True

    def __repr__(self) -> str:
        return "(for %s in %s%s: %s)" % (
            self.name,
            self.value,
            "" if self.where is None else " where %s" % (self.where,),
            self.body,
        )
