import random
import typing


class _Number(float):
    def __repr__(self) -> str:
        result = f"{float(self):.2f}"
        if result.endswith(".00"):
            result = result[:-3]
        return result


class DiceRollError(ValueError):
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
                for key, value in self.probability_table():
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
        raise DiceRollError("'%s' is not a sequence" % self)

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
    def as_sequence(self) -> "Sequence":
        return self

    def mean(self) -> float:
        try:
            if self.constant():
                return _mean(self.roll())
            else:
                result = 0.0
                for key, value in self.probability_table():
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


class Number(Expression):
    def __init__(self, value):
        self.value = float(value)

    def roll(self):
        return _Number(self.value)

    def __repr__(self):
        return str(int(self.value) if int(self.value) == self.value else self.value)


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


class Tuple(Sequence):
    def __init__(self, *args: Expression) -> None:
        self.args = tuple(args)

    def roll(self):
        return tuple(arg.roll() for arg in self.args)

    def __repr__(self):
        return "(" + ", ".join(str(arg) for arg in self.args) + ")"


class BiMathOp(Expression):
    def op(self, lhs: float, rhs: float) -> float:
        raise NotImplementedError

    def __init__(self, lhs: Expression, rhs: Expression):
        self.lhs = lhs
        self.rhs = rhs

    def roll(self):
        return _Number(self.op(self.lhs.roll(), self.rhs.roll()))

    def constant(self) -> bool:
        return self.lhs.constant() and self.rhs.constant()

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_lhs, lhs_expanded = self.lhs.expand()
        expanded_rhs, rhs_expanded = self.rhs.expand()
        return self.__class__(expanded_lhs, expanded_rhs), lhs_expanded or rhs_expanded

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        if self.lhs.constant():
            value = self.lhs.roll()
            return {
                self.op(k, value): v for k, v in self.rhs.probability_table().items()
            }
        elif self.rhs.constant():
            value = self.rhs.roll()
            return {
                self.op(k, value): v for k, v in self.lhs.probability_table().items()
            }
        else:
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
        return self.op(self.lhs.mean(), self.rhs.mean())

    def min(self) -> float:
        return self.op(self.lhs.min(), self.rhs.min())

    def max(self) -> float:
        return self.op(self.lhs.max(), self.rhs.max())


class Add(BiMathOp):
    def op(self, lhs: float, rhs: float) -> float:
        return lhs + rhs

    def __repr__(self):
        return "%s + %s" % (self.lhs, self.rhs)


class Sub(BiMathOp):
    def op(self, lhs: float, rhs: float) -> float:
        return lhs - rhs

    def __repr__(self):
        return "%s - %s" % (self.lhs, self.rhs)


class Mul(BiMathOp):
    def op(self, lhs: float, rhs: float) -> float:
        return lhs * rhs

    def __repr__(self):
        return "%s * %s" % (self.lhs, self.rhs)


class Div(BiMathOp):
    def op(self, lhs: float, rhs: float) -> float:
        return lhs / rhs

    def __repr__(self):
        return "%s / %s" % (self.lhs, self.rhs)


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


def _get_die_probabilities(n_dice: int, n_faces: int) -> typing.Dict[int, float]:
    if n_dice == 1:
        return {face: 1.0 / n_faces for face in range(1, n_faces + 1)}
    else:
        result_minus1 = _get_die_probabilities(n_dice - 1, n_faces)
        result = {}
        for total, prob in result_minus1.items():
            for face in range(1, n_faces + 1):
                new_key = total + face
                new_value = prob * 1.0 / n_faces

                result.setdefault(new_key, 0)
                result[new_key] += new_value
        return result


class DiceRoll(Expression):
    def __init__(self, lhs: Expression, rhs: Expression):
        self.lhs = lhs
        self.rhs = rhs

    def roll(self):
        n_dice = int(self.lhs.roll())
        die_size = int(self.rhs.roll())
        result = 0
        for _ in range(n_dice):
            result += random.randint(1, die_size)
        return _Number(result)

    def constant(self) -> bool:
        return False

    def probability_table_impl(self) -> typing.Dict[typing.Any, float]:
        if self.lhs.constant() and self.rhs.constant():
            return _get_die_probabilities(int(self.lhs.roll()), int(self.rhs.roll()))
        else:
            table1 = self.lhs.probability_table()
            table2 = self.rhs.probability_table()
            result: typing.Dict[typing.Any, float] = {}
            for key1, value1 in table1.items():
                for key2, value2 in table2.items():
                    for key3, value3 in _get_die_probabilities(key1, key2).items():
                        result.setdefault(key3, 0.0)
                        result[key3] += value1 * value2 * value3
            return result

    def as_sequence(self) -> Sequence:
        dice_roll = self

        class DiceRollSeq(Sequence):
            def roll(self):
                return tuple(
                    DiceRoll(Number(1), dice_roll.rhs).roll()
                    for _ in range(int(dice_roll.lhs.roll()))
                )

            def constant(self) -> bool:
                return False

        return DiceRollSeq()

    def expand(self) -> typing.Tuple["Expression", bool]:
        expanded_lhs, lhs_expanded = self.lhs.expand()
        expanded_rhs, rhs_expanded = self.rhs.expand()
        if lhs_expanded or rhs_expanded:
            return DiceRoll(expanded_lhs, expanded_rhs), lhs_expanded or rhs_expanded
        else:
            zero = Number(0)
            result = zero
            for die in (
                DiceRoll(Number(1), self.rhs) for _ in range(int(self.lhs.roll()))
            ):
                if result == zero:
                    result = Number(die.roll())
                else:
                    result = Add(result, Number(die.roll()))
            return result, True

    def __repr__(self):
        return "%sd%s" % (self.lhs, self.rhs)

    def mean(self) -> float:
        if self.lhs.constant() and self.rhs.constant():
            return ((self.rhs.roll() + 1) / 2) * self.lhs.roll()
        else:
            return super().mean()

    def min(self) -> float:
        return self.lhs.min()

    def max(self) -> float:
        return self.lhs.max() * self.rhs.max()


class BinCompOp(Expression):
    def op(self, lhs, rhs) -> bool:
        raise NotImplementedError

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


class Eq(BinCompOp):
    def op(self, lhs, rhs) -> bool:
        return lhs == rhs

    def __repr__(self):
        return "%s == %s" % (self.lhs, self.rhs)

    def probability(self) -> float:
        if self.constant():
            return 1.0 if self.op(self.lhs.roll(), self.rhs.roll()) else 0.0
        elif self.lhs.constant():
            return self.rhs.probability_table()[self.lhs.roll()]
        elif self.rhs.constant():
            return self.lhs.probability_table()[self.rhs.roll()]
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
    def op(self, lhs, rhs) -> bool:
        return lhs != rhs

    def __repr__(self):
        return "%s != %s" % (self.lhs, self.rhs)


class Le(BinCompOp):
    def op(self, lhs, rhs) -> bool:
        return lhs <= rhs

    def __repr__(self):
        return "%s <= %s" % (self.lhs, self.rhs)


class Ge(BinCompOp):
    def op(self, lhs, rhs) -> bool:
        return lhs >= rhs

    def __repr__(self):
        return "%s >= %s" % (self.lhs, self.rhs)


class Lt(BinCompOp):
    def op(self, lhs, rhs) -> bool:
        return lhs < rhs

    def __repr__(self):
        return "%s < %s" % (self.lhs, self.rhs)


class Gt(BinCompOp):
    def op(self, lhs, rhs) -> bool:
        return lhs > rhs

    def __repr__(self):
        return "%s > %s" % (self.lhs, self.rhs)


class EvaluatedSequence(Sequence):
    def __init__(self, *args):
        self.args = tuple(args)

    def roll(self):
        return self.args

    def __repr__(self):
        return "(" + ", ".join(str(arg) for arg in self.args) + ")"


class DropKeepOp(Expression):
    def __init__(self, lhs: Expression, rhs: Expression):
        self.lhs = lhs
        self.rhs = rhs

    def constant(self) -> bool:
        return self.lhs.constant() and self.rhs.constant()

    def expand(self) -> typing.Tuple["Expression", bool]:
        _, lhs_expanded = self.lhs.expand()
        expanded_rhs, rhs_expanded = self.rhs.expand()
        return (
            self.__class__(
                EvaluatedSequence(*self.lhs.as_sequence().roll()), expanded_rhs
            ),
            lhs_expanded or rhs_expanded,
        )


class DropWorst(DropKeepOp):
    def roll(self):
        values = list(self.lhs.as_sequence().roll())
        values.sort()
        n_to_drop = int(self.rhs.roll())
        return _Number(sum(values[n_to_drop:]))

    def __repr__(self):
        return "%s drop worst %s" % (self.lhs, self.rhs)


class DropBest(DropKeepOp):
    def roll(self):
        values = list(self.lhs.as_sequence().roll())
        values.sort()
        n_to_drop = int(self.rhs.roll())
        return _Number(sum(values[:-n_to_drop]))

    def __repr__(self):
        return "%s drop best %s" % (self.lhs, self.rhs)


class KeepWorst(DropKeepOp):
    def roll(self):
        values = list(self.lhs.as_sequence().roll())
        values.sort()
        n_to_keep = int(self.rhs.roll())
        return _Number(sum(values[:n_to_keep]))

    def __repr__(self):
        return "%s keep worst %s" % (self.lhs, self.rhs)


class KeepBest(DropKeepOp):
    def roll(self):
        values = list(self.lhs.as_sequence().roll())
        values.sort()
        n_to_keep = int(self.rhs.roll())
        return _Number(sum(values[-n_to_keep:]))

    def __repr__(self):
        return "%s keep best %s" % (self.lhs, self.rhs)


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
        return "%s and %s" % (self.lhs, self.rhs)


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
        return "%s or %s" % (self.lhs, self.rhs)


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
        return "not %s" % self.lhs
