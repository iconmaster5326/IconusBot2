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
        return "if %s then %s else %s" % (self.if_, self.then, self.else_)

    def as_sequence(self) -> Sequence:
        ifte = self

        class IFTESeq(Sequence):
            def roll(self):
                if ifte.if_.roll():
                    return ifte.then.as_sequence().roll()
                else:
                    return ifte.else_.as_sequence().roll()

            def constant(self) -> bool:
                return ifte.constant()

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

    def __repr__(self) -> str:
        return "let %s = %s in %s" % (self.name, self.value, self.body)


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
        return self.let.value.constant()

    def roll(self):
        if self.let is None:
            raise DiceRollError("Unknown variable %s" % self.name)
        if self.let._cached_value is None:
            return self.let.value.roll()
        return self.let._cached_value

    def probability(self):
        if self.let is None:
            raise DiceRollError("Unknown variable %s" % self.name)
        return self.let.value.probability()

    def mean(self):
        if self.let is None:
            raise DiceRollError("Unknown variable %s" % self.name)
        return self.let.value.mean()

    def min(self):
        if self.let is None:
            raise DiceRollError("Unknown variable %s" % self.name)
        return self.let.value.min()

    def max(self):
        if self.let is None:
            raise DiceRollError("Unknown variable %s" % self.name)
        return self.let.value.max()

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

    def __repr__(self) -> str:
        return str(self.name)
