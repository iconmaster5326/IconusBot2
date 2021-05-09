import typing
import iconusbot.roll as roll
import iconusbot.functions as functions
import os
import lark


@lark.v_args(inline=True)
class _RollParser(lark.Transformer):
    number = roll.Number
    add = roll.Add
    sub = roll.Sub
    mul = roll.Mul
    div = roll.Div
    neg = roll.Neg
    dice = roll.Dice
    die_number = lambda self, _, *args: roll.DieNumber(*args)
    die_sequence_0 = lambda self, _, *args: roll.DieSequence(roll.Tuple(*args))
    die_sequence_n = lambda self, _, *args: roll.DieSequence(roll.Tuple(*args))
    true = roll.TrueValue
    false = roll.FalseValue
    eq = roll.Eq
    ne = roll.Ne
    lt = roll.Lt
    le = roll.Le
    gt = roll.Gt
    ge = roll.Ge
    drop_worst = roll.DropWorst
    drop_best = roll.DropBest
    keep_worst = roll.KeepWorst
    keep_best = roll.KeepBest
    and_ = roll.And
    or_ = roll.Or
    not_ = roll.Not
    tuple_0 = roll.Tuple
    tuple_1 = roll.Tuple
    tuple_n = roll.Tuple
    fn_call = lambda self, name, arg: functions.resolve_function_call(name, arg)
    ifte = roll.IfThenElse
    range = roll.Range
    unpack = roll.Unpack
    count = roll.Count

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vars: typing.Dict[str, typing.List[roll.Var]] = {}

    def let(self, var: str, value: roll.Expression, body: roll.Expression):
        result = roll.Let(var, value, body)
        for var_expr in self.vars.get(var, []):
            var_expr.let = result
        self.vars.get(var, []).clear()
        return result

    def var(self, var: str):
        result = roll.Var(name=var)
        self.vars.setdefault(var, [])
        self.vars[var].append(result)
        return result


_grammar_file = os.path.join(os.path.dirname(__file__), "roll.lark")
_grammar = calc_parser = lark.Lark(open(_grammar_file), parser="lalr")


def parse(text: str) -> roll.Expression:
    try:
        return _RollParser().transform(_grammar.parse(text))
    except lark.exceptions.VisitError as e:
        raise e.orig_exc
    except lark.exceptions.ParseError as e:
        raise roll.DiceRollError("syntax error:\n```\n%s\n```" % e)
