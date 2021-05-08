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
    dice_roll = lambda self, lhs, _, rhs: roll.DiceRoll(lhs, rhs)
    true = roll.TrueValue
    false = roll.FalseValue
    probability_of = lambda self, _, arg: functions.ProbabilityOf(arg)
    eq = roll.Eq
    ne = roll.Ne
    lt = roll.Lt
    le = roll.Le
    gt = roll.Gt
    ge = roll.Ge
    probtab = functions.ProbTab
    drop_worst = roll.DropWorst
    drop_best = roll.DropBest
    keep_worst = roll.KeepWorst
    keep_best = roll.KeepBest
    mean = functions.Mean
    and_ = roll.And
    or_ = roll.Or
    not_ = roll.Not
    min = functions.Min
    max = functions.Max
    tuple_0 = roll.Tuple
    tuple_1 = roll.Tuple
    tuple_n = roll.Tuple
    fn_call = lambda self, name, arg: functions.resolve_function_call(name, arg)
    ifte = roll.IfThenElse


_grammar_file = os.path.join(os.path.dirname(__file__), "roll.lark")
_grammar = calc_parser = lark.Lark(open(_grammar_file), parser="lalr")


def parse(text: str) -> roll.Expression:
    try:
        return _RollParser().transform(_grammar.parse(text))
    except lark.exceptions.VisitError as e:
        raise e.orig_exc
    except lark.exceptions.ParseError as e:
        raise roll.DiceRollError("syntax error:\n```\n%s\n```" % e)
