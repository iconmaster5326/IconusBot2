import discord
from discord import activity
from iconusbot.roll import DiceRollError
import sys
import typing
import enum
import discord.ext.commands as commands
import iconusbot.roll as roll
import iconusbot.functions as roll_functions
import iconusbot.roll_parser as roll_parser
import random

client = commands.Bot(
    command_prefix="!",
    activity=discord.Game(name="!help"),
    status=discord.Status.idle,
)


@client.event
async def on_ready():
    print("We have logged in as {0.user}".format(client))


class DamageDie(enum.Enum):
    HIT = (1,)
    DOUBLE_HIT = (2,)
    MISS = (3, 4)
    SPECIAL = (5, 6)


DMG_DIE_TO_EMOJI = {
    DamageDie.HIT: "<:die1:840298111863095316>",
    DamageDie.DOUBLE_HIT: "<:die2:840298111955501086>",
    DamageDie.MISS: "<:die34:840298111884853268>",
    DamageDie.SPECIAL: "<:die56:840298112043712532>",
}

DMG_DIE_TO_DAMAGE = {
    DamageDie.HIT: 1,
    DamageDie.DOUBLE_HIT: 2,
    DamageDie.MISS: 0,
    DamageDie.SPECIAL: 1,
}

DIE_ROLL_TO_DMG_DIE = {
    1: DamageDie.HIT,
    2: DamageDie.DOUBLE_HIT,
    3: DamageDie.MISS,
    4: DamageDie.MISS,
    5: DamageDie.SPECIAL,
    6: DamageDie.SPECIAL,
}


@client.command(
    brief="fallout 2d20 damage roll",
    description="""!damage <n>

Parameters:
    n - The number of dice to roll.

Result:
    Rolls the given number of dice and produces the damage and effects dealt.
""",
)
async def damage(ctx: commands.Context, n_dice: str):
    try:
        message = ""
        damage = 0
        specials = 0
        for _ in range(int(n_dice)):
            die = DIE_ROLL_TO_DMG_DIE[random.randint(1, 6)]
            message += str(DMG_DIE_TO_EMOJI[die])
            damage += DMG_DIE_TO_DAMAGE[die]
            if die == DamageDie.SPECIAL:
                specials += 1
        message += "\nTotal: **%s** damage and **%s** effect activations!" % (
            damage,
            specials,
        )
        await ctx.send(message)

    except BaseException as e:
        try:
            await ctx.send("An internal error occured. Sorry!")
        except BaseException:
            pass
        raise e


@client.command(
    brief="fallout 2d20 skill check",
    description="""!check [<dice>] [vs <n>] [diff <n>] [risk <n>] [tag <n>]

Parameters:
    dice - The number of dice to roll. Default is 2.
    vs - The target number.
    diff - The difficulty of the check.
    risk - The chance of gaining complications. Default is 1.
    tag - Use if the skill you're checking is a tagged skill.
          Specify for <n> the value of your skill.

Result:
    Performs a skill check. Procuces success, complication,
    pass/fail, and AP information.
""",
)
async def check(ctx: commands.Context, *args_):
    args = list(args_)

    async def get_arg(name: str, default=None):
        for i, arg in tuple(enumerate(args)):
            if arg == name:
                if i < len(args) - 1:
                    result = args[i + 1]
                    del args[i + 1], args[i]
                    return result
                else:
                    await ctx.send("error: argument %s needs a value" % name)
                    raise ValueError()
        return default

    vs = int(await get_arg("vs", "-1"))
    diff = int(await get_arg("diff", "-1"))
    risk = int(await get_arg("risk", "1"))
    tag = int(await get_arg("tag", "-1"))
    if not args:
        dice = 2
    else:
        dice = int(args[0])

    dice_rolled = [random.randint(1, 20) for _ in range(dice)]
    message = "Roll: "
    crit_succs = 0
    complications = 0
    succs = 0
    first = True

    for die in dice_rolled:
        if first:
            first = False
        else:
            message += ", "

        if die == 1:
            crit_succs += 1
            succs += 1
        if die > 20 - risk:
            complications += 1
        if vs >= 0 and die <= vs:
            succs += 1
        if die != 1 and tag >= 0 and die <= tag:
            crit_succs += 1
            succs += 1

        if vs >= 0 and die <= vs:
            message += "**"
        message += str(die)
        if vs >= 0 and die <= vs:
            message += "**"

    message += "\n\n"
    if vs >= 0:
        message += "**%s** successes!\n" % succs
    if complications > 0:
        message += "**%s** complications occured.\n" % complications
    if diff >= 0:
        message += "Test **" + ("passed" if succs >= diff else "failed") + "**!\n"
        if succs > diff:
            message += "**%s** AP generated.\n" % (succs - diff)
    await ctx.send(message)


@client.command(
    name="roll",
    brief="roll dice",
    description="""!roll <expr>

Parameters:
    expr - The expression to evaluate.

Result:
    Evaluates a dice expression. You can use usual XdY notation, basic math,
    and some more unique operations. These include:
        p(X) - calculates the probability of X.
               You can use conditions in here (==, !=, >=, etc.),
               as well as boolean operations (and, or, not, etc.).
        X drop worst Y - rolls X, and drops the worst Y results.
                         also try "drop best" and "keep worst".
        
    For a list of other functions you can use, call !rollhelp.
""",
)
async def roll_(ctx: commands.Context, *args: str):
    try:
        result = roll_parser.parse(" ".join(args))
        message = "Input: %s\n" % result
        needs_further_expansion = True
        while needs_further_expansion:
            result, needs_further_expansion = result.expand()
            if needs_further_expansion:
                message += "= %s\n" % result
        await ctx.send(message + "Result: %s" % (result.roll(),))
    except DiceRollError as e:
        await ctx.send("Error in input: %s" % e.args[0])
    except BaseException as e:
        try:
            await ctx.send("An internal error occured. Sorry!")
        except BaseException:
            pass
        raise e


@client.command(
    brief="get or list functions for !roll",
    description="""!rollhelp [<fn>]

Parameters:
    fn - Optional. The function to describe.

Result:
    Prints help on a !roll function, or if no specific function was
    given, prints a list of all valid functions.
""",
)
async def rollhelp(ctx: commands.Context, *args: str):
    if not args:
        message = "```\n"
        max_namelen = max(len(x) for x in roll_functions.NAMES_TO_FUNCTIONS.keys())
        for name, fn in roll_functions.NAMES_TO_FUNCTIONS.items():
            message += (
                name + " " * (max_namelen - len(name) + 2) + fn.description() + "\n"
            )
        message += "\nType !rollhelp <name> to get help on the function <name>.```"
        await ctx.send(message)
    else:
        for arg in args:
            fn = roll_functions.NAMES_TO_FUNCTIONS.get(arg.lower())
            if fn is None:
                await ctx.send("error: function %s not found." % arg)
            else:
                await ctx.send("```\n" + fn.help() + "\n```")


def main(argv: typing.List[str] = sys.argv) -> int:
    client.run(open("token.txt").read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
