import asyncio
import atexit
import io
import os
import shutil
import discord
from iconusbot.roll import DiceRollError
import sys
import typing
import enum
import discord.ext.commands as commands
import iconusbot.roll as roll
import iconusbot.functions as roll_functions
import iconusbot.roll_parser as roll_parser
import random
import yaml

client = commands.Bot(
    command_prefix="!",
    activity=discord.Game(name="!help"),
    status=discord.Status.idle,
)


settings: typing.Dict[str, typing.Any] = {}
userdata: typing.Dict[str, typing.Dict[str, typing.Any]] = {}


def get_userdata(
    ctx: commands.Context, key: str, default: typing.Callable[[], typing.Any]
):
    server: str = ""
    if ctx.guild is not None:
        server = ctx.guild.id
    else:
        server = ctx.author.id
    userdata.setdefault(server, {})
    if key not in userdata[server]:
        userdata[server][key] = default()
    return userdata[server][key]


def save_userdata():
    result = {}
    for server, datas in userdata.items():
        result[server] = {}
        for key, value in datas.items():
            result[server][key] = USERDATA_ON_SAVE[key](value)
    yaml.safe_dump(result, open("userdata.yaml", "w"))


def is_admin(ctx: commands.Context) -> bool:
    return ctx.author.id in settings["admins"]


@client.event
async def on_ready():
    print("We have logged in as {0.user}".format(client))


class DamageDieFace(enum.Enum):
    HIT = (1,)
    DOUBLE_HIT = (2,)
    MISS = (3, 4)
    SPECIAL = (5, 6)


DMG_DIE_TO_EMOJI = {
    DamageDieFace.HIT: "<:die1:840298111863095316>",
    DamageDieFace.DOUBLE_HIT: "<:die2:840298111955501086>",
    DamageDieFace.MISS: "<:die34:840298111884853268>",
    DamageDieFace.SPECIAL: "<:die56:840298112043712532>",
}

DMG_DIE_TO_DAMAGE = {
    DamageDieFace.HIT: 1,
    DamageDieFace.DOUBLE_HIT: 2,
    DamageDieFace.MISS: 0,
    DamageDieFace.SPECIAL: 1,
}

DIE_ROLL_TO_DMG_DIE = {
    1: DamageDieFace.HIT,
    2: DamageDieFace.DOUBLE_HIT,
    3: DamageDieFace.MISS,
    4: DamageDieFace.MISS,
    5: DamageDieFace.SPECIAL,
    6: DamageDieFace.SPECIAL,
}

DMG_DIE_TO_REROLL_WEIGHT = {
    DamageDieFace.HIT: 2,
    DamageDieFace.DOUBLE_HIT: 3,
    DamageDieFace.MISS: 1,
    DamageDieFace.SPECIAL: 3,
}


class DamageDice:
    def __init__(self, dice: int) -> None:
        self.dice = [DamageDie(self) for _ in range(dice)]
        self.refresh_info()

    def refresh_info(self):
        self.damage = sum(die.damage for die in self.dice)
        self.special = sum(die.special for die in self.dice)

    def miss_fortune(self):
        nonrerolled = [die for die in self.dice if not die.rerolled]
        if len(nonrerolled) == 0:
            return
        lowest_die = min(
            nonrerolled, key=lambda die: DMG_DIE_TO_REROLL_WEIGHT[die.face]
        )
        lowest_die.reroll(True)
        self.refresh_info()

    def __repr__(self) -> str:
        return "%s\nTotal: **%s** damage and **%s** effect activations!" % (
            "".join(str(die) for die in self.dice),
            self.damage,
            self.special,
        )


class DamageDie:
    def __init__(self, dice: DamageDice) -> None:
        self.dice = dice
        self.reroll(False)

    def reroll(self, rerolled: bool):
        self.rerolled = rerolled
        self.roll = random.randint(1, 6)
        self.face = DIE_ROLL_TO_DMG_DIE[self.roll]
        self.damage = DMG_DIE_TO_DAMAGE[self.face]
        self.special = 1 if self.face == DamageDieFace.SPECIAL else 0
        self.emoji = DMG_DIE_TO_EMOJI[self.face]

    def __repr__(self) -> str:
        result = self.emoji
        if self.rerolled:
            result = "(%s)" % result
        return result


message_to_damage: typing.Dict[int, DamageDice] = {}
FOUR_LEAF_CLOVER: str = "🍀"


@client.command(
    brief="fallout 2d20 damage roll",
    description="""!damage <n>

Parameters:
    n - The number of dice to roll.

Result:
    Rolls the given number of dice and produces the damage and effects dealt.

    React to the resulting message with 🍀 to apply Miss Fortune to one die,
    rerolling it. The reroller prioritises misses, then hits, and then finally
    double-hits and specials (with equal priority). A die has been rerolled
    already if it is surrounded in parenthesis.
""",
)
async def damage(ctx: commands.Context, *raw_n_dice: str):
    if len(raw_n_dice) != 1:
        await ctx.send("Error: expected 1 argument, got %s" % len(raw_n_dice))
        return

    try:
        n_dice = int(raw_n_dice[0])
    except ValueError:
        await ctx.send("Error: argument was not a number: %s" % raw_n_dice)
        return

    try:
        dice = DamageDice(n_dice)
        msg: discord.Message = await ctx.send(dice)
        message_to_damage[msg.id] = dice
        await msg.add_reaction(FOUR_LEAF_CLOVER)
    except BaseException as e:
        try:
            await ctx.send("An internal error occured. Sorry!")
        except BaseException:
            pass
        raise e


class Check:
    def __init__(
        self,
        vs: typing.Optional[int] = None,
        diff: typing.Optional[int] = None,
        risk: int = 1,
        tag: typing.Optional[int] = None,
        dice: int = 2,
    ) -> None:
        self.vs = vs
        self.diff = diff
        self.risk = risk
        self.tag = tag
        self.dice = [CheckDie(self) for _ in range(dice)]
        self.refresh_info()

    def refresh_info(self):
        self.complications = sum(1 for die in self.dice if die.complication)
        self.successes = (
            None
            if self.vs is None
            else sum(die.effective_successes for die in self.dice)
        )
        self.success = (
            None
            if self.diff is None or self.successes is None
            else self.successes >= self.diff
        )

    def miss_fortune(self):
        nonrerolled = [die for die in self.dice if not die.rerolled]
        if len(nonrerolled) == 0:
            return
        lowest_die = max(nonrerolled, key=lambda die: die.roll)
        lowest_die.reroll(True)
        self.refresh_info()

    def __repr__(self) -> str:
        message = "Roll: %s\n\n" % " ".join(str(die) for die in self.dice)
        if self.vs is not None:
            message += "**%s** successes!\n" % self.successes
        if self.complications > 0:
            message += "**%s** complications occured.\n" % self.complications
        if self.diff is not None:
            message += (
                "Test **"
                + ("passed" if self.successes >= self.diff else "failed")
                + "**!\n"
            )
            if self.successes > self.diff:
                message += "**%s** AP generated." % (self.successes - self.diff)
        return message


message_to_check: typing.Dict[int, Check] = {}


class CheckDie:
    def __init__(self, check: Check) -> None:
        self.check = check
        self.reroll(False)

    def reroll(self, rerolled: bool):
        self.rerolled = rerolled
        self.roll = random.randint(1, 20)
        self.complication = self.roll > 20 - self.check.risk
        self.critical = self.roll == 1 or (
            self.check.tag is not None and self.roll <= self.check.tag
        )
        self.success = False if self.check.vs is None else (self.roll <= self.check.vs)

    def __repr__(self) -> str:
        result = str(self.roll)
        if self.rerolled:
            result = "*%s*" % result
        if self.success:
            result = "**%s**" % result
        if self.critical:
            result = "__%s__" % result
        if self.complication:
            result = "~~%s~~" % result
        return result

    @property
    def effective_successes(self) -> int:
        return 0 if not self.success else (2 if self.critical else 1)


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
    Performs a skill check. Produces success, complication,
    pass/fail, and AP information.

    React to the resulting message with 🍀 to apply Miss Fortune to one die,
    rerolling it. A die has been rerolled already if it is italisized.
""",
)
async def check(ctx: commands.Context, *args_):
    args = list(args_)

    async def get_arg(name: str, default, type=lambda x: x):
        for i, arg in tuple(enumerate(args)):
            if arg == name:
                if i < len(args) - 1:
                    result = args[i + 1]
                    del args[i + 1], args[i]
                    return type(result)
                else:
                    await ctx.send("error: argument %s needs a value" % name)
                    raise ValueError()
        return default

    vs = await get_arg("vs", None, int)
    diff = await get_arg("diff", None, int)
    risk = await get_arg("risk", 1, int)
    tag = await get_arg("tag", None, int)
    if not args:
        dice = 2
    else:
        dice = int(args[0])

    check = Check(vs, diff, risk, tag, dice)
    msg: discord.Message = await ctx.send(check)
    message_to_check[msg.id] = check
    await msg.add_reaction(FOUR_LEAF_CLOVER)


@client.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    if payload.user_id == client.user.id:
        return

    # for !damage
    if payload.message_id in message_to_damage:
        e: discord.PartialEmoji = payload.emoji
        if not e.is_unicode_emoji():
            return
        if str(e) != FOUR_LEAF_CLOVER:
            return
        dice = message_to_damage[payload.message_id]
        dice.miss_fortune()
        channel = await client.fetch_channel(payload.channel_id)
        if not isinstance(channel, discord.abc.Messageable):
            return
        msg: discord.Message = await channel.fetch_message(payload.message_id)
        await msg.edit(content=dice)

    # for !check
    if payload.message_id in message_to_check:
        e: discord.PartialEmoji = payload.emoji
        if not e.is_unicode_emoji():
            return
        if str(e) != FOUR_LEAF_CLOVER:
            return
        check = message_to_check[payload.message_id]
        check.miss_fortune()
        channel = await client.fetch_channel(payload.channel_id)
        if not isinstance(channel, discord.abc.Messageable):
            return
        msg: discord.Message = await channel.fetch_message(payload.message_id)
        await msg.edit(content=check)


class APTracker(typing.DefaultDict[str, int]):
    def __init__(self, *args, **kwargs):
        super().__init__(lambda: 0, *args, **kwargs)

    @classmethod
    def on_load(cls, raw_data) -> "APTracker":
        return APTracker(raw_data)

    @classmethod
    def on_save(cls, tracker: "APTracker"):
        return dict(tracker)


@client.command(
    brief="fallout 2d20 AP counter",
    description="""!ap [<pool>] [<n>]

Parameters:
    pool - The pool to get/set AP from.
    n - Either a number, a + followed by a number,
        a - followed by a number, or `remove`.

Result:
    Keeps track of AP in the party's, the GM's, and other AP pools.

    With no arguments, returns a list of AP pools.
    With one arguments, returns a specific AP pool.
    With two arguments, sets an AP pool, increases it, decreases it, or removes
    it; depending on the input as enumerated by <n> above.
""",
)
async def ap(
    ctx: commands.Context,
    pool: typing.Optional[str] = None,
    n: typing.Optional[str] = None,
):
    tracker: APTracker = get_userdata(ctx, "ap", APTracker)

    async def print_pool(pool: str):
        await ctx.send("The pool **%s** has **%s** AP." % (pool, tracker[pool]))

    if pool is None:
        if len(tracker.keys()) == 0:
            await ctx.send("No AP pools are currently being tracked.")
        else:
            for pool in tracker.keys():
                await print_pool(pool)
        return

    if n is None:
        await print_pool(pool)
        return

    if n == "remove":
        del tracker[pool]
        await ctx.send("Pool **%s** removed." % pool)
    elif n.startswith("+"):
        tracker[pool] += int(n[1:])
        await ctx.send("Pool **%s** now has **%s** AP!" % (pool, tracker[pool]))
    elif n.startswith("-"):
        tracker[pool] -= int(n[1:])
        await ctx.send("Pool **%s** now has **%s** AP!" % (pool, tracker[pool]))
    else:
        tracker[pool] = int(n)
        await ctx.send("Pool **%s** now has **%s** AP!" % (pool, tracker[pool]))

    save_userdata()


class Initiative:
    def __init__(self, name: str, value: int, handle: typing.Optional[str]) -> None:
        self.name = name
        self.value = value
        self.handle = handle

    @property
    def handle_or_name(self) -> str:
        return self.handle if self.handle is not None else self.name

    @classmethod
    def on_load(cls, raw_data) -> "Initiative":
        return Initiative(raw_data["name"], raw_data["value"], raw_data["handle"])

    @classmethod
    def on_save(cls, init: "Initiative"):
        return {"name": init.name, "value": init.value, "handle": init.handle}


INIT_RANDOM_PREFIX = (
    "%s, you're up.",
    "%s, your turn.",
    "%s, you go.",
    "Your time to shine, %s.",
    "You're up to bat, %s.",
    "Go, %s.",
    "%s, your move.",
    "Your turn, %s.",
    "You're up, %s.",
    "Your move, %s.",
)


INIT_RANDOM_POSTFIX = (
    "Go get 'em!",
    "Kill them all!",
    "Don't mess it up!",
    "You got this!",
    "Finish them!",
    "Do great things!",
    "Choose wisely!",
    "You can do it!",
    "May the dice be in your favor!",
    "Good luck!",
    "Give 'em hell!",
)


class InitiativeTracker(typing.List[Initiative]):
    @classmethod
    def on_load(cls, raw_data) -> "InitiativeTracker":
        return InitiativeTracker(Initiative.on_load(x) for x in raw_data)

    @classmethod
    def on_save(cls, tracker: "InitiativeTracker"):
        return list(Initiative.on_save(x) for x in tracker)

    def reset(self):
        self.sort(key=lambda x: -x.value)

    def next_init(self):
        first = self.pop(0)
        self.append(first)

    def add_init(self, init: Initiative):
        for other_init in sorted(self, key=lambda x: -x.value):
            if other_init.value <= init.value:
                self.insert(self.index(other_init), init)
                return

        if len(self) == 0:
            self.append(init)
        else:
            self.insert(self.index(min(self, key=lambda x: x.value))+1, init)

    def as_table(self) -> str:
        if len(self) == 0:
            return (
                "No initatives have yet been defined."
                " Use `!init add` to add players/NPCs to initative."
            )

        result = "```\n"
        longest = max(len(x.name) for x in self)
        for init in self:
            result += ("%" + str(longest) + "s - %s\n") % (init.name, init.value)
        return result + "```"

    def whos_up(self) -> str:
        init = self[0]
        return "%s %s" % (
            random.choice(INIT_RANDOM_PREFIX) % init.handle_or_name,
            random.choice(INIT_RANDOM_POSTFIX),
        )


@client.command(
    brief="initative tracker",
    description="""!init get 

    Gets the current initative order, without pinging anyone.

!init add <name> <value> [<handle>]

    Adds a player/NPC to initative. `handle`, if provided,
    should be the @ to ping the player with when it is time
    for thier turn.

!init reset

    Sorts initative. Pings the player who should go first.

!init next

    Advances initative. Pings the player who should now go.

!init remove <name>

    Removes a player/NPC from the initative.

!init update <name> <value>

    Changes the initative of a player/NPC.

!init clear

    Deletes ALL initative entries.
""",
)
async def init(ctx: commands.Context, subcommand: str = "", *args_: str):
    tracker: InitiativeTracker = get_userdata(ctx, "init", InitiativeTracker)

    if not subcommand:
        await ctx.send(
            "error: no subcommand specified."
            " Type `!help init` for a list of valid subcommands."
        )
        return

    async def get():
        await ctx.send(tracker.as_table())

    async def add():
        if len(args_) < 2 or len(args_) > 3:
            await ctx.send(
                "error: expected 2 or 3 argumenets, got %s."
                " Type `!help init` for help with this subcommand." % len(args_)
            )

        name = args_[0]
        value = int(args_[1])
        handle = None if len(args_) == 2 else args_[2]
        tracker.add_init(Initiative(name, value, handle))
        await ctx.send("Initative is now:\n" + tracker.as_table())

    async def reset():
        if len(args_) != 0:
            await ctx.send(
                "error: expected 0 argumenets, got %s."
                " Type `!help init` for help with this subcommand." % len(args_)
            )

        tracker.reset()
        await ctx.send(tracker.as_table() + "\n" + tracker.whos_up())

    async def next():
        if len(args_) != 0:
            await ctx.send(
                "error: expected 0 argumenets, got %s."
                " Type `!help init` for help with this subcommand." % len(args_)
            )

        tracker.next_init()
        await ctx.send(tracker.as_table() + "\n" + tracker.whos_up())

    async def remove():
        if len(args_) != 1:
            await ctx.send(
                "error: expected 1 argumenet, got %s."
                " Type `!help init` for help with this subcommand." % len(args_)
            )

        name = args_[0]
        for i, init in enumerate(tuple(tracker)):
            if name.lower() == init.name.lower():
                del tracker[i]
                await ctx.send("Initative is now:\n" + tracker.as_table())
                return
        await ctx.send(
            "error: initative entry `%s` not found."
            " Type `!help init` for help with this subcommand." % name
        )

    async def update():
        if len(args_) != 2:
            await ctx.send(
                "error: expected 2 argumenets, got %s."
                " Type `!help init` for help with this subcommand." % len(args_)
            )

        name = args_[0]
        value = int(args_[1])
        for i, init in enumerate(tuple(tracker)):
            if name.lower() == init.name.lower():
                init.value = value
                del tracker[i]
                tracker.add_init(init)
                await ctx.send("Initative is now:\n" + tracker.as_table())
                return
        await ctx.send(
            "error: initative entry `%s` not found."
            " Type `!help init` for help with this subcommand." % name
        )

    async def clear():
        if len(args_) != 0:
            await ctx.send(
                "error: expected 0 argumenets, got %s."
                " Type `!help init` for help with this subcommand." % len(args_)
            )

        tracker.clear()
        await ctx.send("Initative cleared.")

    SUBCOMMANDS = {
        "get": get,
        "add": add,
        "reset": reset,
        "next": next,
        "remove": remove,
        "update": update,
        "clear": clear,
    }

    if subcommand in SUBCOMMANDS:
        await SUBCOMMANDS[subcommand]()
        save_userdata()
    else:
        await ctx.send(
            "errror: unknown subcommand `%s`."
            " Type `!help init` for a list of valid subcommands." % subcommand
        )


USERDATA_ON_LOAD: typing.Dict[str, typing.Callable[[typing.Any], typing.Any]] = {
    "ap": APTracker.on_load,
    "init": InitiativeTracker.on_load,
}
USERDATA_ON_SAVE: typing.Dict[str, typing.Callable[[typing.Any], typing.Any]] = {
    "ap": APTracker.on_save,
    "init": InitiativeTracker.on_save,
}


@client.command(
    name="roll",
    brief="roll dice",
    description="""!roll <expr>

Parameters:
    expr - The expression to evaluate.

Result:
    Evaluates a dice expression. You can use usual XdY notation, basic math,
    and some more unique operations. These include:
        d{X,Y,Z} - Roll a die with the faces X, Y, and Z.
        p(X) - calculates the probability of X.
               You can use conditions in here (==, !=, >=, etc.),
               as well as boolean operations (and, or, not, etc.).
        X drop worst Y - rolls X, and drops the worst Y results.
                         also try "drop best" and "keep worst".
        if C then X else Y - if C is true, roll X, else roll Y.
        let V = X in Y - sets the variable V to X, and then
                         evaluates Y.
        (1,2,3) - Creates a sequence of values.
        d{1,\\*(2,3),4} - The same as d{1,2,3,4}.
        #X - Get the length of the sequence or XdY expression X.
        (for x in 3d6 where x < 3: x+1) - Loop over a sequence,
                                          optionally filter with
                                          `where`, and produce a
                                          sequence as a result.
        
    For a list of other functions you can use, call !rollhelp.
""",
)
async def roll_(ctx: commands.Context, *args: str):
    try:

        async def roll_impl():
            result = roll_parser.parse(" ".join(args))
            message = "**Input:** %s\n" % result
            needs_further_expansion = True
            while needs_further_expansion:
                result, needs_further_expansion = result.expand()
                if needs_further_expansion:
                    message += "**=>** %s\n" % result
            rolled_result = result.roll()
            await ctx.send(
                message + "**Result:** %s" % (rolled_result,),
                file=discord.File(io.BytesIO(rolled_result.data), filename="image.png")
                if isinstance(rolled_result, roll.ImageResult)
                else None,
            )

        await asyncio.wait_for(roll_impl(), timeout=settings["timeout"])
    except TimeoutError:
        await ctx.send("Your roll took too long to evaluate. Sorry!")
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
        for name, fn in sorted(roll_functions.NAMES_TO_FUNCTIONS.items()):
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
    if not os.path.exists("settings.yaml"):
        shutil.copy(
            os.path.join(os.path.dirname(__file__), "settings.default.yaml"),
            "settings.yaml",
        )
        print(
            "settings.yaml not detected!"
            " A default one has been provided."
            " Please edit that file and re-run this program."
        )
        return 1

    global settings
    settings = yaml.safe_load(open("settings.yaml"))

    global userdata
    if os.path.exists("userdata.yaml"):
        for server, raw_userdata in yaml.safe_load(open("userdata.yaml")).items():
            userdata[server] = {}
            for key, value in raw_userdata.items():
                userdata[server][key] = USERDATA_ON_LOAD[key](value)

    atexit.register(save_userdata)

    client.run(settings["token"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
