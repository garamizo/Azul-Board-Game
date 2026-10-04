# Azul Web — Plan 1 of 4: Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the C# Azul engine safe to host many games in one server process: per-game randomness, a side-effect-free `Clone()`, a validated save format, strict move checks, legal-move hints and forced-move detection, all under xUnit tests, with the desktop pygame app still working.

**Architecture:** `AzulLibrary` becomes a `net10.0` class library; `Game` becomes a `partial` class with new files for cloning, snapshots and hints. Benchmarks move to a new `AzulBench` console project. All .NET builds and tests run in the pinned SDK container through `make`.

**Tech Stack:** .NET 10 (SDK container `mcr.microsoft.com/dotnet/sdk:10.0`, 10.0.401 on 2026-10-03), C#, xUnit (from the SDK's `xunit` template), Python 3 + pythonnet 3.2.0 (desktop smoke only).

**Spec:** `docs/superpowers/specs/2026-10-03-azul-web-design.md` (§2.1, §2.2, §3). Plans 2–4 build on this one.

## Global Constraints

Shared by all four plans; every task's requirements include them.

- Work only in the worktree `/home/garamizo/Azul-Board-Game-web` on branch `feat/web`. Never touch the main checkout.
- .NET target `net10.0`. No .NET SDK on the host: every `dotnet` command runs through `make` (the `DOTNET` variable in the `Makefile`), as the host user (`--user $(id -u):$(id -g)`), so no root-owned files appear in the checkout.
- NuGet packages cache on the host at `~/.nuget/packages`, mounted into the container.
- Engine rules stay as they are (R5), including the wall phase and the FIRST-marker-alone take. Fix only the bugs this plan names.
- Engine colours: 0 blue, 1 yellow, 2 red, 3 black, 4 white, 5 = FIRST marker. Rows 0..4 pattern lines, 5 = floor. Wall columns -1 (no completed line), 0..4, 5 = floor.
- The desktop app (`azul/`) keeps working with the new DLL path `AzulLibrary/bin/Release/net10.0`.
- Commit after every task with a message ending in the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

### Deviations from the spec found while planning (apply these, not the spec text)

1. Spec §3.3 says "in the take phase `CountPlayerClearedRound` is 0". The engine resets it only when the wall phase starts (`Logic.cs:1175`), so it stays at `NumPlayers` through the next take phase. The validator checks the range `0..NumPlayers` only.
2. Spec §3.3 says the FIRST marker is absent "only after its holder has resolved their wall turn". The validator checks "exactly one in the take phase, at most one in the wall phase", which needs no holder tracking.

## Review Focus

Inputs and conditions the spec implies that no task would otherwise test, most likely first. Each has a test in the owning task.

1. A snapshot whose JSON came from an older or hand-edited database row (missing arrays, `null` rows): `FromSnapshot` must throw `InvalidSnapshotException`, never `NullReferenceException` or `IndexOutOfRangeException` → Task 5 `NullAndRaggedArraysAreRejected`.
2. A wall move whose `colors` array is out of range (−2, 6) reaching `IsValid(int[], int[])`: must return false, not index outside `line` → Task 4 `WallColorsOutOfRangeRejected`.
3. Many games with the same seed played in parallel with MCTS bots thinking: results must not depend on thread interleaving → Task 3 `ParallelMctsSearchesDoNotInterfere`.
4. A game whose bag and discard run dry so factories stay partly empty: the centre must stay at index `numFactories` with 6 slots, and snapshots of that state must round-trip → Task 5 `ExhaustedBagRoundTrips`.
5. `ForcedMove()` called on a finished game, or in a wall phase where the forced assignment would be invalid, must return null rather than throw → Task 6 `ForcedMoveIsNullWhenFinished`.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `Makefile` | `dotnet` wrapper (SDK container), `build`, `test`, `desktop-smoke`. Plan 4 adds serve targets. |
| `.gitignore` | Adds nested `bin/`, `obj/` and later web/server scratch. |
| `Azul.slnx` | All .NET projects (replaces `Azul-Board-Game.sln`). |
| `AzulLibrary/AzulLibrary.csproj` | Class library, `net10.0`, `InternalsVisibleTo` for tests and bench, no CsvHelper. |
| `AzulLibrary/Logic.cs` | Existing rules; `Game` becomes `partial`; instance buffers; strict `IsValid`; `IsFinished`. |
| `AzulLibrary/Clone.cs` | `Game.Clone()`, private non-dealing constructor. |
| `AzulLibrary/Snapshot.cs` | `GameSnapshot`, `PlayerSnapshot`, `InvalidSnapshotException`, `ToSnapshot`, `FromSnapshot`, validation. |
| `AzulLibrary/Hints.cs` | `WallRowOption`, `LegalTakes()`, `WallOptions()`, `ForcedMove()`. |
| `AzulLibrary/Utils.cs` | `Game<M>.rng` becomes an instance field; abstract `Clone()`; benchmark code moves out. |
| `AzulLibrary/Ai.cs` | `Clone()` instead of `DeepCopier`; `c` becomes readonly. |
| `AzulLibrary/TicTacToe.cs` | Implements `Clone()`; no `DeepCopier`. |
| `AzulLibrary/DeepCopy/` | Deleted (Task 3). |
| `AzulBench/AzulBench.csproj`, `AzulBench/Program.cs`, `AzulBench/Benchmark.cs`, `AzulBench/benchmark3.csv` | The old `Test.cs` and the CsvHelper benchmark. |
| `AzulLibrary.Tests/` | xUnit project: `GameplayTests.cs`, `DeterminismTests.cs`, `CloneTests.cs`, `ValidationTests.cs`, `SnapshotTests.cs`, `HintTests.cs`, `TurnOrderTests.cs`, helpers `GameAssert.cs`, `Scenarios.cs`. |
| `azul/ai_wrapper.py`, `azul/logic_wrapper.py` | DLL path `net10.0`. |
| `tools/desktop-smoke/Dockerfile`, `tools/desktop-smoke/smoke.py` | Python + pythonnet smoke test of the desktop wrappers. |

---

### Task 1: Toolchain, library conversion, bench project, first test, desktop smoke

**Files:**
- Create: `Makefile`, `Azul.slnx` (via `dotnet new sln`), `AzulBench/AzulBench.csproj`, `AzulBench/Benchmark.cs`, `AzulLibrary.Tests/` (via `dotnet new xunit`), `AzulLibrary.Tests/GameplayTests.cs`, `tools/desktop-smoke/Dockerfile`, `tools/desktop-smoke/smoke.py`
- Move: `AzulLibrary/Test.cs` → `AzulBench/Program.cs`; `AzulLibrary/benchmark3.csv` → `AzulBench/benchmark3.csv`
- Modify: `AzulLibrary/AzulLibrary.csproj`, `AzulLibrary/Utils.cs` (remove lines 1-8 `using CsvHelper/IO/Globalization` and the `Benchmark<G, M>` class, lines 245-419), `azul/ai_wrapper.py:9`, `azul/logic_wrapper.py:9`, `.gitignore`
- Delete: `Azul-Board-Game.sln`

**Interfaces:**
- Produces: `make build`, `make test [FILTER=...]`, `make dotnet ARGS="..."`, `make desktop-smoke`. Test project `AzulLibrary.Tests` referencing `AzulLibrary` with internals visible.

- [ ] **Step 1: Write the Makefile**

```make
# Every .NET command runs in the pinned SDK container as the host user, so
# nothing is installed on the host and no root-owned files land in the tree.
SHELL := /bin/bash
UID := $(shell id -u)
GID := $(shell id -g)
SDK_IMAGE := mcr.microsoft.com/dotnet/sdk:10.0
NUGET_DIR := $(HOME)/.nuget/packages
DOTNET := docker run --rm -i --user $(UID):$(GID) \
	-e HOME=/tmp -e DOTNET_CLI_HOME=/tmp -e NUGET_PACKAGES=/nuget \
	-e DOTNET_NOLOGO=1 -e DOTNET_CLI_TELEMETRY_OPTOUT=1 -e DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 \
	-v $(CURDIR):/src -v $(NUGET_DIR):/nuget -w /src $(SDK_IMAGE) dotnet

.PHONY: dotnet build test desktop-smoke

$(NUGET_DIR):
	mkdir -p $@

dotnet: | $(NUGET_DIR)
	$(DOTNET) $(ARGS)

build: | $(NUGET_DIR)
	$(DOTNET) build Azul.slnx -c Release

test: | $(NUGET_DIR)
	$(DOTNET) test Azul.slnx $(if $(FILTER),--filter "$(FILTER)",)

desktop-smoke: | $(NUGET_DIR)
	$(DOTNET) build AzulLibrary/AzulLibrary.csproj -c Release
	docker build -q -t azul-desktop-smoke tools/desktop-smoke
	docker run --rm --user $(UID):$(GID) -e HOME=/tmp -v $(CURDIR):/src -w /src \
		azul-desktop-smoke /venv/bin/python tools/desktop-smoke/smoke.py
```

- [ ] **Step 2: Extend `.gitignore`**

Append:

```gitignore
**/bin/
**/obj/
.data/
web/node_modules/
web/dist/
web/test-results/
web/playwright-report/
web/.e2e-data/
.env.serve
```

- [ ] **Step 3: Convert the library and move the bench code**

Replace `AzulLibrary/AzulLibrary.csproj` with:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
    <TieredCompilationQuickJitForLoops>true</TieredCompilationQuickJitForLoops>
  </PropertyGroup>

  <ItemGroup>
    <InternalsVisibleTo Include="AzulLibrary.Tests" />
    <InternalsVisibleTo Include="AzulBench" />
    <InternalsVisibleTo Include="AzulServer" />
    <InternalsVisibleTo Include="AzulServer.Tests" />
  </ItemGroup>

</Project>
```

Then:

```bash
cd /home/garamizo/Azul-Board-Game-web
mkdir -p AzulBench
git mv AzulLibrary/Test.cs AzulBench/Program.cs
git mv AzulLibrary/benchmark3.csv AzulBench/benchmark3.csv
git rm -q Azul-Board-Game.sln
```

Create `AzulBench/AzulBench.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
  </PropertyGroup>

  <ItemGroup>
    <ProjectReference Include="../AzulLibrary/AzulLibrary.csproj" />
    <PackageReference Include="CsvHelper" Version="30.0.1" />
  </ItemGroup>

</Project>
```

Cut the whole `class Benchmark<G, M> ... }` block (the comment above it starting `/*\n    Evaluate one agent against standard policies` through the class's closing brace, `Utils.cs:245-419` today) out of `AzulLibrary/Utils.cs` and paste it into `AzulBench/Benchmark.cs` under this header:

```csharp
namespace GameUtils;
using System.Diagnostics;
using System.IO;
using CsvHelper;
using System.Globalization;

// Benchmark<G, M> moved here from AzulLibrary/Utils.cs (it needs CsvHelper).
```

In `AzulLibrary/Utils.cs` delete the now unused `using System.IO;`, `using CsvHelper;` and `using System.Globalization;` lines at the top.

- [ ] **Step 4: Create the solution and the test project**

```bash
cd /home/garamizo/Azul-Board-Game-web
make dotnet ARGS="new sln -n Azul --format slnx"
make dotnet ARGS="new xunit -o AzulLibrary.Tests"
rm -f AzulLibrary.Tests/UnitTest1.cs
make dotnet ARGS="add AzulLibrary.Tests/AzulLibrary.Tests.csproj reference AzulLibrary/AzulLibrary.csproj"
make dotnet ARGS="sln Azul.slnx add AzulLibrary/AzulLibrary.csproj AzulBench/AzulBench.csproj AzulLibrary.Tests/AzulLibrary.Tests.csproj"
ls Azul.slnx
grep -n '<Using Include="Xunit"' AzulLibrary.Tests/AzulLibrary.Tests.csproj
```

Expected: `Azul.slnx` exists, and the test project has a global `using Xunit` (`<Using Include="Xunit" />`). If the grep prints nothing, add `<ItemGroup><Using Include="Xunit" /></ItemGroup>` to `AzulLibrary.Tests/AzulLibrary.Tests.csproj`; the test files below rely on it.

- [ ] **Step 5: Write the first test**

`AzulLibrary.Tests/GameplayTests.cs`:

```csharp
using Azul;

namespace AzulLibrary.Tests;

public class GameplayTests
{
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    public void GreedyGameFinishes(int numPlayers)
    {
        var game = new Game(numPlayers);
        int moves = 0;
        while (!game.IsGameOver())
        {
            var move = game.GetGreedyMove();
            Assert.True(game.IsValid(move), $"greedy move {move} invalid");
            game.Play(move);
            Assert.True(++moves < 2000, "game did not end");
        }
        Assert.All(game.players, p => Assert.True(p.score >= 0));
    }
}
```

- [ ] **Step 6: Build and run the tests**

Run: `make build && make test`
Expected: build succeeds (nullable warnings are fine); `GreedyGameFinishes` passes for 2, 3 and 4 players. If `AzulBench/Program.cs` fails to compile because a type it uses is internal to the library, the `InternalsVisibleTo` entry for `AzulBench` covers it; any other error is real and must be fixed before continuing.

- [ ] **Step 7: Point the desktop wrappers at the new build**

In `azul/ai_wrapper.py` and `azul/logic_wrapper.py`, change:

```python
    sys.path.append(r"AzulLibrary/bin/Release/net7.0")
```

to:

```python
    sys.path.append(r"AzulLibrary/bin/Release/net10.0")
```

- [ ] **Step 8: Write the desktop smoke test**

`tools/desktop-smoke/Dockerfile`:

```dockerfile
# The desktop app loads AzulLibrary through pythonnet; this image has the
# .NET 10 runtime (from the SDK image) plus Python and pythonnet, no display.
FROM mcr.microsoft.com/dotnet/sdk:10.0
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /venv \
 && /venv/bin/pip install --no-cache-dir pythonnet==3.2.0
```

`tools/desktop-smoke/smoke.py`:

```python
"""Plays one greedy game through the desktop app's pythonnet wrappers."""
import sys

sys.path.insert(0, "azul")

from logic_wrapper import Game, Move  # noqa: E402  loads AzulLibrary/bin/Release/net10.0
from ai_wrapper import MCTS  # noqa: E402

game = Game(3)
tree = MCTS[Game, Move](game, 0.0)
tree.GrowWhile(0.2, 500)
assert game.IsValid(tree.GetBestAction()), "MCTS proposed an invalid move"

steps = 0
while not game.IsGameOver():
    move = game.GetGreedyMove()
    assert game.IsValid(move), f"invalid greedy move {move}"
    game.Play(move)
    steps += 1
    assert steps < 2000, "game did not end"

print("desktop smoke ok", steps, [p.score for p in game.players])
```

- [ ] **Step 9: Run the desktop smoke test**

Run: `make desktop-smoke`
Expected: last line `desktop smoke ok <steps> [<score>, <score>, <score>]`.

If it fails with a pythonnet error about a missing runtime configuration, do both of these and re-run:
1. Add `<GenerateRuntimeConfigurationFiles>true</GenerateRuntimeConfigurationFiles>` to the `PropertyGroup` of `AzulLibrary/AzulLibrary.csproj`.
2. In both wrappers replace `pythonnet.load("coreclr")` with
   `pythonnet.load("coreclr", runtime_config="AzulLibrary/bin/Release/net10.0/AzulLibrary.runtimeconfig.json")`.

- [ ] **Step 10: Commit**

```bash
git add -A Makefile .gitignore Azul.slnx AzulLibrary AzulBench AzulLibrary.Tests azul/ai_wrapper.py azul/logic_wrapper.py tools
git commit -m "build: net10 class library, bench project, xunit tests, desktop smoke

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Per-game randomness and buffers (spec §3.1)

**Files:**
- Modify: `AzulLibrary/Utils.cs:239` (`Game<M>.rng`), `AzulLibrary/Logic.cs:11` (class header), `:26-27` (static buffers), `:41-73` (constructors), `AzulLibrary/Ai.cs:185`
- Test: `AzulLibrary.Tests/DeterminismTests.cs`

**Interfaces:**
- Produces: `public Random rng` instance field on `GameUtils.Game<M>`; `public Game(int numPlayers, Random rng)` on `Azul.Game`; `public partial class Game` (later tasks add partial files).

- [ ] **Step 1: Write the failing tests**

`AzulLibrary.Tests/DeterminismTests.cs`:

```csharp
using Azul;

namespace AzulLibrary.Tests;

public class DeterminismTests
{
    internal static Game PlayGreedy(int numPlayers, int seed)
    {
        var game = new Game(numPlayers, new Random(seed));
        while (!game.IsGameOver())
            game.Play(game.GetGreedyMove());
        return game;
    }

    internal static string Fingerprint(Game g) =>
        g.ToString()
        + "|" + string.Join(",", g.players.Select(p => p.score))
        + "|" + string.Join(",", g.bag) + "|" + string.Join(",", g.discarded)
        + "|" + g.step + "|" + g.roundIdx;

    [Fact]
    public void SameSeedSameGame()
    {
        Assert.Equal(Fingerprint(PlayGreedy(3, 7)), Fingerprint(PlayGreedy(3, 7)));
    }

    [Fact]
    public void ParallelGamesMatchSequentialGames()
    {
        int[] seeds = Enumerable.Range(1, 16).ToArray();
        string[] sequential = seeds.Select(s => Fingerprint(PlayGreedy(4, s))).ToArray();
        string[] parallel = new string[seeds.Length];
        Parallel.For(0, seeds.Length, new ParallelOptions { MaxDegreeOfParallelism = 8 },
            i => parallel[i] = Fingerprint(PlayGreedy(4, seeds[i])));
        Assert.Equal(sequential, parallel);
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `make test FILTER=FullyQualifiedName~DeterminismTests`
Expected: compile error, `Game` has no constructor taking `(int, Random)`.

- [ ] **Step 3: Implement**

`AzulLibrary/Utils.cs`, in `public abstract class Game<M>`, replace

```csharp
    public static Random rng = new();
```

with

```csharp
    // Per game, never shared: games and MCTS clones run on different threads.
    public Random rng = new();
```

`AzulLibrary/Logic.cs`:
- Change `public class Game : GameUtils.Game<Move>` to `public partial class Game : GameUtils.Game<Move>`.
- Replace

```csharp
        static int[] rowIdxArray = new int[ROWS + 1];
        static int[] colorIdxArray = new int[NUM_COLORS];
```

with

```csharp
        // Shuffled in place during move generation, so each game owns its own.
        int[] rowIdxArray = { 0, 1, 2, 3, 4, 5 };
        int[] colorIdxArray = { 0, 1, 2, 3, 4 };
```

- Replace the constructor head `public Game(int numPlayers)` and its first line with a seeded constructor plus a delegating one:

```csharp
        public Game(int numPlayers) : this(numPlayers, new Random()) { }

        public Game(int numPlayers, Random rng)
        {
            this.rng = rng;  // before FillFactories, which draws from it
            this.numPlayers = numPlayers;
```

  (the rest of the existing body stays), and delete the two loops at the end of that body that reset `rowIdxArray` and `colorIdxArray` (`for (int i = 0; i < Constants.numRows + 1; i++) rowIdxArray[i] = i;` and the `colorIdxArray` loop).

`AzulLibrary/Ai.cs:185` replace `public static float c = MathF.Sqrt(2.0f);` with `public static readonly float c = MathF.Sqrt(2.0f);`.

Build. Any remaining compile error of the form "an object reference is required for the non-static field `rng`" is a static member using the old static field: change it to use a game instance (`game.rng`) or a local `new Random()`, keeping behaviour.

- [ ] **Step 4: Run the tests**

Run: `make test`
Expected: all tests pass, including `SameSeedSameGame` and `ParallelGamesMatchSequentialGames`.

- [ ] **Step 5: Commit**

```bash
git add -A AzulLibrary AzulLibrary.Tests AzulBench
git commit -m "engine: per-game RNG and shuffle buffers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Side-effect-free `Clone()` and removal of DeepCopy (spec §3.2)

**Files:**
- Create: `AzulLibrary/Clone.cs`, `AzulLibrary.Tests/GameAssert.cs`, `AzulLibrary.Tests/CloneTests.cs`
- Modify: `AzulLibrary/Utils.cs` (abstract `Clone`), `AzulLibrary/Ai.cs:5,37,89,150,217,233,289,580,596,631`, `AzulLibrary/TicTacToe.cs:2,191` (+ new `Clone`), `AzulLibrary/Logic.cs:8` (`using DeepCopy;`), `AzulBench/Program.cs:10,484`
- Delete: `AzulLibrary/DeepCopy/`

**Interfaces:**
- Consumes: `Game(int, Random)`, instance `rng` (Task 2).
- Produces: `public abstract Game<M> Clone()` on `GameUtils.Game<M>`; `public override Game Clone()` and `public Game Clone(Random rng)` on `Azul.Game`; private `Game(CloneTag)` constructor and `CloneTag` for Task 5; test helper `GameAssert.Equal(Game, Game)` and `GameAssert.SharesNothing(Game, Game)`.

- [ ] **Step 1: Write the test helper and failing tests**

`AzulLibrary.Tests/GameAssert.cs`:

```csharp
using Azul;

namespace AzulLibrary.Tests;

/// Field-by-field comparison. Game.Equals is not used: it ignores bag,
/// discard, phase and more (Logic.cs:679).
internal static class GameAssert
{
    public static void Equal(Game expected, Game actual)
    {
        Assert.Equal(expected.numPlayers, actual.numPlayers);
        Assert.Equal(expected.activePlayer, actual.activePlayer);
        Assert.Equal(expected.step, actual.step);
        Assert.Equal(expected.chanceHash, actual.chanceHash);
        Assert.Equal(expected.roundIdx, actual.roundIdx);
        Assert.Equal(expected.isRegularPhase, actual.isRegularPhase);
        Assert.Equal(expected.newRoundPlayer, actual.newRoundPlayer);
        Assert.Equal(expected.countPlayerClearedRound, actual.countPlayerClearedRound);
        Assert.Equal(expected.numFactories, actual.numFactories);
        Assert.Equal(expected.CENTER, actual.CENTER);
        Assert.Equal(expected.factories.Length, actual.factories.Length);
        for (int i = 0; i < expected.factories.Length; i++)
            Assert.Equal(expected.factories[i], actual.factories[i]);
        Assert.Equal(expected.bag, actual.bag);
        Assert.Equal(expected.discarded, actual.discarded);
        Assert.Equal(expected.players.Length, actual.players.Length);
        for (int i = 0; i < expected.players.Length; i++)
        {
            var e = expected.players[i];
            var a = actual.players[i];
            Assert.Equal(e.score, a.score);
            Assert.Equal(e.grid.Cast<int>(), a.grid.Cast<int>());
            Assert.Equal(e.line.Cast<int>(), a.line.Cast<int>());
            Assert.Equal(e.floor, a.floor);
        }
    }

    public static void SharesNothing(Game a, Game b)
    {
        Assert.NotSame(a.factories, b.factories);
        for (int i = 0; i < a.factories.Length; i++)
            Assert.NotSame(a.factories[i], b.factories[i]);
        Assert.NotSame(a.bag, b.bag);
        Assert.NotSame(a.discarded, b.discarded);
        Assert.NotSame(a.players, b.players);
        for (int i = 0; i < a.players.Length; i++)
        {
            Assert.NotSame(a.players[i], b.players[i]);
            Assert.NotSame(a.players[i].grid, b.players[i].grid);
            Assert.NotSame(a.players[i].line, b.players[i].line);
            Assert.NotSame(a.players[i].floor, b.players[i].floor);
        }
        Assert.NotSame(a.rng, b.rng);
    }
}
```

`AzulLibrary.Tests/CloneTests.cs`:

```csharp
using Ai;
using Azul;

namespace AzulLibrary.Tests;

public class CloneTests
{
    [Theory]
    [InlineData(2, 1)]
    [InlineData(3, 2)]
    [InlineData(4, 3)]
    public void CloneEqualsOriginalAfterEveryMove(int numPlayers, int seed)
    {
        var game = new Game(numPlayers, new Random(seed));
        while (!game.IsGameOver())
        {
            var clone = game.Clone();
            GameAssert.Equal(game, clone);
            GameAssert.SharesNothing(game, clone);
            game.Play(game.GetGreedyMove());
        }
        GameAssert.Equal(game, game.Clone());
    }

    [Fact]
    public void PlayingTheCloneLeavesTheOriginalAlone()
    {
        var game = new Game(3, new Random(5));
        var before = game.Clone();
        var clone = game.Clone();
        while (!clone.IsGameOver())
            clone.Play(clone.GetGreedyMove());
        GameAssert.Equal(before, game);
    }

    [Fact]
    public void CloningDoesNotAdvanceTheOriginalsRandomness()
    {
        var a = new Game(2, new Random(11));
        var b = new Game(2, new Random(11));
        var clone = a.Clone();
        while (!clone.IsGameOver())
            clone.Play(clone.GetRandomMove());
        Assert.Equal(b.rng.Next(), a.rng.Next());
    }

    [Fact]
    public void MctsStillProposesValidMoves()
    {
        var game = new Game(3, new Random(3));
        var tree = new MCTS_Stochastic<Game, Move>(game, 0f);
        for (int i = 0; i < 300; i++) tree.Grow();  // a fixed count, not a time budget
        Assert.Equal(300, tree.numRolls);
        Assert.True(game.IsValid(tree.GetBestAction()));
    }

    [Fact]
    public void ParallelMctsSearchesDoNotInterfere()
    {
        // Each search owns its game and tree; run 8 at once and check every
        // proposed move is valid for its own game.
        var games = Enumerable.Range(1, 8).Select(s => new Game(4, new Random(s))).ToArray();
        var moves = new Move[games.Length];
        Parallel.For(0, games.Length, i =>
        {
            var tree = new MCTS_Stochastic<Game, Move>(games[i], 0f);
            for (int r = 0; r < 400; r++) tree.Grow();
            moves[i] = tree.GetBestAction();
        });
        for (int i = 0; i < games.Length; i++)
            Assert.True(games[i].IsValid(moves[i]), $"game {i}: {moves[i]}");
    }

    [Fact]
    public void TicTacToeGreedyStillWorks()
    {
        var ttt = new TicTacToe.Game();
        var move = ttt.GetGreedyMove();
        Assert.True(ttt.IsValid(move));
        var copy = ttt.Clone();
        copy.Play(move);
        Assert.True(ttt.IsValid(move), "playing the clone changed the original");
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `make test FILTER=FullyQualifiedName~CloneTests`
Expected: compile error, `Game` has no method `Clone`.

- [ ] **Step 3: Implement `Clone`**

`AzulLibrary/Utils.cs`, inside `public abstract class Game<M>`, after `public abstract Game<M> Reset(int numPlayers);` add:

```csharp
    /// Deep copy with no side effects on this game (no dealing, no draws from rng).
    public abstract Game<M> Clone();
```

`AzulLibrary/Clone.cs`:

```csharp
namespace Azul
{
    using System;
    using System.Linq;

    public partial class Game
    {
        sealed class CloneTag
        {
            public static readonly CloneTag Instance = new();
        }

        // Allocates nothing game-specific and deals nothing; Clone and
        // FromSnapshot fill every field themselves.
        Game(CloneTag _) { }

        /// Deep copy; the copy gets its own fresh Random, so simulations on it
        /// never advance this game's randomness.
        public override Game Clone() => Clone(new Random());

        public Game Clone(Random random)
        {
            var g = new Game(CloneTag.Instance)
            {
                rng = random,
                activePlayer = activePlayer,
                numPlayers = numPlayers,
                step = step,
                chanceHash = chanceHash,
                numFactories = numFactories,
                CENTER = CENTER,
                roundIdx = roundIdx,
                isRegularPhase = isRegularPhase,
                newRoundPlayer = newRoundPlayer,
                countPlayerClearedRound = countPlayerClearedRound,
                factories = factories.Select(f => (int[])f.Clone()).ToArray(),
                bag = (int[])bag.Clone(),
                discarded = (int[])discarded.Clone(),
                factoryIdxArray = (int[])factoryIdxArray.Clone(),
                rowIdxArray = (int[])rowIdxArray.Clone(),
                colorIdxArray = (int[])colorIdxArray.Clone(),
                players = players.Select(p => new Player(p)).ToArray(),
            };
            return g;
        }
    }
}
```

`AzulLibrary/TicTacToe.cs`: delete `using DeepCopy;`; in `GetGreedyMove` replace `Game g = DeepCopier.Copy(this);` with `Game g = Clone();`; add to the class:

```csharp
    public override Game Clone()
    {
        var g = new Game(numPlayers) { activePlayer = activePlayer, step = step, chanceHash = chanceHash };
        g.grid = (int[,])grid.Clone();
        return g;
    }
```

`AzulLibrary/Ai.cs`: delete `using DeepCopy;  // DeepCopier (dotnet add package DeepCopy)`; replace every `DeepCopier.Copy(state)` with `(TGame)state.Clone()` and every `DeepCopier.Copy(parent.state)` with `(TGame)parent.state.Clone()` (lines 37, 89, 150, 217, 233, 289, 580, 596, 631; confirm with `grep -n DeepCopier AzulLibrary/Ai.cs` that none remain).

`AzulLibrary/Logic.cs`: delete `using DeepCopy;`.

`AzulBench/Program.cs`: delete `using DeepCopy;`; replace `gOld = DeepCopier.Copy(g);` with `gOld = g.Clone();`.

Delete the library:

```bash
git rm -rq AzulLibrary/DeepCopy
grep -rn "DeepCop" AzulLibrary AzulBench --include=*.cs | grep -v "^\S*:\s*//" || true
```

Expected: no live (uncommented) references printed.

- [ ] **Step 4: Run the tests**

Run: `make test`
Expected: all pass. Note: `TicTacToeGreedyStillWorks` exercises `TicTacToe.Clone` through `GetGreedyMove`.

- [ ] **Step 5: Commit**

```bash
git add -A AzulLibrary AzulLibrary.Tests AzulBench
git commit -m "engine: explicit Clone() replaces DeepCopy

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Strict move checks, FIRST-alone take, `IsFinished` (spec §3.4)

**Files:**
- Create: `AzulLibrary.Tests/Scenarios.cs`, `AzulLibrary.Tests/ValidationTests.cs`
- Modify: `AzulLibrary/Logic.cs` — `IsValid(int[] colIdx, int[] colors)` (`:1262-1295`), `IsValid(int factoryIdx, int color, int row)` (`:1297`), `IsValid(Move)` (`:1328`), `Play(Move)` (`:1147-1214`), plus a new `IsFinished` property; `AzulLibrary/Clone.cs` (copy `IsFinished`); `AzulLibrary.Tests/GameAssert.cs` (compare `IsFinished`)

**Interfaces:**
- Consumes: `Game(int, Random)`, `Clone()`.
- Produces: `public bool IsFinished { get; private set; }`; `Play` throws `InvalidOperationException` once finished; strict `IsValid` overloads; test helper `Scenarios.WallPhaseWithCompletedLine(int numPlayers)` returning a `Game` whose active player is in the wall phase with at least one completed line, and `Scenarios.CompletedRows(Game)`.

- [ ] **Step 1: Write the scenario helper and failing tests**

`AzulLibrary.Tests/Scenarios.cs`:

```csharp
using Azul;

namespace AzulLibrary.Tests;

internal static class Scenarios
{
    /// Rows of the active player's pattern lines that are complete.
    public static List<int> CompletedRows(Game g)
    {
        var rows = new List<int>();
        var p = g.players[g.activePlayer];
        for (int row = 0; row < 5; row++)
            for (int color = 0; color < 5; color++)
                if (p.line[row, color] > row) rows.Add(row);
        return rows;
    }

    /// Plays seeded greedy games until the active player is in the wall
    /// phase with at least one completed line.
    public static Game WallPhaseWithCompletedLine(int numPlayers)
    {
        for (int seed = 1; seed < 200; seed++)
        {
            var g = new Game(numPlayers, new Random(seed));
            while (!g.IsGameOver())
            {
                if (!g.isRegularPhase && CompletedRows(g).Count > 0)
                    return g;
                g.Play(g.GetGreedyMove());
            }
        }
        throw new InvalidOperationException("no wall-phase state with a completed line found");
    }

    public static Game FinishedGame(int numPlayers, int seed)
    {
        var g = new Game(numPlayers, new Random(seed));
        while (!g.IsGameOver())
            g.Play(g.GetGreedyMove());
        return g;
    }
}
```

`AzulLibrary.Tests/ValidationTests.cs`:

```csharp
using Azul;

namespace AzulLibrary.Tests;

public class ValidationTests
{
    static Move TakeRequest(Game g, int factory, int color, int row) =>
        new Move(new Move { factoryIdx = factory, color = color, row = row }, g);

    [Fact]
    public void WallColumnSixIsRejected()
    {
        var g = Scenarios.WallPhaseWithCompletedLine(2);
        int row = Scenarios.CompletedRows(g)[0];
        var cols = Enumerable.Repeat(-1, 5).ToArray();
        foreach (var r in Scenarios.CompletedRows(g)) cols[r] = 5;  // floor: valid
        Assert.True(g.IsValid(new Move(cols, g)));
        cols[row] = 6;
        Assert.False(g.IsValid(new Move(cols, g)));
        cols[row] = -2;
        Assert.False(g.IsValid(new Move(cols, g)));
    }

    [Fact]
    public void WallColorsOutOfRangeRejected()
    {
        var g = Scenarios.WallPhaseWithCompletedLine(2);
        var cols = Enumerable.Repeat(-1, 5).ToArray();
        var colors = new[] { 6, -1, -1, -1, -1 };
        Assert.False(g.IsValid(cols, colors));
        colors[0] = -2;
        Assert.False(g.IsValid(cols, colors));
        Assert.False(g.IsValid(new int[4], new int[5]));
    }

    [Fact]
    public void OutOfRangeTakesAreRejectedWithoutThrowing()
    {
        var g = new Game(2, new Random(1));
        Assert.False(g.IsValid(-1, 0, 0));
        Assert.False(g.IsValid(g.numFactories + 1, 0, 0));
        Assert.False(g.IsValid(0, -1, 0));
        Assert.False(g.IsValid(0, 6, 0));
        Assert.False(g.IsValid(0, 0, -1));
        Assert.False(g.IsValid(0, 0, 6));
    }

    [Fact]
    public void WrongPhaseMovesAreRejected()
    {
        var take = new Game(2, new Random(1));
        var wallMove = new Move(Enumerable.Repeat(-1, 5).ToArray(), new int[] { -1, -1, -1, -1, -1 }, take.activePlayer);
        Assert.False(take.IsValid(wallMove));

        var wall = Scenarios.WallPhaseWithCompletedLine(2);
        var takeMove = new Move(0, 0, 5, 1, false, wall.activePlayer);
        Assert.False(wall.IsValid(takeMove));
    }

    [Fact]
    public void MoveForAnotherPlayerIsRejected()
    {
        var g = new Game(2, new Random(1));
        var legal = g.GetGreedyMove();
        Assert.True(g.IsValid(legal));
        legal.playerIdx = 1 - g.activePlayer;
        Assert.False(g.IsValid(legal));
    }

    [Fact]
    public void TakingOnlyTheFirstMarkerIsLegal()
    {
        var g = new Game(2, new Random(1));
        int player = g.activePlayer;
        var move = TakeRequest(g, g.CENTER, 5, 5);
        Assert.True(move.isFirst);
        Assert.Equal(1, move.numTiles);
        Assert.True(g.IsValid(move));
        g.Play(move);
        Assert.Equal(1, g.players[player].floor[5]);
        Assert.Equal(0, g.factories[g.CENTER][5]);
        Assert.NotEqual(player, g.activePlayer);
    }

    [Fact]
    public void FirstMarkerOnlyFromTheCentreAndOnlyToTheFloor()
    {
        var g = new Game(2, new Random(1));
        Assert.False(g.IsValid(0, 5, 5));
        Assert.False(g.IsValid(g.CENTER, 5, 0));
    }

    [Theory]
    [InlineData(2, 4)]
    [InlineData(3, 5)]
    [InlineData(4, 6)]
    public void FinishedGameRefusesFurtherPlay(int numPlayers, int seed)
    {
        var g = Scenarios.FinishedGame(numPlayers, seed);
        Assert.True(g.IsFinished);
        var scores = g.players.Select(p => p.score).ToArray();
        var wallMove = new Move(Enumerable.Repeat(-1, 5).ToArray(), g);
        Assert.False(g.IsValid(wallMove));
        Assert.Throws<InvalidOperationException>(() => g.Play(wallMove));
        Assert.Equal(scores, g.players.Select(p => p.score).ToArray());
    }

    [Fact]
    public void IsFinishedOnlyAfterTheLastMove()
    {
        var g = new Game(2, new Random(9));
        while (!g.IsGameOver())
        {
            Assert.False(g.IsFinished);
            g.Play(g.GetGreedyMove());
        }
        Assert.True(g.IsFinished);
        Assert.True(g.Clone().IsFinished);
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `make test FILTER=FullyQualifiedName~ValidationTests`
Expected: compile error, `Game` has no member `IsFinished`.

- [ ] **Step 3: Implement**

In `AzulLibrary/Logic.cs`, add next to the other state fields (after `public int countPlayerClearedRound = 0;`):

```csharp
        /// Set by the Play call that scores the end of the game; Play refuses
        /// to run afterwards, so end bonuses (Player.UpdateGame) apply once.
        public bool IsFinished { get; private set; }
```

In `Play(Move m)`, add as the first statement:

```csharp
            if (IsFinished)
                throw new InvalidOperationException("game is finished");
```

and in the game-over branch replace

```csharp
                    if (IsGameOver())  // update player scores
                    {
                        for (int i = 0; i < numPlayers; i++)
                            players[i].UpdateGame();
                        return false;
                    }
```

with

```csharp
                    if (IsGameOver())  // update player scores
                    {
                        for (int i = 0; i < numPlayers; i++)
                            players[i].UpdateGame();
                        IsFinished = true;
                        return false;
                    }
```

Replace `IsValid(int[] colIdx, int[] colors)` with:

```csharp
        public bool IsValid(int[] colIdx, int[] colors)
        {
            if (colIdx.Length != Constants.numRows || colors.Length != Constants.numRows)
                return false;
            var p = players[activePlayer];
            for (int row = 0; row < Constants.numRows; row++)
            {
                // -1: no completed line, 0..4: wall column, 5: floor. Nothing else.
                if (colIdx[row] < -1 || colIdx[row] > Constants.numCols)
                    return false;
                int color = colors[row];
                if (color < -1 || color >= Constants.numColors)
                    return false;
                if (color == -1)
                {
                    if (colIdx[row] != -1)
                        return false;  // no line colour, but a target set
                    continue;
                }
                int numTiles = p.line[row, color];

                if (numTiles < row + 1)  // incomplete line
                {
                    if (colIdx[row] != -1)
                        return false;  // line incomplete, but idx set
                    continue;  // check next row
                }
                if (colIdx[row] == Constants.numCols) continue; // floor
                // line complete, but idx set out of bounds
                if (colIdx[row] < 0)
                    return false;
                // grid already filled
                if (p.grid[row, colIdx[row]] != EMPTY_TILE)
                    return false;
                // color already in same row or col
                for (int i = 0; i < Constants.numCols; i++)
                    if (p.grid[row, i] == color || p.grid[i, colIdx[row]] == color)
                        return false;
                // previous line already filled this column
                for (int i = row - 1; i >= 0; i--)
                    if (colIdx[i] == colIdx[row] && colors[i] == colors[row])
                        return false;
            }
            return true;
        }
```

In `IsValid(int factoryIdx, int color, int row)`, insert before `ref var factory = ref factories[factoryIdx];`:

```csharp
            if (factoryIdx < 0 || factoryIdx > numFactories)
                return false;
            if (color < 0 || color > (int)Tiles.FIRST_MOVE)
                return false;
            if (row < 0 || row > (int)Rows.FLOOR)
                return false;
```

Replace `IsValid(Move action)` with:

```csharp
        public override bool IsValid(Move action)
        {
            if (IsFinished || action.playerIdx != activePlayer)
                return false;
            bool isTake = action.colIdx[0] == Move.NOT_SET;
            if (isTake != isRegularPhase)
                return false;  // the phase comes from the game, not the move
            return isTake
                ? IsValid(action.factoryIdx, action.color, action.row)
                : IsValid(action.colIdx, action.colors);
        }
```

In `AzulLibrary/Clone.cs` add `IsFinished = IsFinished,` to the object initializer. In `AzulLibrary.Tests/GameAssert.cs` add `Assert.Equal(expected.IsFinished, actual.IsFinished);` after the `countPlayerClearedRound` line.

- [ ] **Step 4: Run the tests**

Run: `make test`
Expected: all pass. `GreedyGameFinishes` (Task 1) still passes, which shows greedy moves carry the right `playerIdx`.

- [ ] **Step 5: Commit**

```bash
git add -A AzulLibrary AzulLibrary.Tests
git commit -m "engine: strict move validation, IsFinished guard, wall column bound

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Versioned, validated snapshots (spec §3.3)

**Files:**
- Create: `AzulLibrary/Snapshot.cs`, `AzulLibrary.Tests/SnapshotTests.cs`

**Interfaces:**
- Consumes: `Game(CloneTag)`, `CloneTag`, `IsFinished`, private static `FactoriesVsPlayer(int)` (all inside `partial class Game`).
- Produces (Plan 2 relies on these exact names):
  - `public sealed record PlayerSnapshot(int Score, int[][] Grid, int[][] Line, int[] Floor)`
  - `public sealed record GameSnapshot(int Format, int NumPlayers, int ActivePlayer, int Step, ulong ChanceHash, int RoundIdx, bool IsRegularPhase, int NewRoundPlayer, int CountPlayerClearedRound, bool IsFinished, int[][] Factories, int[] Bag, int[] Discarded, PlayerSnapshot[] Players)` with `public const int CurrentFormat = 1`
  - `public sealed class InvalidSnapshotException : Exception`
  - `public GameSnapshot Game.ToSnapshot()`, `public static Game Game.FromSnapshot(GameSnapshot s, Random? rng = null)`

- [ ] **Step 1: Write the failing tests**

`AzulLibrary.Tests/SnapshotTests.cs`:

```csharp
using System.Text.Json;
using Azul;

namespace AzulLibrary.Tests;

public class SnapshotTests
{
    static GameSnapshot ViaJson(GameSnapshot s) =>
        JsonSerializer.Deserialize<GameSnapshot>(JsonSerializer.Serialize(s))!;

    [Theory]
    [InlineData(2, 21)]
    [InlineData(3, 22)]
    [InlineData(4, 23)]
    public void RoundTripsAfterEveryMove(int numPlayers, int seed)
    {
        var g = new Game(numPlayers, new Random(seed));
        while (true)
        {
            var restored = Game.FromSnapshot(ViaJson(g.ToSnapshot()));
            GameAssert.Equal(g, restored);
            if (g.IsGameOver()) break;
            g.Play(g.GetGreedyMove());
        }
        Assert.True(Game.FromSnapshot(g.ToSnapshot()).IsFinished);
    }

    [Fact]
    public void RestoredGameKeepsPlaying()
    {
        var g = new Game(3, new Random(4));
        for (int i = 0; i < 10; i++) g.Play(g.GetGreedyMove());
        var r = Game.FromSnapshot(g.ToSnapshot(), new Random(1));
        while (!r.IsGameOver()) r.Play(r.GetGreedyMove());
        Assert.True(r.IsFinished);
    }

    [Fact]
    public void ExhaustedBagRoundTrips()
    {
        // Move every tile from the factories, bag and discard onto player 0's
        // floor: the next deal finds nothing to draw and leaves factories empty.
        var s = new Game(4, new Random(2)).ToSnapshot();
        var floor = s.Players[0].Floor;
        for (int c = 0; c < 5; c++)
        {
            floor[c] += s.Bag[c] + s.Discarded[c];
            s.Bag[c] = 0;
            s.Discarded[c] = 0;
            foreach (var f in s.Factories) { floor[c] += f[c]; f[c] = 0; }
        }
        var g = Game.FromSnapshot(s);
        g.FillFactories();
        Assert.Equal(6, g.factories[g.CENTER].Length);
        Assert.Equal(1, g.factories[g.CENTER][5]);
        GameAssert.Equal(g, Game.FromSnapshot(ViaJson(g.ToSnapshot())));
    }

    public static TheoryData<string, Func<GameSnapshot, GameSnapshot>> Corruptions => new()
    {
        { "format", s => s with { Format = 2 } },
        { "players", s => s with { NumPlayers = 5 } },
        { "factories length", s => s with { Factories = s.Factories[..^1] } },
        { "active player", s => s with { ActivePlayer = s.NumPlayers } },
        { "negative bag", s => { var b = (int[])s.Bag.Clone(); b[0] = -1; return s with { Bag = b }; } },
        { "tile missing", s => { var b = (int[])s.Bag.Clone(); b[0] -= 1; return s with { Bag = b }; } },
        { "two colours on a line", s => {
            var p = s.Players[0];
            p.Line[1][0] = 1; p.Line[1][2] = 1;
            var b = (int[])s.Bag.Clone(); b[0] -= 1; b[2] -= 1;
            return s with { Bag = b }; } },
        { "line overfull", s => {
            var p = s.Players[0];
            p.Line[0][3] = 2;
            var b = (int[])s.Bag.Clone(); b[3] -= 2;
            return s with { Bag = b }; } },
        { "duplicate wall colour in a row", s => {
            var p = s.Players[0];
            p.Grid[0][0] = 1; p.Grid[0][1] = 1;
            var b = (int[])s.Bag.Clone(); b[1] -= 2;
            return s with { Bag = b }; } },
        { "duplicate wall colour in a column", s => {
            var p = s.Players[0];
            p.Grid[0][0] = 4; p.Grid[1][0] = 4;
            var b = (int[])s.Bag.Clone(); b[4] -= 2;
            return s with { Bag = b }; } },
        { "wall cell out of range", s => { s.Players[0].Grid[2][2] = 7; return s; } },
        { "two first markers", s => { s.Players[0].Floor[5] = 1; return s; } },
        { "finished in take phase", s => s with { IsFinished = true } },
    };

    [Theory]
    [MemberData(nameof(Corruptions))]
    public void CorruptSnapshotsAreRejected(string what, Func<GameSnapshot, GameSnapshot> corrupt)
    {
        var s = corrupt(new Game(2, new Random(1)).ToSnapshot());
        var ex = Assert.Throws<InvalidSnapshotException>(() => Game.FromSnapshot(s));
        Assert.False(string.IsNullOrEmpty(ex.Message), what);
    }

    [Fact]
    public void NullAndRaggedArraysAreRejected()
    {
        var s = new Game(2, new Random(1)).ToSnapshot();
        Assert.Throws<InvalidSnapshotException>(() => Game.FromSnapshot(s with { Bag = null! }));
        Assert.Throws<InvalidSnapshotException>(() => Game.FromSnapshot(s with { Players = null! }));
        var ragged = s.Factories.Select(f => (int[])f.Clone()).ToArray();
        ragged[0] = new int[3];
        Assert.Throws<InvalidSnapshotException>(() => Game.FromSnapshot(s with { Factories = ragged }));
        var p = s.Players[0] with { Grid = s.Players[0].Grid[..4] };
        Assert.Throws<InvalidSnapshotException>(() => Game.FromSnapshot(s with { Players = new[] { p, s.Players[1] } }));
        // A row whose JSON has "Discarded": null deserializes to a null array.
        var fromJson = JsonSerializer.Deserialize<GameSnapshot>(JsonSerializer.Serialize(s with { Discarded = null! }))!;
        Assert.Throws<InvalidSnapshotException>(() => Game.FromSnapshot(fromJson));
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `make test FILTER=FullyQualifiedName~SnapshotTests`
Expected: compile error, `GameSnapshot` not found.

- [ ] **Step 3: Implement**

`AzulLibrary/Snapshot.cs`:

```csharp
namespace Azul
{
    using System;
    using System.Linq;

    public sealed record PlayerSnapshot(int Score, int[][] Grid, int[][] Line, int[] Floor);

    /// Everything a game needs to continue, in a JSON-friendly shape (jagged
    /// arrays; System.Text.Json cannot serialize int[,]). The RNG is not part
    /// of it: a restored game draws from a fresh Random.
    public sealed record GameSnapshot(
        int Format, int NumPlayers, int ActivePlayer, int Step, ulong ChanceHash,
        int RoundIdx, bool IsRegularPhase, int NewRoundPlayer, int CountPlayerClearedRound,
        bool IsFinished, int[][] Factories, int[] Bag, int[] Discarded, PlayerSnapshot[] Players)
    {
        public const int CurrentFormat = 1;
    }

    public sealed class InvalidSnapshotException(string message) : Exception(message);

    public partial class Game
    {
        const int FIRST = 5;

        public GameSnapshot ToSnapshot() => new(
            GameSnapshot.CurrentFormat, numPlayers, activePlayer, step, chanceHash,
            roundIdx, isRegularPhase, newRoundPlayer, countPlayerClearedRound, IsFinished,
            factories.Select(f => (int[])f.Clone()).ToArray(),
            (int[])bag.Clone(), (int[])discarded.Clone(),
            players.Select(p => new PlayerSnapshot(p.score, ToJagged(p.grid), ToJagged(p.line), (int[])p.floor.Clone())).ToArray());

        /// Rebuilds a game without dealing; throws InvalidSnapshotException
        /// for any shape or rule violation.
        public static Game FromSnapshot(GameSnapshot s, Random? rng = null)
        {
            ValidateShape(s);
            int nf = FactoriesVsPlayer(s.NumPlayers);
            var g = new Game(CloneTag.Instance)
            {
                rng = rng ?? new Random(),
                numPlayers = s.NumPlayers,
                activePlayer = s.ActivePlayer,
                step = s.Step,
                chanceHash = s.ChanceHash,
                roundIdx = s.RoundIdx,
                isRegularPhase = s.IsRegularPhase,
                newRoundPlayer = s.NewRoundPlayer,
                countPlayerClearedRound = s.CountPlayerClearedRound,
                IsFinished = s.IsFinished,
                numFactories = nf,
                CENTER = nf,
                factories = s.Factories.Select(f => (int[])f.Clone()).ToArray(),
                bag = (int[])s.Bag.Clone(),
                discarded = (int[])s.Discarded.Clone(),
                factoryIdxArray = Enumerable.Range(0, nf + 1).ToArray(),
                players = s.Players.Select(p => new Player
                {
                    score = p.Score,
                    grid = To2D(p.Grid),
                    line = To2D(p.Line),
                    floor = (int[])p.Floor.Clone(),
                }).ToArray(),
            };
            g.ValidateRules();
            return g;
        }

        static int[][] ToJagged(int[,] a) =>
            Enumerable.Range(0, a.GetLength(0))
                .Select(r => Enumerable.Range(0, a.GetLength(1)).Select(c => a[r, c]).ToArray())
                .ToArray();

        static int[,] To2D(int[][] a)
        {
            var r = new int[a.Length, a[0].Length];
            for (int i = 0; i < a.Length; i++)
                for (int j = 0; j < a[i].Length; j++)
                    r[i, j] = a[i][j];
            return r;
        }

        static void Require(bool ok, string what)
        {
            if (!ok) throw new InvalidSnapshotException(what);
        }

        static bool Shape(int[]? a, int length) => a is not null && a.Length == length;

        static bool Shape(int[][]? a, int rows, int cols) =>
            a is not null && a.Length == rows && a.All(r => Shape(r, cols));

        static void ValidateShape(GameSnapshot? s)
        {
            Require(s is not null, "snapshot is null");
            Require(s!.Format == GameSnapshot.CurrentFormat, $"unknown format {s.Format}");
            Require(s.NumPlayers is >= 2 and <= 4, "numPlayers must be 2..4");
            int nf = FactoriesVsPlayer(s.NumPlayers);
            Require(s.Factories is not null && s.Factories.Length == nf + 1, "factories length");
            for (int i = 0; i < nf; i++)
                Require(Shape(s.Factories![i], Constants.numColors), $"factory {i} shape");
            Require(Shape(s.Factories![nf], Constants.numColors + 1), "centre shape");
            Require(Shape(s.Bag, Constants.numColors), "bag shape");
            Require(Shape(s.Discarded, Constants.numColors), "discard shape");
            Require(s.Players is not null && s.Players.Length == s.NumPlayers, "players length");
            foreach (var p in s.Players!)
            {
                Require(p is not null, "player is null");
                Require(Shape(p!.Grid, 5, 5), "wall shape");
                Require(Shape(p.Line, 5, 5), "pattern lines shape");
                Require(Shape(p.Floor, Constants.numColors + 1), "floor shape");
            }
            Require(s.ActivePlayer >= 0 && s.ActivePlayer < s.NumPlayers, "activePlayer out of range");
            Require(s.NewRoundPlayer >= 0 && s.NewRoundPlayer < s.NumPlayers, "newRoundPlayer out of range");
            // The engine resets this only when the wall phase starts, so it
            // stays at numPlayers through the following take phase.
            Require(s.CountPlayerClearedRound >= 0 && s.CountPlayerClearedRound <= s.NumPlayers,
                "countPlayerClearedRound out of range");
            Require(s.Step >= 0 && s.RoundIdx >= 0, "negative counters");
        }

        void ValidateRules()
        {
            foreach (var f in factories)
                Require(f.All(v => v >= 0), "negative factory count");
            Require(bag.All(v => v >= 0) && discarded.All(v => v >= 0), "negative bag or discard");

            var total = new int[Constants.numColors];
            for (int c = 0; c < Constants.numColors; c++)
                total[c] = bag[c] + discarded[c] + factories.Sum(f => f[c]);

            int firstMarkers = factories[CENTER][FIRST];
            Require(factories[CENTER][FIRST] is 0 or 1, "centre FIRST slot");
            foreach (var p in players)
            {
                Require(p.score >= 0, "negative score");
                Require(p.floor.All(v => v >= 0), "negative floor count");
                Require(p.floor[FIRST] is 0 or 1, "floor FIRST slot");
                firstMarkers += p.floor[FIRST];
                for (int c = 0; c < Constants.numColors; c++)
                    total[c] += p.floor[c];

                for (int row = 0; row < 5; row++)
                {
                    int lineColors = 0;
                    for (int c = 0; c < Constants.numColors; c++)
                    {
                        int n = p.line[row, c];
                        Require(n >= 0, "negative line count");
                        if (n == 0) continue;
                        lineColors++;
                        Require(n <= row + 1, $"line {row} overfull");
                        for (int col = 0; col < 5; col++)
                            Require(p.grid[row, col] != c, $"line {row} colour already on that wall row");
                        total[c] += n;
                    }
                    Require(lineColors <= 1, $"line {row} holds two colours");

                    for (int col = 0; col < 5; col++)
                    {
                        int v = p.grid[row, col];
                        Require(v >= -1 && v < Constants.numColors, "wall cell out of range");
                        if (v < 0) continue;
                        total[v]++;
                        for (int other = col + 1; other < 5; other++)
                            Require(p.grid[row, other] != v, "colour twice in a wall row");
                        for (int otherRow = row + 1; otherRow < 5; otherRow++)
                            Require(p.grid[otherRow, col] != v, "colour twice in a wall column");
                    }
                }
            }

            for (int c = 0; c < Constants.numColors; c++)
                Require(total[c] == Constants.totalTilesPerColor, $"colour {c} has {total[c]} tiles, not 20");

            if (isRegularPhase)
                Require(firstMarkers == 1, "take phase needs exactly one FIRST marker");
            else
                Require(firstMarkers <= 1, "more than one FIRST marker");

            if (IsFinished)
                Require(!isRegularPhase && countPlayerClearedRound == numPlayers && IsGameOver(),
                    "finished flag on a game that is not over");
        }
    }
}
```

Check: `Player` must allow object-initializer assignment of `score`, `grid`, `line`, `floor` (they are public fields, `Logic.cs:1490-1494`) and has a public parameterless constructor (`Logic.cs:1497`).

- [ ] **Step 4: Run the tests**

Run: `make test`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A AzulLibrary AzulLibrary.Tests
git commit -m "engine: versioned GameSnapshot with invariant checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Legal-move hints, forced moves, turn order (spec §3.5, §3.6)

**Files:**
- Create: `AzulLibrary/Hints.cs`, `AzulLibrary.Tests/HintTests.cs`, `AzulLibrary.Tests/TurnOrderTests.cs`

**Interfaces:**
- Consumes: strict `IsValid`, `IsFinished`, `FromSnapshot`/`ToSnapshot`.
- Produces (Plan 2 relies on these):
  - `public sealed record WallRowOption(int Color, int[] Targets)`
  - `public List<(int Factory, int Color, int Row)> Game.LegalTakes()` — empty outside the take phase
  - `public WallRowOption?[]? Game.WallOptions()` — null outside the wall phase; else 5 entries, null for rows without a completed line; targets ascending, floor (5) last
  - `public Move? Game.ForcedMove()` — the only legal move, or null

- [ ] **Step 1: Write the failing tests**

`AzulLibrary.Tests/HintTests.cs`:

```csharp
using Azul;

namespace AzulLibrary.Tests;

public class HintTests
{
    static IEnumerable<Game> States(int numPlayers, int seed)
    {
        var g = new Game(numPlayers, new Random(seed));
        while (!g.IsGameOver())
        {
            yield return g.Clone(new Random(seed));
            g.Play(g.GetGreedyMove());
        }
    }

    static IEnumerable<int[]> Expand(WallRowOption?[] opts)
    {
        IEnumerable<int[]> acc = new[] { Array.Empty<int>() };
        for (int row = 0; row < 5; row++)
        {
            var targets = opts[row]?.Targets ?? new[] { -1 };
            acc = acc.SelectMany(prefix => targets.Select(t => prefix.Append(t).ToArray()));
        }
        // The one cross-row rule: same colour, same wall column.
        return acc.Where(cols => !Enumerable.Range(0, 5).Any(i => Enumerable.Range(0, i).Any(j =>
            cols[i] is >= 0 and < 5 && cols[i] == cols[j] && opts[i]!.Color == opts[j]!.Color)));
    }

    [Theory]
    [InlineData(2, 31)]
    [InlineData(3, 32)]
    [InlineData(4, 33)]
    public void HintsMatchTheEnginesGenerators(int numPlayers, int seed)
    {
        foreach (var g in States(numPlayers, seed))
        {
            if (g.isRegularPhase)
            {
                var expected = g.GetPossibleActions().Select(m => (m.factoryIdx, m.color, m.row)).ToHashSet();
                Assert.Equal(expected, g.LegalTakes().Select(t => (t.Factory, t.Color, t.Row)).ToHashSet());
                Assert.Null(g.WallOptions());
            }
            else
            {
                var expected = g.GetColIdxMoves().Select(m => string.Join(",", m.colIdx)).ToHashSet();
                var opts = g.WallOptions()!;
                Assert.Equal(expected, Expand(opts).Select(c => string.Join(",", c)).ToHashSet());
                Assert.Empty(g.LegalTakes());
                foreach (var o in opts.Where(o => o is not null))
                    Assert.Equal(5, o!.Targets[^1]);
            }
        }
    }

    [Fact]
    public void HintsDoNotTouchTheRandomness()
    {
        foreach (var (a, b) in States(3, 40).Zip(States(3, 40)))
        {
            a.LegalTakes();
            a.WallOptions();
            a.ForcedMove();
            Assert.Equal(b.rng.Next(), a.rng.Next());
        }
    }

    [Fact]
    public void OnlyTheFirstMarkerLeftIsForced()
    {
        var s = new Game(2, new Random(1)).ToSnapshot();
        for (int f = 0; f < s.Factories.Length; f++)
            for (int c = 0; c < 5; c++) { s.Bag[c] += s.Factories[f][c]; s.Factories[f][c] = 0; }
        var g = Game.FromSnapshot(s);
        var forced = g.ForcedMove();
        Assert.NotNull(forced);
        Assert.Equal((g.CENTER, 5, 5), (forced!.factoryIdx, forced.color, forced.row));
        Assert.True(g.IsValid(forced));
    }

    [Fact]
    public void WallTurnWithNothingToPlaceIsForced()
    {
        var g = Scenarios.WallPhaseWithCompletedLine(2);
        var s = g.ToSnapshot();
        var lines = s.Players[s.ActivePlayer].Line;
        for (int row = 0; row < 5; row++)
            for (int c = 0; c < 5; c++) { s.Bag[c] += lines[row][c]; lines[row][c] = 0; }
        var empty = Game.FromSnapshot(s);
        var forced = empty.ForcedMove();
        Assert.NotNull(forced);
        Assert.All(forced!.colIdx, c => Assert.Equal(-1, c));
        Assert.True(empty.IsValid(forced));
    }

    [Fact]
    public void NormalTurnsAreNotForced()
    {
        Assert.Null(new Game(3, new Random(1)).ForcedMove());
        // A wall turn with a line that can go on the wall has a choice.
        var wall = Scenarios.WallPhaseWithCompletedLine(2);
        if (wall.WallOptions()!.Any(o => o is not null && o.Targets.Length > 1))
            Assert.Null(wall.ForcedMove());
    }

    [Fact]
    public void ForcedMoveIsNullWhenFinished()
    {
        Assert.Null(Scenarios.FinishedGame(2, 4).ForcedMove());
    }
}
```

`AzulLibrary.Tests/TurnOrderTests.cs`:

```csharp
using Azul;

namespace AzulLibrary.Tests;

public class TurnOrderTests
{
    [Theory]
    [InlineData(2, 51)]
    [InlineData(3, 52)]
    [InlineData(4, 53)]
    public void WallPhaseFollowsTheEnginesOrder(int numPlayers, int seed)
    {
        var g = new Game(numPlayers, new Random(seed));
        bool sawEmptyWallTurn = false;
        while (!g.IsGameOver())
        {
            // Take phase.
            int lastTaker = -1;
            while (g.isRegularPhase)
            {
                lastTaker = g.activePlayer;
                g.Play(g.GetGreedyMove());
            }
            Assert.Equal((lastTaker + 1) % numPlayers, g.activePlayer);
            int holder = Array.FindIndex(g.players, p => p.floor[5] == 1);
            Assert.True(holder >= 0, "someone holds the FIRST marker after the take phase");

            // Wall phase: every seat once, in turn, including seats with nothing to place.
            var seen = new List<int>();
            int round = g.roundIdx;
            for (int i = 0; i < numPlayers; i++)
            {
                Assert.False(g.isRegularPhase);
                Assert.False(g.IsFinished);
                if (Scenarios.CompletedRows(g).Count == 0)
                {
                    sawEmptyWallTurn = true;
                    var forced = g.ForcedMove();
                    Assert.NotNull(forced);
                    Assert.All(forced!.colIdx, c => Assert.Equal(-1, c));
                }
                seen.Add(g.activePlayer);
                g.Play(g.GetGreedyMove());
            }
            Assert.Equal(Enumerable.Range(0, numPlayers).ToHashSet(), seen.ToHashSet());
            if (!g.IsGameOver())
            {
                Assert.True(g.isRegularPhase);
                Assert.Equal(round + 1, g.roundIdx);
                Assert.Equal(holder, g.activePlayer);
            }
        }
        Assert.True(g.IsFinished);
        Assert.True(sawEmptyWallTurn, "seed never produced a wall turn with nothing to place; pick another seed");
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `make test FILTER="FullyQualifiedName~HintTests|FullyQualifiedName~TurnOrderTests"`
Expected: compile error, `WallRowOption` not found.

- [ ] **Step 3: Implement**

`AzulLibrary/Hints.cs`:

```csharp
namespace Azul
{
    using System.Collections.Generic;
    using System.Linq;

    /// A completed pattern line's colour and where it may go: wall columns
    /// (ascending) and 5 = floor, always last.
    public sealed record WallRowOption(int Color, int[] Targets);

    public partial class Game
    {
        /// Every legal take for the active player, deterministic and without
        /// side effects (no RNG, no shuffling). Same set as GetPossibleActions.
        public List<(int Factory, int Color, int Row)> LegalTakes()
        {
            var takes = new List<(int, int, int)>();
            if (!isRegularPhase || IsFinished)
                return takes;
            for (int f = 0; f <= numFactories; f++)
                for (int color = 0; color < Constants.numColors; color++)
                    for (int row = 0; row <= Constants.numRows; row++)
                        if (IsValid(f, color, row))
                            takes.Add((f, color, row));
            if (factories[CENTER][FIRST] > 0)
                takes.Add((CENTER, FIRST, (int)Rows.FLOOR));
            return takes;
        }

        /// Per-row wall choices for the active player; null outside the wall
        /// phase. Same per-row rule as GetColIdxMoves, without its shuffling.
        public WallRowOption?[]? WallOptions()
        {
            if (isRegularPhase || IsFinished)
                return null;
            var p = players[activePlayer];
            var result = new WallRowOption?[Constants.numRows];
            for (int row = 0; row < Constants.numRows; row++)
            {
                for (int color = 0; color < Constants.numColors; color++)
                {
                    if (p.line[row, color] <= row)
                        continue;
                    var targets = new List<int>();
                    for (int col = 0; col < Constants.numCols; col++)
                    {
                        if (p.grid[row, col] != EMPTY_TILE)
                            continue;
                        bool colorInColumn = false;
                        for (int r = 0; r < Constants.numRows; r++)
                            if (p.grid[r, col] == color) { colorInColumn = true; break; }
                        if (!colorInColumn)
                            targets.Add(col);
                    }
                    targets.Add(Constants.numCols);  // floor
                    result[row] = new WallRowOption(color, targets.ToArray());
                    break;
                }
            }
            return result;
        }

        /// The active player's only legal move, or null when there is a choice
        /// (or the game is over). Floor is always open to a completed line, so
        /// a wall turn is forced only when every completed line can go nowhere
        /// but the floor (or there is none).
        public Move? ForcedMove()
        {
            if (IsFinished)
                return null;
            if (isRegularPhase)
            {
                var takes = LegalTakes();
                if (takes.Count != 1)
                    return null;
                var t = takes[0];
                var take = new Move(t.Factory, t.Color, t.Row, this);
                return IsValid(take) ? take : null;
            }
            var options = WallOptions()!;
            if (options.Any(o => o is not null && o.Targets.Length != 1))
                return null;
            var cols = options.Select(o => o is null ? -1 : o.Targets[0]).ToArray();
            var wall = new Move(cols, this);
            return IsValid(wall) ? wall : null;
        }
    }
}
```

`FIRST` is the `const int FIRST = 5` declared in `Snapshot.cs` (same partial class).

- [ ] **Step 4: Run the tests**

Run: `make test`
Expected: all pass. If `WallPhaseFollowsTheEnginesOrder` fails only on its last assertion (`sawEmptyWallTurn`) for one seed, change that seed (try 54, 55, …) until a game includes a wall turn with no completed line; every other assertion must pass unchanged.

- [ ] **Step 5: Run the whole engine suite and the desktop smoke once more**

Run: `make test && make desktop-smoke`
Expected: all tests pass; `desktop smoke ok ...`.

- [ ] **Step 6: Commit**

```bash
git add -A AzulLibrary AzulLibrary.Tests
git commit -m "engine: deterministic legal-move hints, forced moves, turn-order tests

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
