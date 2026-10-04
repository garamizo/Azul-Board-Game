using System.Text.RegularExpressions;
using Azul;
using AzulServer.Games;
using AzulServer.Hub;

namespace AzulServer.Tests;

public class BotIdentityTests
{
    static readonly Guid Engine = Guid.Parse("0f8fad5b-d9cb-469f-a165-70867728950e");

    [Fact]
    public void SameInputsGiveTheSameKey() =>
        Assert.Equal(BotIdentity.For(3, 300_000, 1, Engine), BotIdentity.For(3, 300_000, 1, Engine));

    public static TheoryData<BotIdentity> Changed => new()
    {
        BotIdentity.For(2.5, 300_000, 1, Engine),
        BotIdentity.For(3, 1_000, 1, Engine),
        BotIdentity.For(3, 300_000, 2, Engine),
        BotIdentity.For(3, 300_000, 1, Guid.Parse("7c9e6679-7425-40de-944b-e07fc1f90ae7")),
    };

    [Theory]
    [MemberData(nameof(Changed))]
    public void AnyInputChangesTheKey(BotIdentity other) =>
        Assert.NotEqual(BotIdentity.For(3, 300_000, 1, Engine).Key, other.Key);

    [Fact]
    public void KeyMatchesTheHubPattern()
    {
        var id = BotIdentity.For(3, 300_000, 1, Engine);
        Assert.Matches(new Regex("^mcts@[0-9a-f]{12}$"), id.Key);
        Assert.Matches(new Regex("^[a-z0-9_.:@-]{1,64}$"), id.Key);
        Assert.Equal("Azul MCTS", id.Name);
    }

    [Fact]
    public void OptionsUseTheBrainConstantsAndTheEngineAssembly() =>
        Assert.Equal(
            BotIdentity.For(3, MctsBrain.RolloutCap, MctsBrain.BrainRevision, typeof(Game).Assembly.ManifestModule.ModuleVersionId),
            BotIdentity.For(new AzulOptions { BotThinkSeconds = 3 }));

    [Fact]
    public void TheAppRegistersItsKey()
    {
        using var app = new TestApp(new AzulOptions { BotWorkers = 0, BotThinkSeconds = 1.5 });
        Assert.Equal(BotIdentity.For(app.Options), app.Service<BotIdentity>());
    }
}
