using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Ai;
using Azul;
using AzulServer.Games;

namespace AzulServer.Hub;

/// The bot's hub account key (hub spec 5.1): it changes whenever the bot's
/// play can. `engine` is the AzulLibrary MVID: conservative (any change to the
/// compiled engine, even unrelated code, gives a new key; spec section 5).
public sealed record BotIdentity(string Key, string Name)
{
    public const string DisplayName = "Azul MCTS";

    public static BotIdentity For(AzulOptions o) =>
        For(o.BotThinkSeconds, MctsBrain.RolloutCap, MctsBrain.BrainRevision,
            typeof(Game).Assembly.ManifestModule.ModuleVersionId);

    public static BotIdentity For(double thinkSeconds, int rolloutCap, int brainRevision, Guid engineMvid)
    {
        var inputs = new SortedDictionary<string, string>(StringComparer.Ordinal)
        {
            ["brain"] = nameof(MctsBrain),
            ["brainRevision"] = brainRevision.ToString(CultureInfo.InvariantCulture),
            ["engine"] = engineMvid.ToString("N"),
            ["exploration"] = MCTS_Stochastic<Game, Move>.c.ToString("R", CultureInfo.InvariantCulture),
            ["rolloutCap"] = rolloutCap.ToString(CultureInfo.InvariantCulture),
            ["thinkSeconds"] = thinkSeconds.ToString("R", CultureInfo.InvariantCulture),
        };
        var hash = SHA256.HashData(Encoding.UTF8.GetBytes(JsonSerializer.Serialize(inputs)));
        return new BotIdentity($"mcts@{Convert.ToHexStringLower(hash)[..12]}", DisplayName);
    }
}
