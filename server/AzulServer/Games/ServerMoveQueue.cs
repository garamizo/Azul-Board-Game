using System.Collections.Concurrent;
using System.Threading.Channels;

namespace AzulServer.Games;

/// Game ids that may need a server-made move. Deduplicated (bounded by the
/// number of games), never blocks the caller. The database stays the source
/// of truth: the sweep refills this from it.
public sealed class ServerMoveQueue
{
    readonly ConcurrentDictionary<string, byte> pending = new();
    readonly Channel<string> channel = Channel.CreateUnbounded<string>();
    readonly ConcurrentDictionary<string, CancellationTokenSource> searches = new();
    readonly ConcurrentDictionary<string, byte> active = new();

    /// One worker per game at a time; a second take of a game in progress is
    /// skipped (the running worker's commit pokes the game again, and the
    /// sweep covers a move that ended in 409).
    public bool TryStart(string gameId) => active.TryAdd(gameId, 0);

    public void Finish(string gameId) => active.TryRemove(gameId, out _);

    public void Poke(string gameId)
    {
        if (pending.TryAdd(gameId, 0))
            channel.Writer.TryWrite(gameId);
    }

    public async ValueTask<string> TakeAsync(CancellationToken ct)
    {
        var id = await channel.Reader.ReadAsync(ct);
        pending.TryRemove(id, out _);
        return id;
    }

    public CancellationTokenSource BeginSearch(string gameId, CancellationToken shutdown)
    {
        var cts = CancellationTokenSource.CreateLinkedTokenSource(shutdown);
        searches[gameId] = cts;
        return cts;
    }

    public void EndSearch(string gameId, CancellationTokenSource cts)
    {
        searches.TryRemove(KeyValuePair.Create(gameId, cts));
        cts.Dispose();
    }

    /// Any change to the game (take-back, deletion) stops a search on the old version.
    public void CancelSearch(string gameId)
    {
        if (searches.TryGetValue(gameId, out var cts))
        {
            try { cts.Cancel(); }
            catch (ObjectDisposedException) { }
        }
    }
}
