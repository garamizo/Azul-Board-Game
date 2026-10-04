using System.Collections.Concurrent;
using System.Threading.Channels;

namespace AzulServer.Games;

/// Version notices per game. Each subscriber holds at most one pending
/// notice (latest wins): every SSE event carries the full state.
public sealed class EventHub
{
    public const long Deleted = -1;

    readonly ConcurrentDictionary<string, ConcurrentDictionary<Guid, Channel<long>>> subscribers = new();

    public (Guid Id, ChannelReader<long> Reader) Subscribe(string gameId)
    {
        var channel = Channel.CreateBounded<long>(new BoundedChannelOptions(1)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
        });
        var id = Guid.NewGuid();
        subscribers.GetOrAdd(gameId, _ => new())[id] = channel;
        return (id, channel.Reader);
    }

    public void Unsubscribe(string gameId, Guid id)
    {
        if (subscribers.TryGetValue(gameId, out var subs))
            subs.TryRemove(id, out _);
    }

    public int SubscriberCount(string gameId) =>
        subscribers.TryGetValue(gameId, out var subs) ? subs.Count : 0;

    public void Publish(string gameId, long version)
    {
        if (!subscribers.TryGetValue(gameId, out var subs)) return;
        foreach (var channel in subs.Values)
            channel.Writer.TryWrite(version);
    }

    public void PublishDeleted(string gameId) => Publish(gameId, Deleted);
}
