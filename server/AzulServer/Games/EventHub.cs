namespace AzulServer.Games;

public sealed class EventHub
{
    public const long Deleted = -1;

    public void Publish(string gameId, long version) { }

    public void PublishDeleted(string gameId) => Publish(gameId, Deleted);
}
