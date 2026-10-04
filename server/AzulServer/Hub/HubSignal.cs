namespace AzulServer.Hub;

/// "There may be something to send": GameService wakes the sender after
/// queueing a report, so delivery does not wait for the 60 s idle cycle.
public sealed class HubSignal
{
    readonly SemaphoreSlim signal = new(0, 1);

    public void Wake()
    {
        try { signal.Release(); }
        catch (SemaphoreFullException) { }  // already awake
    }

    public Task<bool> WaitAsync(TimeSpan timeout, CancellationToken ct) => signal.WaitAsync(timeout, ct);
}
