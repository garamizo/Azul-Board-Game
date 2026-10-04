using AzulServer.Auth;
using AzulServer.Games;

namespace AzulServer.Api;

public static class EventStream
{
    public static async Task Handle(HttpContext ctx, string id, GameService games, EventHub hub, AzulOptions options, TimeProvider time)
    {
        var viewer = ctx.Email();
        // Subscribe before reading the first snapshot, so no version can slip
        // between the two; the client ignores versions it already has.
        var (subscription, reader) = hub.Subscribe(id);
        try
        {
            var first = games.GetView(id, viewer);
            if (first is null)
            {
                ctx.Response.StatusCode = StatusCodes.Status404NotFound;
                return;
            }
            ctx.Response.Headers.ContentType = "text/event-stream";
            ctx.Response.Headers.CacheControl = "no-cache";
            ctx.Response.Headers["X-Accel-Buffering"] = "no";
            long sent = first.Version;
            await WriteState(ctx, first);

            var ct = ctx.RequestAborted;
            var until = time.GetUtcNow() + TimeSpan.FromMinutes(options.SseMaxMinutes);
            var heartbeat = TimeSpan.FromSeconds(options.SseHeartbeatSeconds);
            while (!ct.IsCancellationRequested && time.GetUtcNow() < until)
            {
                bool hasNotice;
                using (var wait = CancellationTokenSource.CreateLinkedTokenSource(ct))
                {
                    wait.CancelAfter(heartbeat);
                    try
                    {
                        hasNotice = await reader.WaitToReadAsync(wait.Token);
                    }
                    catch (OperationCanceledException) when (!ct.IsCancellationRequested)
                    {
                        // Heartbeat. Also catch up if a notification was lost
                        // after a commit (GameService.AfterCommit failed).
                        var current = games.GetView(id, viewer);
                        if (current is null)
                        {
                            await WriteDeleted(ctx);
                            return;
                        }
                        if (current.Version > sent)
                        {
                            sent = current.Version;
                            await WriteState(ctx, current);
                        }
                        else
                        {
                            await ctx.Response.WriteAsync(": ping\n\n", ct);
                            await ctx.Response.Body.FlushAsync(ct);
                        }
                        continue;
                    }
                }
                if (!hasNotice) return;
                while (reader.TryRead(out var version))
                {
                    if (version == EventHub.Deleted)
                    {
                        await WriteDeleted(ctx);
                        return;
                    }
                    if (version <= sent) continue;
                    var view = games.GetView(id, viewer);
                    if (view is null)
                    {
                        await WriteDeleted(ctx);
                        return;
                    }
                    if (view.Version > sent)
                    {
                        sent = view.Version;
                        await WriteState(ctx, view);
                    }
                }
            }
        }
        catch (OperationCanceledException) when (ctx.RequestAborted.IsCancellationRequested)
        {
            // client went away
        }
        finally
        {
            hub.Unsubscribe(id, subscription);
        }
    }

    static async Task WriteState(HttpContext ctx, GameView view)
    {
        await ctx.Response.WriteAsync($"event: state\nid: {view.Version}\ndata: {Json.Serialize(view)}\n\n", ctx.RequestAborted);
        await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
    }

    static async Task WriteDeleted(HttpContext ctx)
    {
        await ctx.Response.WriteAsync("event: deleted\ndata: {}\n\n", ctx.RequestAborted);
        await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
    }
}
