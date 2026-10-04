namespace AzulServer.Auth;

/// State-changing requests must be JSON (forces a CORS preflight, which this
/// server never grants) and, when the browser sends Origin, come from us.
public sealed class CsrfMiddleware(RequestDelegate next, AzulOptions options)
{
    public async Task InvokeAsync(HttpContext ctx)
    {
        var method = ctx.Request.Method;
        bool changes = HttpMethods.IsPost(method) || HttpMethods.IsPut(method)
                       || HttpMethods.IsPatch(method) || HttpMethods.IsDelete(method);
        if (changes)
        {
            if (!HttpMethods.IsDelete(method)
                && ctx.Request.ContentType?.StartsWith("application/json", StringComparison.OrdinalIgnoreCase) != true)
            {
                await Refuse(ctx, "json-required");
                return;
            }
            var origin = ctx.Request.Headers.Origin.FirstOrDefault();
            if (origin is not null && !Allowed(origin, ctx))
            {
                await Refuse(ctx, "bad-origin");
                return;
            }
        }
        await next(ctx);
    }

    bool Allowed(string origin, HttpContext ctx)
    {
        if (options.PublicOrigin is { } pub && string.Equals(origin.TrimEnd('/'), pub, StringComparison.OrdinalIgnoreCase))
            return true;
        return Uri.TryCreate(origin, UriKind.Absolute, out var uri)
               && string.Equals(uri.Authority, ctx.Request.Host.Value, StringComparison.OrdinalIgnoreCase);
    }

    static async Task Refuse(HttpContext ctx, string code)
    {
        ctx.Response.StatusCode = StatusCodes.Status403Forbidden;
        await ctx.Response.WriteAsJsonAsync(new { error = code });
    }
}
