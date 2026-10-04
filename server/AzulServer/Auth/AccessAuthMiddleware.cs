namespace AzulServer.Auth;

/// Identity for every request except /api/health (spec A6). In Access mode
/// it comes only from the verified JWT; in dev mode from X-Dev-User or the
/// azul_dev_user cookie.
public sealed class AccessAuthMiddleware(RequestDelegate next, AzulOptions options, AccessVerifier verifier)
{
    public const string EmailKey = "azul.email";

    public async Task InvokeAsync(HttpContext ctx)
    {
        if (ctx.Request.Path.Equals("/api/health", StringComparison.OrdinalIgnoreCase))
        {
            await next(ctx);
            return;
        }
        string email;
        if (options.DevMode)
        {
            email = (ctx.Request.Headers["X-Dev-User"].FirstOrDefault()
                     ?? ctx.Request.Cookies["azul_dev_user"]
                     ?? "dev@localhost").Trim().ToLowerInvariant();
        }
        else
        {
            var result = await verifier.VerifyAsync(ctx.Request.Headers["Cf-Access-Jwt-Assertion"].FirstOrDefault(), ctx.RequestAborted);
            if (result.Outcome != AuthOutcome.Ok)
            {
                ctx.Response.StatusCode = result.Outcome == AuthOutcome.Unavailable ? 503 : 401;
                await ctx.Response.WriteAsJsonAsync(new { error = result.Outcome == AuthOutcome.Unavailable ? "auth-unavailable" : "unauthorized" });
                return;
            }
            email = result.Email!;
        }
        ctx.Items[EmailKey] = email;
        await next(ctx);
    }
}

public static class HttpContextIdentity
{
    public static string Email(this HttpContext ctx) =>
        ctx.Items[AccessAuthMiddleware.EmailKey] as string
        ?? throw new InvalidOperationException("no identity on this request");
}
