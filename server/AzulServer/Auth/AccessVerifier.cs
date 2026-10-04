using Microsoft.IdentityModel.JsonWebTokens;
using Microsoft.IdentityModel.Tokens;

namespace AzulServer.Auth;

public enum AuthOutcome { Ok, Unauthorized, Unavailable }

public sealed record AuthResult(AuthOutcome Outcome, string? Email);

public sealed class AccessVerifier(AzulOptions options, JwksCache cache)
{
    readonly JsonWebTokenHandler handler = new();

    public async Task<AuthResult> VerifyAsync(string? token, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(token))
            return new(AuthOutcome.Unauthorized, null);
        string? kid;
        try { kid = new JsonWebToken(token).Kid; }
        catch (Exception) { return new(AuthOutcome.Unauthorized, null); }

        var lookup = await cache.LookupAsync(kid, ct);
        if (lookup == KeyLookupResult.Unavailable)
            return new(AuthOutcome.Unavailable, null);

        var result = await handler.ValidateTokenAsync(token, new TokenValidationParameters
        {
            ValidIssuer = $"https://{options.TeamDomain}",
            ValidAudience = options.Aud,
            IssuerSigningKeys = cache.Keys,
            ValidAlgorithms = [SecurityAlgorithms.RsaSha256],
            RequireSignedTokens = true,
            RequireExpirationTime = true,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            ClockSkew = TimeSpan.FromSeconds(60),
        });
        if (!result.IsValid)
            return new(AuthOutcome.Unauthorized, null);
        var email = result.Claims.TryGetValue("email", out var e) ? e as string : null;
        return string.IsNullOrWhiteSpace(email)
            ? new(AuthOutcome.Unauthorized, null)
            : new(AuthOutcome.Ok, email.Trim().ToLowerInvariant());
    }
}
