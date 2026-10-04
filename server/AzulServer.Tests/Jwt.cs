using System.Security.Cryptography;
using AzulServer.Auth;
using Microsoft.IdentityModel.JsonWebTokens;
using Microsoft.IdentityModel.Tokens;

namespace AzulServer.Tests;

public sealed class FakeJwks : IJwksFetcher
{
    public string Json = "{\"keys\":[]}";
    public bool Fail;
    public TimeSpan Delay = TimeSpan.Zero;
    public int Calls;

    public async Task<string> FetchAsync(string teamDomain, CancellationToken ct)
    {
        Interlocked.Increment(ref Calls);
        if (Delay > TimeSpan.Zero) await Task.Delay(Delay, ct);
        if (Fail) throw new HttpRequestException("JWKS down");
        return Json;
    }
}

public static class Jwt
{
    public const string Team = "team.example.com";
    public const string Aud = "aud-1";

    public static string JwksJson(params (RSA Rsa, string Kid)[] keys) =>
        "{\"keys\":[" + string.Join(",", keys.Select(k =>
        {
            var p = k.Rsa.ExportParameters(false);
            return $"{{\"kty\":\"RSA\",\"kid\":\"{k.Kid}\",\"use\":\"sig\",\"alg\":\"RS256\"," +
                   $"\"n\":\"{Base64UrlEncoder.Encode(p.Modulus)}\",\"e\":\"{Base64UrlEncoder.Encode(p.Exponent)}\"}}";
        })) + "]}";

    public static string Token(RSA rsa, string kid, string? email = "alice@example.com",
        string iss = "https://" + Team, string aud = Aud, DateTime? expires = null,
        string alg = SecurityAlgorithms.RsaSha256)
    {
        var exp = expires ?? DateTime.UtcNow.AddMinutes(10);
        var claims = new Dictionary<string, object>();
        if (email is not null) claims["email"] = email;
        return new JsonWebTokenHandler().CreateToken(new SecurityTokenDescriptor
        {
            Issuer = iss,
            Audience = aud,
            Claims = claims,
            NotBefore = exp.AddHours(-1),
            IssuedAt = exp.AddHours(-1),
            Expires = exp,
            SigningCredentials = new SigningCredentials(new RsaSecurityKey(rsa) { KeyId = kid }, alg),
        });
    }
}
