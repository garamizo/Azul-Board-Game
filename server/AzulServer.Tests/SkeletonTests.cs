using System.Net;

namespace AzulServer.Tests;

public class SkeletonTests
{
    [Fact]
    public async Task HealthAnswersWithoutIdentity()
    {
        using var app = new TestApp(new AzulOptions { BotWorkers = 0, TeamDomain = "team.example.com", Aud = "aud" });
        var res = await app.Client(email: null).GetAsync("/api/health");
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
    }

    [Fact]
    public void OptionsNeedBothAccessValuesOrNeither()
    {
        Assert.Throws<InvalidOperationException>(() => AzulOptions.FromEnvironment(k => k == "AZUL_ACCESS_AUD" ? "x" : null));
        Assert.Throws<InvalidOperationException>(() => AzulOptions.FromEnvironment(k => k == "AZUL_ACCESS_TEAM_DOMAIN" ? "x" : null));
        Assert.True(AzulOptions.FromEnvironment(_ => null).DevMode);
    }

    [Fact]
    public void TeamDomainIsNormalised()
    {
        var o = AzulOptions.FromEnvironment(k => k switch
        {
            "AZUL_ACCESS_TEAM_DOMAIN" => " https://team.cloudflareaccess.com/ ",
            "AZUL_ACCESS_AUD" => "aud",
            _ => null,
        });
        Assert.Equal("team.cloudflareaccess.com", o.TeamDomain);
        Assert.False(o.DevMode);
    }

    [Fact]
    public async Task SpaFilesAndFallback()
    {
        var root = Directory.CreateTempSubdirectory("azul-web-").FullName;
        File.WriteAllText(Path.Combine(root, "index.html"), "<!doctype html><title>Azul</title>");
        File.WriteAllText(Path.Combine(root, "app.js"), "console.log(1)");
        using var app = new TestApp(new AzulOptions { BotWorkers = 0, WebRoot = root });
        var c = app.Client();
        Assert.Equal("console.log(1)", await c.GetStringAsync("/app.js"));
        Assert.Contains("<title>Azul</title>", await c.GetStringAsync("/g/abc123"));
        Assert.Contains("<title>Azul</title>", await c.GetStringAsync("/"));
        Assert.Equal(HttpStatusCode.NotFound, (await c.GetAsync("/api/nope")).StatusCode);
    }
}
