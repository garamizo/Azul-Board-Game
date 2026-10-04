using AzulServer.Hub;

namespace AzulServer.Tests;

public class HubContractTests
{
    [Fact]
    public void HubSettingsComeFromTheEnvironment()
    {
        var env = new Dictionary<string, string>
        {
            ["AZUL_HUB_URL"] = "http://playhub:3000/",
            ["AZUL_HUB_KEY"] = "phk_x",
            ["AZUL_HUB_PUBLIC_URL"] = "https://play.example/",
        };
        var o = AzulOptions.FromEnvironment(k => env.GetValueOrDefault(k));
        Assert.Equal("http://playhub:3000", o.Hub.Url);
        Assert.Equal("phk_x", o.Hub.Key);
        Assert.Equal("https://play.example", o.Hub.PublicUrl);
        Assert.True(o.Hub.SenderConfigured);
        Assert.False(o.Hub.HalfConfigured);
    }

    [Fact]
    public void OneHalfOfTheSenderConfigIsReported()
    {
        var o = AzulOptions.FromEnvironment(k => k == "AZUL_HUB_URL" ? "http://h" : null);
        Assert.False(o.Hub.SenderConfigured);
        Assert.True(o.Hub.HalfConfigured);
        Assert.False(AzulOptions.FromEnvironment(_ => null).Hub.HalfConfigured);
    }

    [Theory]
    [InlineData("Ann@Example.com", "ann@example.com")]
    [InlineData(" a.b+c@sub.example.co ", "a.b+c@sub.example.co")]
    [InlineData("dev@localhost", null)]
    [InlineData("a@b.co\nx", null)]
    [InlineData("no-at-sign", null)]
    [InlineData(null, null)]
    public void EmailsTheHubWouldRejectBecomeNull(string? raw, string? expected) =>
        Assert.Equal(expected, HubContract.Email(raw));

    [Theory]
    [InlineData("bob", "bob")]
    [InlineData("  bob  ", "bob")]
    [InlineData("a\0b", "ab")]
    [InlineData("\uFEFF", "Player")]
    [InlineData("", "Player")]
    [InlineData(null, "Player")]
    public void NamesFollowTheHubRules(string? raw, string expected) =>
        Assert.Equal(expected, HubContract.Name(raw, "Player"));

    [Fact]
    public void NamesAreCutAtFortyCodePoints()
    {
        Assert.Equal(new string('a', 40), HubContract.Name(new string('a', 45), "Player"));
        var emoji = new string('a', 39) + "😀😀";  // 41 code points, 43 UTF-16 units
        Assert.Equal(new string('a', 39) + "😀", HubContract.Name(emoji, "Player"));
    }

    [Fact]
    public void JsTrimRemovesWhatJavaScriptTrims() =>
        Assert.Equal("x", HubContract.JsTrim("\uFEFF\u00a0\u2028 x\t\u3000"));
}
