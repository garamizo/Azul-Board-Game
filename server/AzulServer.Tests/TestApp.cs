using System.Net.Http.Json;
using AzulServer.Games;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Data.Sqlite;
using Microsoft.Extensions.DependencyInjection;

namespace AzulServer.Tests;

/// One server instance on its own temporary data directory. Bots are off
/// (BotWorkers = 0) unless a test turns them on; the brain is GreedyBrain.
public sealed class TestApp : IDisposable
{
    public string DataDir { get; }
    public AzulOptions Options { get; }
    readonly WebApplicationFactory<Program> factory;

    public TestApp(AzulOptions? options = null, Action<IServiceCollection>? services = null, string? dataDir = null)
    {
        DataDir = dataDir ?? Directory.CreateTempSubdirectory("azul-test-").FullName;
        Options = (options ?? new AzulOptions { BotWorkers = 0 }) with { DataDir = DataDir };
        factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
        {
            b.UseEnvironment("Testing");
            b.ConfigureTestServices(s =>
            {
                s.AddSingleton(Options);
                s.AddSingleton<IBotBrain, GreedyBrain>();
                services?.Invoke(s);
            });
        });
    }

    public HttpClient Client(string? email = "alice@example.com")
    {
        var c = factory.CreateClient(new WebApplicationFactoryClientOptions { AllowAutoRedirect = false });
        if (email is not null) c.DefaultRequestHeaders.Add("X-Dev-User", email);
        return c;
    }

    public T Service<T>() where T : notnull => factory.Services.GetRequiredService<T>();

    public void Dispose()
    {
        factory.Dispose();
        SqliteConnection.ClearAllPools();
    }
}

public static class HttpExtensions
{
    public static Task<HttpResponseMessage> Post(this HttpClient c, string url, object? body = null) =>
        c.PostAsJsonAsync(url, body ?? new { }, Json.Options);

    public static async Task<T> Read<T>(this HttpResponseMessage r) =>
        (await r.Content.ReadFromJsonAsync<T>(Json.Options))!;
}
