namespace AzulServer.Hub;

/// AZUL_HUB_URL / AZUL_HUB_KEY turn the sender on (both or it stays off);
/// AZUL_HUB_PUBLIC_URL is the hub's public origin for the header links.
public sealed record HubOptions
{
    public string? Url { get; init; }
    public string? Key { get; init; }
    public string? PublicUrl { get; init; }

    public bool SenderConfigured => Url is not null && Key is not null;
    public bool HalfConfigured => (Url is null) != (Key is null);
}
