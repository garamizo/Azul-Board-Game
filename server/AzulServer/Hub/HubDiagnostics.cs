using System.Text.RegularExpressions;

namespace AzulServer.Hub;

/// Hub answers and request errors end up in last_error (and so in `hub
/// status`) and in the logs. Neither may carry an email or a key.
public static partial class HubDiagnostics
{
    public const int MaxLength = 500;

    // Zod's z.email() character classes without the anchors, so any address
    // the hub accepts is found inside a longer text.
    [GeneratedRegex(@"(?:[A-Za-z0-9_'+\-]+\.)*[A-Za-z0-9_'+\-]*[A-Za-z0-9_+-]@(?:[A-Za-z0-9][A-Za-z0-9\-]*\.)+[A-Za-z]{2,}")]
    private static partial Regex Email();

    [GeneratedRegex(@"phk_[A-Za-z0-9_-]+")]
    private static partial Regex GameKey();

    [GeneratedRegex(@"Bearer\s+\S+", RegexOptions.IgnoreCase)]
    private static partial Regex BearerToken();

    /// Characters that can belong to an email or a key: a text cut inside
    /// such a run would keep a fragment the patterns no longer recognise.
    static bool InToken(char c) => char.IsAsciiLetterOrDigit(c) || c is '_' or '\'' or '+' or '-' or '.' or '@';

    /// At most MaxLength characters with emails and keys replaced. A text
    /// that was (or may have been) cut loses its trailing partial token first.
    public static string? Clean(string? text, bool truncated = false)
    {
        if (text is null) return null;
        if (text.Length > MaxLength) { text = text[..MaxLength]; truncated = true; }
        if (truncated)
        {
            int end = text.Length;
            while (end > 0 && InToken(text[end - 1])) end--;
            text = text[..end];
        }
        text = BearerToken().Replace(text, "Bearer <key>");
        text = GameKey().Replace(text, "<key>");
        return Email().Replace(text, "<email>");
    }
}
