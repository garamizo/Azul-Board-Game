using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace AzulServer.Hub;

/// The hub validates reports with Zod and JavaScript string semantics
/// (~/playhub/src/lib/server/ingest/schema.ts); these make a value pass there
/// instead of turning the whole report into a permanent 422.
public static partial class HubContract
{
    // Zod v4 z.email(): zod/v4/core/regexes.js `email`, with `$` written as
    // \z because .NET's `$` also matches before a final newline.
    [GeneratedRegex(@"^(?:[A-Za-z0-9_'+\-]+\.)*[A-Za-z0-9_'+\-]*[A-Za-z0-9_+-]@(?:[A-Za-z0-9][A-Za-z0-9\-]*\.)+[A-Za-z]{2,}\z")]
    private static partial Regex ZodEmail();

    /// The address the hub would store, or null for one it would refuse (the
    /// hub then records "unidentified player" instead of rejecting the report).
    public static string? Email(string? email)
    {
        if (email is null) return null;
        var e = JsTrim(email).ToLowerInvariant();  // hub: z.string().trim().toLowerCase()
        return ZodEmail().IsMatch(e) ? e : null;
    }

    /// 1-40 code points after a JavaScript trim, no NUL; `fallback` when empty.
    public static string Name(string? raw, string fallback)
    {
        var s = JsTrim((raw ?? "").Replace("\0", ""));
        var cut = new StringBuilder();
        foreach (var rune in s.EnumerateRunes().Take(40)) cut.Append(rune.ToString());
        s = JsTrim(cut.ToString());
        return s.Length == 0 ? fallback : s;
    }

    /// String.prototype.trim: WhiteSpace (incl. U+FEFF and every Zs) and
    /// LineTerminator; unlike .NET's Trim, not U+0085.
    public static string JsTrim(string s)
    {
        int a = 0, b = s.Length;
        while (a < b && IsJsSpace(s[a])) a++;
        while (b > a && IsJsSpace(s[b - 1])) b--;
        return s[a..b];
    }

    static bool IsJsSpace(char c) =>
        c is '\t' or '\n' or '\v' or '\f' or '\r' or ' ' or '\u00a0' or '\uFEFF' or '\u2028' or '\u2029'
        || CharUnicodeInfo.GetUnicodeCategory(c) == UnicodeCategory.SpaceSeparator;
}
