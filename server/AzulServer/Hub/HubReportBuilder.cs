using AzulServer.Data;

namespace AzulServer.Hub;

/// GameService builds reports through this, so a test can make the build
/// throw for real (spec 10); production uses HubReport.Build unchanged.
public interface IHubReportBuilder
{
    HubBuild Build(GameRecord g, ISet<int> botPlayedSeats, string? publicOrigin);
}

public sealed class HubReportBuilder : IHubReportBuilder
{
    public HubBuild Build(GameRecord g, ISet<int> botPlayedSeats, string? publicOrigin) =>
        HubReport.Build(g, botPlayedSeats, publicOrigin);
}
