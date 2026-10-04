/// One take's tiles in the air: `line.count` tiles of `color` to the pattern
/// line, and `floor` (colours, 5 = the marker) to the floor of `seat`.
export interface FlightPlan {
  id: number;
  seat: number;
  source: number;  // factory index; the centre is the number of factories
  color: number;
  line: { row: number; count: number } | null;
  floor: number[];
}
