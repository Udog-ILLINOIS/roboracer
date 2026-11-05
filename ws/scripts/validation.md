# Local race validation (v1)

AutoDRIVE must be started manually on macOS/the host. Select the matching track,
connect to `127.0.0.1:4567`, and choose Autonomous driving. This runner cannot
verify the simulator's selected track and does not assume headless support.

From the repository root:

```bash
docker compose up -d
# Build once, or after source/launch changes:
docker compose exec ros bash -lc 'colcon build --symlink-install'
# Separate terminal; leave running:
docker compose exec ros bash -lc 'source install/setup.bash; ros2 launch autodrive_bridge autodrive_launch.py'
# After connecting the simulator, with no other controller/localizer running:
docker compose exec ros bash -lc 'source install/setup.bash; python3 scripts/validate_race.py --track test_track_v2 --localizer mcl --speed-scale 0.3 --laps 2 --timeout 180'
```

Inside an already sourced container shell, the command is simply
`python3 scripts/validate_race.py`. Use `--help` for options. MCL and speed scale
0.3 are conservative defaults; Cartographer is selectable. The runner uses the
current ROS domain and existing vehicle/controller parameters, keeps avoidance
off, and preserves adaptive lookahead. It does not build, map, optimize, or start
another bridge. The existing track must contain `raceline.csv`, `map.yaml`, and
its image; Cartographer also requires `map.pbstream`.

The runner checks bridge nodes and fresh sensor traffic, rejects existing drive
publishers/localizers, requests the same simulator reset used by `race.sh`, then
launches `race.launch.py` through `validation.launch.py`. It waits for the selected
nodes, `/drive`, and fresh valid map-frame `/localized_pose`, and monitors them
throughout the test. Startup has a separate bound (`--startup-timeout`, default
30 s); `--timeout` covers setup plus racing, excluding bounded cleanup. An active
runner holds a domain-specific lock; this does not prevent external commands, so
never launch another controller during validation.

Validation-only remaps disconnect `/ground_truth/odom` and privileged `sim_reset`
hints from the localizer. Normal launches and algorithm code are unchanged.
Cartographer starts at the mapping origin after the explicit reset; crashes do
not trigger privileged reinitialization in this validation scope. MCL continues
its normal sensor-based localization. The unused remap destinations must have no
publishers. This launch wiring still needs verification in ROS Humble.

The separate evaluator subscribes to native `lap_count`, `collision_count`, and
lap-time fields under `/autodrive/roboracer_1/`, plus `/localized_pose`. Native
counters are baselined after reset settles; decreases invalidate the run. The
trajectory is the **localized estimate**, not ground truth. Raw observations
include receive-monotonic times and pose header stamps. Native lap times are
recorded as supplied, without interpreting a partial first lap as a full timed lap.
No trajectory-derived lap estimate is substituted for missing native counters.

**Existing bridge limitation:** `publish_restricted_topics:=true` (the existing
default) exposes lap/collision telemetry, but the unchanged bridge also uses
privileged state for crash/teleport detection and temporary safety stops. The
localizer receives none of those hints. This dependency is reported, not removed
or represented as ground-truth-independent validation. For a strict run with no
privileged bridge input, start the bridge with
`publish_restricted_topics:=false`; native lap/collision metrics will be missing
and this version cannot return a measured pass. Separating telemetry export from
bridge safety behavior requires a later, explicitly scoped bridge change.

Results go to a unique directory under `ws/log/validation/` (already git-ignored),
never into `ws/tracks/`. `--results-dir` must be outside `--tracks-dir` (which also
respects `TRACKS_DIR`). Each run writes:

- `report.json`: versioned result, configuration, artifact hashes, command,
  observed metrics/availability, failure reason, limitations, cleanup escalation.
- `summary.txt`: human-readable result, also printed to the terminal.
- `observations.jsonl`: native telemetry and estimated trajectory samples.
- `race.log`: launch and child-node output, including startup failures.

Exit 0 means requested native laps completed with fresh collision telemetry and
zero measured collisions. Collisions, missing/stale localization or sensors,
unexpected node termination, counter discontinuities, and timeouts fail. Missing
lap telemetry times out as inconclusive; completed laps without reliable collision
data are also inconclusive. Missing metrics are null, never zero. Exit 1 covers
failed/inconclusive runs; interruption returns 130. A pass only describes this
measured configuration, not general safety or deterministic simulation.

On every exit, the runner stops its race process group with SIGINT first (allowing
controller stop bursts), then bounded SIGTERM/SIGKILL escalation if necessary.
Escalation makes the result fail. It never kills the pre-existing bridge. SIGKILL
of the runner itself cannot guarantee cleanup or a report. Check `ros2 node list`
after any abnormal termination before rerunning.

Tests requiring no ROS, Docker, or simulator, from the repository root:

```bash
python3 -m unittest discover -s ws/scripts/test -v
```

These cover CLI/report behavior, mocked lifecycle and failure scenarios, and real
process-group shutdown. They do not establish ROS discovery/remap behavior,
simulator reset/counter semantics, or successful racing. Manually verify those on
`test_track_v2`, including collision, sensor loss, localization loss, Ctrl-C, and
child termination. Preserve the resulting reports and inspect node cleanup.

## Analyze saved results

The analyzer is read-only and needs only Python's standard library; it runs on
the host or in the container without ROS or a live simulator. From the repository
root:

```bash
# Latest finalized report, including failed/interrupted/inconclusive runs:
python3 ws/scripts/analyze_validation.py
# Select a run by directory, report.json path, or name under results-dir:
python3 ws/scripts/analyze_validation.py --run race-p5sx6r98
# Explicit comparison; CSV is optional and must be a new file:
python3 ws/scripts/analyze_validation.py --compare race-mz9ql4vc race-p5sx6r98 \
  --csv /tmp/roboracer-comparison.csv
# Analyze another results collection:
python3 ws/scripts/analyze_validation.py --results-dir /path/to/validation
```

Latest selection uses `started_utc + elapsed_s` as the completion-time estimate
(the runner has no finished timestamp), falling back to report modification time
if needed. A finalized report must have schema version 1, a terminal status,
configuration, metrics, and elapsed time. Unfinished directories are ignored;
malformed reports are skipped with warnings. Explicitly selected invalid reports
produce an error. Missing/truncated observations do not hide report outcomes.

Output includes configuration, outcome/reason, laps, collisions, localization
sample count, telemetry availability/freshness, native timing observations, and
warnings. Differences are B minus A only when both values exist and are valid.
Unknown values print as `unavailable` and produce empty CSV cells, never zeros.
CSV creation refuses existing files and output paths inside selected run or known
track directories. It does not rewrite reports, observations, or track artifacts.
Analysis exit 0 means the analysis succeeded, including for a failed race; exit 2
means an analysis/input/output error.

Timing is deliberately conservative:

- `lap_time` is an active timer, not a completed-lap measurement. Repeated
  `last_lap_time`/`best_lap_time` broadcasts are not independent laps. Raw positive
  value changes are shown, explicitly including possible setup/prior-run values.
- Full-lap candidates require the report's baseline to be observed after the
  first localized sample, consecutive counter increments, and at least two stable
  positive `last_lap_time` samples 0.05–0.5 seconds after a crossing. The first
  crossing is excluded because its lap may be partial. Equal times on different
  counter transitions remain separate candidates.
- Association uses receive times because the native scalar messages lack lap IDs
  and source stamps. This is an explicit heuristic, not proof of frame alignment.
  Missing/ambiguous associations are omitted, never filled from trajectory or
  elapsed runtime. Malformed/discontinuous observations suppress statistics.
- At least two candidates are required for mean, median, min/max, and sample
  standard deviation. Two requested laps may yield no statistics. Cross-run
  timing deltas additionally require matching track and nonempty artifact hashes;
  configuration differences and non-passing outcomes are highlighted. These are
  descriptive comparisons, not automatic claims of performance improvement.

Synthetic analyzer tests alone:

```bash
python3 -m unittest discover -s ws/scripts/test -p 'test_analyze_validation.py' -v
```
