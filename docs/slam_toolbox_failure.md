# First AutoDRIVE SLAM Toolbox map: diagnosis and controlled follow-up

Inputs inspected read-only: `ws/tracks/slam_toolbox_test/map.{pgm,yaml,posegraph,data}`,
`trajectory.csv`, and both `ws/log/map_track_slam_toolbox_test*.log` files. The run
logged two follower laps at 1.0 m/s; it did not generate a validated raceline.
No ground-truth, IPS or privileged odometry was used in this investigation.

## Evidence

| Candidate cause | Finding and limits |
|---|---|
| Online history vs final optimized geometry | Confirmed mismatch. All 401 online stamps match graph scans. Historical samples: 350 free, 32 occupied, 19 unknown. Final samples: 398 free, 3 occupied, 0 unknown. Median correction 0.171 m, maximum 2.989 m; 147 exceed 0.5 m. Historical 3.300 m/0.171 s and 2.977 m/0.107 s jumps disappear in final history. |
| LiDAR mount/angles/timestamps | Serialized laser offset is `(0.2733, 0, 0)` m/rad, matching the adapter; angles approximately ±2.35619 rad with 0.00436332 rad increment. Code preserves acquisition stamps, uses instantaneous scans (`time_increment=0`), and does not reverse ranges. No extrinsic/angle fault established. Original raw scan/TF timing was not recorded, so acquisition latency cannot be ruled out. |
| Odometry translation/heading/calibration | Stored odometry has no metre-scale jumps (max 0.246 m step). Sensor odometry length 53.497 m vs final corrected 49.701 m; accumulated heading -10.663 vs -9.538 rad. These differences merit measurement but cannot identify radius, gyro integration, scale, slip, or steering calibration errors without an independent nonprivileged reference/raw timing. Default heading uses gyro, not the known under-reading steering model. No calibration is changed. |
| Matching / loop closure | Large retroactive corrections are confirmed; final map still contains distorted/collapsed geometry. Exact offending constraints are not proved by INFO logs. Corrected history still fails the existing centerline-length check: 21.4 m centerline vs 27.6 m lap (15% tolerance). Wrong associations are plausible; replacing history alone is insufficient. |
| Async throughput | Four queue-full drops and 13 Ceres thread-limit warnings in a 68.15 s pose history. Processed pose median interval 0.141 s, 95th percentile 0.262 s; initial 3.69 s gap includes startup/stationary time. No input count/processing-age record exists to prove sustained backlog or calculate a rejection fraction. Added runtime diagnostics will measure this on v2. |
| Speed | Follower logs confirm 1.0 m/s and lap completion intervals around 31/34 s. Half speed is a conservative experiment to improve overlap and reduce distance travelled during delays; its benefit has not been demonstrated. |
| Occupancy serialization/origin | Ruled out for this failure: native reconstruction from the serialized corrected graph produces the same 308×188 raster, origin (-3.085331, -5.378589), resolution 0.05 m, and **zero differing pixels**. Saved codes are exactly 0/205/254 (6607/28881/22416 pixels). Row flip, origin and trinary thresholds are consistent. |
| Planner conventions/sampling | CSV is chronological `t,x,y,yaw` in metres/radians in map frame. Native graph stores robot base corrected poses, with LiDAR offset separately applied to rays. Final sample max step 0.379 m. Timestamp matching and corrected path preserve direction. Planner correctly rejects both historical occupancy inconsistency and remaining final-loop geometry; no tolerance changes are justified. |

The final trajectory's high free-cell fraction is not evidence of a valid track:
the map can collapse distinct corridors or create shortcuts while scan poses
remain in its own free cells. Both the overlay and the unchanged planner's
length/topology checks support persistent poor geometry beyond stale history.

## Focused changes and verification boundary

- A native exporter uses installed SDK deserialization and corrected scan poses
  with original times. It rasterizes those same corrected scans, without running
  optimization again. The saver retains `trajectory_online.csv`, and uses the
  consistent final `trajectory.csv` for planning. No historical-pose fallback.
- JSON diagnostics report cell categories, discontinuities, timestamp-matched
  corrections, native-vs-saved raster differences, existing planner result and
  available log/runtime warnings. None of this repairs geometry or changes checks.
- SLAM-only follower default: 0.5 m/s. Loop-search radius: 3 → 1 m; coarse/fine
  minimum responses: 0.35/0.45 → 0.55/0.65. Keep loop closure enabled. This
  conservative experiment targets ambiguous cross-corridor associations, but may
  miss true closures with large odometry drift. Scan queue/rate, beam count,
  mounts and odometry calibration remain unchanged until timing evidence exists.
- Explicit min-pass-through 2 and occupancy threshold 0.1 preserve existing native
  raster defaults and keep graph export configuration synchronized. The installed
  Ceres implementation hardcodes 50 requested threads, then caps them at 10; no
  unsupported YAML thread-setting workaround is introduced.

Static inspection, read-only extraction of the real failed artifacts, automated
synthetic mapping/diagnostic tests and ROS builds verify plumbing and evidence.
They do not demonstrate improved AutoDRIVE map quality. Preserve the failed input
for comparison and use `slam_toolbox_test_v2` for the next experiment. Commands
are in [the pipeline guide](pipeline.md#read-only-mapping-diagnostics).
