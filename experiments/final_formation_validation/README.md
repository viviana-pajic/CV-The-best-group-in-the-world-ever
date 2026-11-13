# Formation scoring validation

Seven deterministic position sequences exercise the production dashboard engine on a 12 × 12 m stage. The cases have known row geometry, a fixed minimum of four reported positions, and 25 frames per second. The score falls with mean perpendicular distance to the fitted line. The check also exercises angle limits, estimated points, and the minimum duration for a row. [Reproduction script](../../scripts/validate_formation.cjs) · [Results](summary.json)

A dancer displaced by 0.48 m reduces the score of a four-person row from 100% to 83.33%. Moving the entire straight row by 0.48 m gives 100% alignment while leaving a 0.48 m error against the original reference. A judge can use the alignment score to review the shape and use the stage map to inspect placement.

These cases isolate the scoring logic with known positions. The end-to-end replays also depend on detection, tracking and calibration. Judging against planned choreography requires reference formations from the performance.

At onare 03:04.68, the dashboard fitted a six-dancer row at 91.67%. The same six dancers scored 91.37% using approximate visual floor-contact picks and the same stage calibration. The seventh visual reference matched a hollow estimated point in the replay. This illustrates the workflow on one previously viewed frame. [Visual picks](onare_reference.json) · [Comparison](onare_comparison.json)

For camera transfer, we compared 38 approximate exposed-edge picks across three replay frames. Registration lowered the median pixel error in each frame compared with using the original static calibration directly. The onare left edge still had a 17.16 px median residual. The picks describe visible image edges, with uncertainty from lighting and perspective. [Edge picks](edge_picks.json) · [Registration results](registration_check.json)
