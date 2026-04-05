============================================================
场景 3/5: TJU_west
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_west.csv
输出: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_west_minimax-m2.5.csv
------------------------------------------------------------
Starting prompt-based network slice management system...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I need to download large files"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 20.0

Intent Analysis: N/A
Recommended Slice: eMBB - Large file downloads require high bandwidth (6-20 MHz) and high data rates (100-400 Mbps). eMBB slice is designed for enhanced mobile broadband services with latency tolerance of 10-100ms, which is suitable for file downloads.
Bandwidth Allocation: 10.0 MHz
Data Rate: 20.0 Mbps
Latency: 50.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-31 11:58:46
Total Users: 1
Average Resource Utilization: 7.69%
eMBB Total Rate: 20.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  10.0/90 MHz       11.11%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → eMBB Slice
CQI: 4, Bandwidth: 10.0 MHz, Rate: 20.00 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |     4 |         10 |            20 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.5, rate: 0.87

Intent Analysis: N/A
Recommended Slice: mMTC - The request involves many low‑power sensors sending periodic air‑quality reports, which is a classic mMTC use case requiring low bandwidth, modest data rates, and tolerant latency. With a CQI of 4, the corresponding spectral efficiency is approximately 0.58 bits/Hz. Allocating 1.5 MHz yields about 0.87 Mbps, which fits within the mMTC rate range (0.1‑1 Mbps) and the latency window (100‑1000 ms). The mMTC slice currently has 0 MHz used out of 10 MHz, so the allocation leaves sufficient capacity for future devices.
Bandwidth Allocation: 1.5 MHz
Data Rate: 0.87 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-31 11:59:22
Total Users: 2
Average Resource Utilization: 8.85%
eMBB Total Rate: 20.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.87 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  10.0/90 MHz       11.11%
URLLC          0  0/30 MHz          0%
mMTC           1  1.5/10 MHz        15.00%

New User Allocation:
User 2 → mMTC Slice
CQI: 4, Bandwidth: 1.5 MHz, Rate: 0.87 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 20.0

Intent Analysis: User request involves controlling precision CNC machines with zero tolerance for delay, which demands ultra‑reliable low‑latency communication (URLLC).
Recommended Slice: URLLC - The user’s requirement for zero‑tolerance delay maps directly to the URLLC slice, which supports latencies of 1‑10 ms. Allocating 5 MHz of bandwidth and a 20 Mbps data rate satisfies the latency requirement while staying within the slice’s allowed ranges. The URLLC slice currently has no active users, so the allocation leaves ample capacity for future URLLC traffic. The eMBB and mMTC slices remain unaffected, preserving balanced resource utilization across the network.
Bandwidth Allocation: 5.0 MHz
Data Rate: 20.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-31 11:59:45
Total Users: 3
Average Resource Utilization: 12.69%
eMBB Total Rate: 20.00 Mbps, URLLC Total Rate: 20.00 Mbps, mMTC Total Rate: 0.87 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  10.0/90 MHz       11.11%
URLLC          1  5.0/30 MHz        16.67%
mMTC           1  1.5/10 MHz        15.00%

New User Allocation:
User 3 → URLLC Slice
CQI: 7, Bandwidth: 5.0 MHz, Rate: 20.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        5   |         20    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to send text messages and use messaging apps"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 0.5, rate: 0.5

Intent Analysis: N/A
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 0.5 MHz
Data Rate: 0.5 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-31 12:00:22
Total Users: 4
Average Resource Utilization: 13.08%
eMBB Total Rate: 20.00 Mbps, URLLC Total Rate: 20.00 Mbps, mMTC Total Rate: 1.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  10.0/90 MHz       11.11%
URLLC          1  5.0/30 MHz        16.67%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 4 → mMTC Slice
CQI: 7, Bandwidth: 0.5 MHz, Rate: 0.50 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I need to use maps for basic navigation"
CQI: 3
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 3.3

Intent Analysis: N/A
Recommended Slice: eMBB - The user request is for basic map navigation, which requires moderate data rate and low latency. A CQI of 3 indicates limited channel quality, resulting in a low spectral efficiency. The URLLC slice supports latency in the 1‑10 ms range and allows data rates from 1 Mbps up to 100 Mbps, accommodating the achievable ~3.3 Mbps. Allocating the maximum available bandwidth (5 MHz) on the URLLC slice maximizes the attainable rate while keeping latency low and staying within slice constraints.
Bandwidth Allocation: 5.0 MHz
Data Rate: 3.3 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-31 12:00:58
Total Users: 5
Average Resource Utilization: 16.92%
eMBB Total Rate: 23.30 Mbps, URLLC Total Rate: 20.00 Mbps, mMTC Total Rate: 1.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  15.0/90 MHz       16.67%
URLLC          1  5.0/30 MHz        16.67%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 5 → eMBB Slice
CQI: 3, Bandwidth: 5.0 MHz, Rate: 3.30 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.0

Intent Analysis: N/A
Recommended Slice: mMTC - Environmental sensors transmit small payloads at low frequency; Air quality reporting does not require ultra-low latency; mMTC slice optimized for massive IoT device connectivity; Current mMTC slice has 80% bandwidth headroom available
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.0 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-31 12:01:18
Total Users: 6
Average Resource Utilization: 17.69%
eMBB Total Rate: 23.30 Mbps, URLLC Total Rate: 20.00 Mbps, mMTC Total Rate: 2.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  15.0/90 MHz       16.67%
URLLC          1  5.0/30 MHz        16.67%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 6 → mMTC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 1.00 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need reliable connectivity for implanted medical devices"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 2.79

Intent Analysis: N/A
Recommended Slice: eMBB - A 3 MHz grant satisfies the low‑latency requirement while staying well within the URLLC bandwidth limits (1‑5 MHz) and leaving headroom for other URLLC users.
Bandwidth Allocation: 3.0 MHz
Data Rate: 2.79 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-31 12:01:54
Total Users: 7
Average Resource Utilization: 20.0%
eMBB Total Rate: 26.09 Mbps, URLLC Total Rate: 20.00 Mbps, mMTC Total Rate: 2.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  18.0/90 MHz       20.00%
URLLC          1  5.0/30 MHz        16.67%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 7 → eMBB Slice
CQI: 6, Bandwidth: 3.0 MHz, Rate: 2.79 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I want to remotely access my work computer"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 80.0

Intent Analysis: Remote desktop access requires moderate bandwidth for screen sharing and file transfers, with moderate latency sensitivity. This is a typical human-to-machine interaction workload.
Recommended Slice: eMBB - Rate adjusted to 80 Mbps based on CQI=4 (moderate channel quality). This provides sufficient throughput for remote desktop while staying within eMBB parameters.
Bandwidth Allocation: 10.0 MHz
Data Rate: 80.0 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-31 12:02:17
Total Users: 8
Average Resource Utilization: 27.69%
eMBB Total Rate: 106.09 Mbps, URLLC Total Rate: 20.00 Mbps, mMTC Total Rate: 2.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  28.0/90 MHz       31.11%
URLLC          1  5.0/30 MHz        16.67%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 8 → eMBB Slice
CQI: 4, Bandwidth: 10.0 MHz, Rate: 80.00 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to use holographic communication"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 15.0, rate: 150.0

Intent Analysis: Holographic communication demands high data rates and low latency for real‑time 3D video rendering. The service profile best matches the eMBB slice, which offers the required bandwidth (6‑20 MHz), high data rates (100‑400 Mbps) and latency in the 10‑100 ms range.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 15.0 MHz
Data Rate: 150.0 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-31 12:02:58
Total Users: 9
Average Resource Utilization: 39.23%
eMBB Total Rate: 256.09 Mbps, URLLC Total Rate: 20.00 Mbps, mMTC Total Rate: 2.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  43.0/90 MHz       47.78%
URLLC          1  5.0/30 MHz        16.67%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 9 → eMBB Slice
CQI: 15, Bandwidth: 15.0 MHz, Rate: 150.00 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 40.0

Intent Analysis: N/A
Recommended Slice: URLLC - The user's request for zero-delay control of precision CNC machines maps directly to Ultra‑Reliable Low‑Latency Communication (URLLC). With a CQI of 15 (the highest), we can apply the most aggressive modulation (256‑QAM) and a high coding rate, enabling a 5 MHz allocation that yields ~40 Mbps while targeting the minimum possible latency of 1 ms – within the URLLC slice limits (1‑5 MHz, 1‑100 Mbps, 1‑10 ms). This allocation stays within slice constraints and leaves ample remaining capacity, preserving workload balance across slices.
Bandwidth Allocation: 5.0 MHz
Data Rate: 40.0 Mbps
Latency: 1.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-31 12:03:51
Total Users: 10
Average Resource Utilization: 43.08%
eMBB Total Rate: 256.09 Mbps, URLLC Total Rate: 60.00 Mbps, mMTC Total Rate: 2.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  43.0/90 MHz       47.78%
URLLC          2  10.0/30 MHz       33.33%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 10 → URLLC Slice
CQI: 15, Bandwidth: 5.0 MHz, Rate: 40.00 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need emergency response coordination during a disaste"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 1.8

Intent Analysis: Emergency response coordination during a disaster – requires reliable, low‑latency communication to support voice, critical data, and possibly situational video.
Recommended Slice: URLLC - Allocated 3 MHz yields ~1.8 Mbps, which lies comfortably within the URLLC rate range (1‑100 Mbps) while satisfying the low‑latency requirement.
Bandwidth Allocation: 3.0 MHz
Data Rate: 1.8 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-31 12:04:29
Total Users: 11
Average Resource Utilization: 45.38%
eMBB Total Rate: 256.09 Mbps, URLLC Total Rate: 61.80 Mbps, mMTC Total Rate: 2.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  43.0/90 MHz       47.78%
URLLC          3  13.0/30 MHz       43.33%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 3, Bandwidth: 3.0 MHz, Rate: 1.80 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I want to use remote surgery equipment"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 7.5

Intent Analysis: Remote surgery equipment (ultra‑reliable low‑latency control and high‑definition video)
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 7.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-31 12:05:17
Total Users: 12
Average Resource Utilization: 49.23%
eMBB Total Rate: 256.09 Mbps, URLLC Total Rate: 69.30 Mbps, mMTC Total Rate: 2.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  43.0/90 MHz       47.78%
URLLC          4  18.0/30 MHz       60.00%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 7, Bandwidth: 5.0 MHz, Rate: 7.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need to participate in an online multiplayer game"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 150.0

Intent Analysis: N/A
Recommended Slice: eMBB - The user wants to take part in an online multiplayer game. Such traffic typically requires moderate bandwidth (tens of Mbps) and latency in the tens‑of‑milliseconds range. The eMBB slice can provide 6‑20 MHz of bandwidth with data rates from 100 Mbps up to 400 Mbps and latency between 10 ms and 100 ms, which matches the gaming requirements. Allocating 10 MHz and targeting a data rate of 150 Mbps (well within the eMBB rate window) satisfies the user while keeping the eMBB slice’s utilization at an acceptable level (~59 %). The allocation does not overload any slice and leaves headroom for additional users.
Bandwidth Allocation: 10.0 MHz
Data Rate: 150.0 Mbps
Latency: 50.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-31 12:06:30
Total Users: 13
Average Resource Utilization: 56.92%
eMBB Total Rate: 406.09 Mbps, URLLC Total Rate: 69.30 Mbps, mMTC Total Rate: 2.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  53.0/90 MHz       58.89%
URLLC          4  18.0/30 MHz       60.00%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 13 → eMBB Slice
CQI: 7, Bandwidth: 10.0 MHz, Rate: 150.00 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |       10   |        150    |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need to track the location of a shipping container"
CQI: 14
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.5, rate: 0.5

Intent Analysis: N/A
Recommended Slice: mMTC - Location tracking of a shipping container is a low‑data‑rate, delay‑tolerant IoT use case. The mMTC slice is designed for massive machine‑type communications with modest bandwidth (1‑3 MHz), low data rates (0.1‑1 Mbps) and容忍 latency (100‑1000 ms). A CQI of 14 indicates good channel conditions, allowing reliable transmission of small location updates within the mMTC constraints. Allocating 1.5 MHz and 0.5 Mbps satisfies the user’s needs while staying well within the slice’s limits.
Bandwidth Allocation: 1.5 MHz
Data Rate: 0.5 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-31 12:06:57
Total Users: 14
Average Resource Utilization: 58.08%
eMBB Total Rate: 406.09 Mbps, URLLC Total Rate: 69.30 Mbps, mMTC Total Rate: 2.87 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  53.0/90 MHz       58.89%
URLLC          4  18.0/30 MHz       60.00%
mMTC           4  4.5/10 MHz        45.00%

New User Allocation:
User 14 → mMTC Slice
CQI: 14, Bandwidth: 1.5 MHz, Rate: 0.50 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |       10   |        150    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.5 |          0.5  |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 100.0

Intent Analysis: N/A
Recommended Slice: eMBB - The request for cloud‑based AI image processing demands high bandwidth and high data rate with moderate latency, which aligns with the eMBB slice capabilities. CQI 15 indicates excellent channel quality, supporting high‑order modulation (64‑QAM) and a spectral efficiency of roughly 5 bits/s/Hz. Allocating the maximum allowed eMBB bandwidth of 20 MHz yields a data rate of 100 Mbps, satisfying the slice’s 100‑400 Mbps rate constraint while keeping latency within the 10‑100 ms range. The eMBB slice’s remaining capacity after allocation is 17 MHz, preserving headroom for future users and maintaining overall network balance.
Bandwidth Allocation: 20.0 MHz
Data Rate: 100.0 Mbps
Latency: 50.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-31 12:07:34
Total Users: 15
Average Resource Utilization: 73.46%
eMBB Total Rate: 506.09 Mbps, URLLC Total Rate: 69.30 Mbps, mMTC Total Rate: 2.87 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  73.0/90 MHz       81.11%
URLLC          4  18.0/30 MHz       60.00%
mMTC           4  4.5/10 MHz        45.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 15, Bandwidth: 20.0 MHz, Rate: 100.00 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |       10   |        150    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        100    |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need to monitor water level in a reservoir"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 1.0

Intent Analysis: monitor water level in a reservoir
Recommended Slice: mMTC - Water‑level monitoring generates small, periodic IoT packets that fit the mMTC profile (low bandwidth, low‑to‑moderate rate, higher latency tolerance). The mMTC slice currently has ample free resources (5.5 MHz available). Allocating 2 MHz with a capped rate of 1 Mbps respects the slice’s bandwidth and rate limits while keeping utilization within capacity. The latency of 500 ms satisfies the 100‑1000 ms requirement for mMTC.
Bandwidth Allocation: 2.0 MHz
Data Rate: 1.0 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-31 12:08:04
Total Users: 16
Average Resource Utilization: 75.0%
eMBB Total Rate: 506.09 Mbps, URLLC Total Rate: 69.30 Mbps, mMTC Total Rate: 3.87 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  73.0/90 MHz       81.11%
URLLC          4  18.0/30 MHz       60.00%
mMTC           5  6.5/10 MHz        65.00%

New User Allocation:
User 16 → mMTC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 1.00 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |       10   |        150    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2   |          1    |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to monitor soil moisture levels in a large farm"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: The user wants to monitor soil moisture across a large farm. This application generates low‑volume, periodic sensor data, which is best served by the massive Machine‑Type Communications (mMTC) slice. The low CQI of 4 indicates a relatively weak radio channel, so a modest bandwidth allocation with a conservative data rate is appropriate.
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-31 12:08:26
Total Users: 17
Average Resource Utilization: 75.77%
eMBB Total Rate: 506.09 Mbps, URLLC Total Rate: 69.30 Mbps, mMTC Total Rate: 4.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  73.0/90 MHz       81.11%
URLLC          4  18.0/30 MHz       60.00%
mMTC           6  7.5/10 MHz        75.00%

New User Allocation:
User 17 → mMTC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |       10   |        150    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1   |          0.5  |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need instant alerts for life-threatening patient conditions"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 3.5

Intent Analysis: The user requires ultra‑reliable, low‑latency communication to transmit critical health‑alert messages. This is characteristic of a URLLC (Ultra‑Reliable Low‑Latency Communications) use case.
Recommended Slice: URLLC - URLLC provides the required 1–10 ms latency and supports the needed modest data rate (1–100 Mbps) with bandwidth allocations of 1–5 MHz, perfectly matching the small, time‑critical alert payload.
Bandwidth Allocation: 2.0 MHz
Data Rate: 3.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-31 12:09:12
Total Users: 18
Average Resource Utilization: 77.31%
eMBB Total Rate: 506.09 Mbps, URLLC Total Rate: 72.80 Mbps, mMTC Total Rate: 4.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  73.0/90 MHz       81.11%
URLLC          5  20.0/30 MHz       66.67%
mMTC           6  7.5/10 MHz        75.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 6, Bandwidth: 2.0 MHz, Rate: 3.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |          3.5  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |       10   |        150    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to download large files"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 12.0, rate: 120.0

Intent Analysis: User wants to download large files, which requires high data throughput and moderate latency, typical of an eMBB slice.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 12.0 MHz
Data Rate: 120.0 Mbps
Latency: 30.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-31 12:10:03
Total Users: 19
Average Resource Utilization: 86.54%
eMBB Total Rate: 626.09 Mbps, URLLC Total Rate: 72.80 Mbps, mMTC Total Rate: 4.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  85.0/90 MHz       94.44%
URLLC          5  20.0/30 MHz       66.67%
mMTC           6  7.5/10 MHz        75.00%

New User Allocation:
User 19 → eMBB Slice
CQI: 8, Bandwidth: 12.0 MHz, Rate: 120.00 Mbps, Latency: 30.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |          3.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |       10   |        150    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       12   |        120    |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I need to synchronize distributed financial ledgers instantly"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 12.5

Intent Analysis: User needs to synchronize distributed financial ledgers instantly. This is a latency‑critical, reliability‑sensitive transaction, best served by a URLLC slice.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 12.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-31 12:10:33
Total Users: 20
Average Resource Utilization: 90.38%
eMBB Total Rate: 626.09 Mbps, URLLC Total Rate: 85.30 Mbps, mMTC Total Rate: 4.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  85.0/90 MHz       94.44%
URLLC          6  25.0/30 MHz       83.33%
mMTC           6  7.5/10 MHz        75.00%

New User Allocation:
User 20 → URLLC Slice
CQI: 9, Bandwidth: 5.0 MHz, Rate: 12.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |          3.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         12.5  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |       10   |        150    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       12   |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 5.46

Intent Analysis: N/A
Recommended Slice: URLLC - Firefighters require ultra‑reliable low‑latency communications, which aligns with the URLLC slice profile. A CQI of 6 maps to 64‑QAM with a code rate of ~0.6, giving a spectral efficiency of ~2.73 bits/Hz. Allocating 2 MHz yields an estimated data rate of ~5.5 Mbps, satisfying the URLLC rate range (1‑100 Mbps) while keeping the slice utilization at 90 % (below the 90 % threshold). The latency of 5 ms meets the URLLC requirement of 1‑10 ms, ensuring reliable and timely communication for responders inside buildings.
Bandwidth Allocation: 2.0 MHz
Data Rate: 5.46 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-31 12:11:01
Total Users: 21
Average Resource Utilization: 91.92%
eMBB Total Rate: 626.09 Mbps, URLLC Total Rate: 90.76 Mbps, mMTC Total Rate: 4.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  85.0/90 MHz       94.44%
URLLC          7  27.0/30 MHz       90.00%
mMTC           6  7.5/10 MHz        75.00%

New User Allocation:
User 21 → URLLC Slice
CQI: 6, Bandwidth: 2.0 MHz, Rate: 5.46 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |          3.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |          5.46 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |       10   |        150    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       12   |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "Smart city parking sensor reporting availability"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.1

Intent Analysis: The request is for a low‑volume, periodic status report from a smart‑city parking sensor. Such traffic is characterised by small payload sizes, infrequent transmission, and relaxed latency requirements, making it ideal for massive Machine‑Type Communications (mMTC).
Recommended Slice: mMTC - Bandwidth set to the minimum (1 MHz) to conserve resources. With CQI 7 (≈16‑QAM, coding ≈0.476) the raw spectral efficiency is ~2.5 bits/s/Hz, yielding a raw rate of ~2.5 Mbps on 1 MHz. To comply with the mMTC slice’s 0.1‑1 Mbps rate cap, the rate is limited to 0.5 Mbps. The selected latency (500 ms) falls within the allowed 100‑1000 ms range.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.1 Mbps
Latency: 100.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-31 12:11:36
Total Users: 22
Average Resource Utilization: 92.69%
eMBB Total Rate: 626.09 Mbps, URLLC Total Rate: 90.76 Mbps, mMTC Total Rate: 4.47 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  85.0/90 MHz       94.44%
URLLC          7  27.0/30 MHz       90.00%
mMTC           7  8.5/10 MHz        85.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 0.10 Mbps, Latency: 100.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |          3.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |       10   |        150    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       12   |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        1   |          0.1  |            100 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 1.8

Intent Analysis: N/A
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 2.0 MHz
Data Rate: 1.8 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-31 12:12:32
Total Users: 23
Average Resource Utilization: 94.23%
eMBB Total Rate: 626.09 Mbps, URLLC Total Rate: 92.56 Mbps, mMTC Total Rate: 4.47 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  85.0/90 MHz       94.44%
URLLC          8  29.0/30 MHz       96.67%
mMTC           7  8.5/10 MHz        85.00%

New User Allocation:
User 23 → URLLC Slice
CQI: 4, Bandwidth: 2.0 MHz, Rate: 1.80 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |          3.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |          1.8  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |       10   |        150    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       12   |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        1   |          0.1  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to monitor IoT sensors in real-time"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.5

Intent Analysis: Real-time IoT sensor monitoring with low latency requirements for continuous data acquisition and transmission
Recommended Slice: URLLC - Real-time monitoring requires low latency (1-10ms) - URLLC is the only slice meeting this constraint; IoT sensor data rates are typically low (< 10 Mbps) - within URLLC capabilities; CQI of 4 indicates moderate channel quality - acceptable for URLLC operation; Alternative: mMTC would work for IoT but lacks real-time latency guarantees
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.5 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-31 12:12:57
Total Users: 24
Average Resource Utilization: 95.0%
eMBB Total Rate: 626.09 Mbps, URLLC Total Rate: 94.06 Mbps, mMTC Total Rate: 4.47 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  85.0/90 MHz       94.44%
URLLC          9  30.0/30 MHz       100.00%
mMTC           7  8.5/10 MHz        85.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 1.50 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |          3.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |          1.8  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          1.5  |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |       10   |        150    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       12   |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        1   |          0.1  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "My smart trash can needs to signal that it's full"
CQI: 5
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.0

Intent Analysis: Transmit simple status update (trash can full indicator)
Recommended Slice: mMTC - Smart trash can is an IoT device requiring M2M communication; Status signaling requires minimal bandwidth (< 1 MHz); Low data rate requirements (0.1-1 Mbps sufficient); Event-triggered transmission pattern typical of mMTC use cases; CQI 5 indicates moderate channel - mMTC can handle this reliably
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.0 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-31 12:13:27
Total Users: 25
Average Resource Utilization: 95.77%
eMBB Total Rate: 626.09 Mbps, URLLC Total Rate: 94.06 Mbps, mMTC Total Rate: 5.47 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  85.0/90 MHz       94.44%
URLLC          9  30.0/30 MHz       100.00%
mMTC           8  9.5/10 MHz        95.00%

New User Allocation:
User 25 → mMTC Slice
CQI: 5, Bandwidth: 1.0 MHz, Rate: 1.00 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |          3.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |          1.8  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          1.5  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |       10   |        150    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       12   |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        1   |          0.1  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        1   |          1    |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 6.0, rate: 7.0

Intent Analysis: N/A
Recommended Slice: eMBB - Video surveillance monitoring requires sustained high bandwidth for continuous streaming, which aligns with eMBB capabilities. The application needs higher data rates (4-20 Mbps) rather than ultra-low latency or massive device connectivity.
Bandwidth Allocation: 6.0 MHz
Data Rate: 7.0 Mbps
Latency: 10.0 ms

CAPACITY CHECK FAILED:
- Required: 6.0 MHz, Available: 5.0 MHz in eMBB slice

----------------------------------------
ALLOCATION FAILED FOR USER 26
----------------------------------------
Request: I want to monitor my home security cameras remotely
Slice type: eMBB
Reason: Insufficient capacity in eMBB slice. Required: 6.0 MHz, Available: 5.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: Industrial equipment monitoring sensor data - typical low‑rate, periodic telemetry from sensors.
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 150.0 ms

CAPACITY CHECK FAILED:
- Required: 1.0 MHz, Available: 0.5 MHz in mMTC slice

----------------------------------------
ALLOCATION FAILED FOR USER 27
----------------------------------------
Request: Industrial equipment monitoring sensor data
Slice type: mMTC
Reason: Insufficient capacity in mMTC slice. Required: 1.0 MHz, Available: 0.5 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: Smart parking IoT sensor reporting occupancy status (binary free/occupied)
Recommended Slice: mMTC - Smart parking sensors are IoT devices requiring low bandwidth, low data rates, and minimal latency tolerance. mMTC slice is optimized for massive machine-type communications with the ability to handle many low-complexity devices. URLLC is excessive for simple status reporting, and eMBB is unsuitable for IoT sensor data.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 500.0 ms

CAPACITY CHECK FAILED:
- Required: 1.0 MHz, Available: 0.5 MHz in mMTC slice

----------------------------------------
ALLOCATION FAILED FOR USER 28
----------------------------------------
Request: My smart parking sensor needs to report if the spot is free
Slice type: mMTC
Reason: Insufficient capacity in mMTC slice. Required: 1.0 MHz, Available: 0.5 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I want to stream music while browsing social media"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 11.5

Intent Analysis: The user request involves streaming music and browsing social media. These are continuous, high-bandwidth data activities requiring moderate latency (10-100ms). This profile aligns perfectly with the eMBB (enhanced Mobile Broadband) slice, designed for high data rates and entertainment services.
Recommended Slice: eMBB - The eMBB slice is operating at 94.44% utilization (85/90 MHz used). The user is allocated the remaining available 5 MHz. While this does not meet the maximum theoretical eMBB standards, it sufficiently supports the user's requested application (Spotify/Social Media typically requires <5 Mbps).
Bandwidth Allocation: 5.0 MHz
Data Rate: 11.5 Mbps
Latency: 15.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-31 12:16:09
Total Users: 26
Average Resource Utilization: 99.62%
eMBB Total Rate: 637.59 Mbps, URLLC Total Rate: 94.06 Mbps, mMTC Total Rate: 5.47 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           9  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           8  9.5/10 MHz        95.00%

New User Allocation:
User 29 → eMBB Slice
CQI: 5, Bandwidth: 5.0 MHz, Rate: 11.50 Mbps, Latency: 15.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        5   |         40    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        3   |          1.8  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |          3.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |          1.8  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          1.5  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        5   |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       10   |         20    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |       10   |        150    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       12   |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | eMBB    |     5 |        5   |         11.5  |             15 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |        5   |          3.3  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |     6 |        3   |          2.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       10   |         80    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        150    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.5 |          0.87 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        1   |          0.1  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |        0.5 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 11
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 0.5

Intent Analysis: Industrial equipment monitoring sensor data
Recommended Slice: URLLC - 2 MHz falls within the URLLC bandwidth limits (1–5 MHz). With CQI 11 (≈3.5 bits/Hz), the resulting 7 Mbps data rate satisfies the 1–100 Mbps URLLC rate requirement and provides ultra‑low latency suitable for industrial monitoring.
Bandwidth Allocation: 2.0 MHz
Data Rate: 0.5 Mbps
Latency: 10.0 ms

CAPACITY CHECK FAILED:
- Required: 2.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 30
----------------------------------------
Request: Industrial equipment monitoring sensor data
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 2.0 MHz, Available: 0.0 MHz

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |     4 |       10   |         20    |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | mMTC    | mMTC           | Yes            |     4 |        1.5 |          0.87 |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | URLLC   | URLLC          | Yes            |     7 |        5   |         20    |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | mMTC    | eMBB           | No             |     7 |        0.5 |          0.5  |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | eMBB    | eMBB           | Yes            |     3 |        5   |          3.3  |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | mMTC    | mMTC           | Yes            |     6 |        1   |          1    |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | eMBB    | URLLC          | No             |     6 |        3   |          2.79 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | eMBB    | eMBB           | Yes            |     4 |       10   |         80    |             10 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | eMBB    | eMBB           | Yes            |    15 |       15   |        150    |             20 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | URLLC   | URLLC          | Yes            |    15 |        5   |         40    |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |     3 |        3   |          1.8  |             10 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     7 |        5   |          7.5  |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | eMBB    | URLLC          | No             |     7 |       10   |        150    |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | mMTC    | mMTC           | Yes            |    14 |        1.5 |          0.5  |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |    15 |       20   |        100    |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | mMTC    | mMTC           | Yes            |     9 |        2   |          1    |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | mMTC    | mMTC           | Yes            |     4 |        1   |          0.5  |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     6 |        2   |          3.5  |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | eMBB    | eMBB           | Yes            |     8 |       12   |        120    |             30 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | URLLC   | URLLC          | Yes            |     9 |        5   |         12.5  |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | URLLC   | URLLC          | Yes            |     6 |        2   |          5.46 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | mMTC    | mMTC           | Yes            |     7 |        1   |          0.1  |            100 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | URLLC   | URLLC          | Yes            |     4 |        2   |          1.8  |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          1.5  |             10 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | mMTC    | mMTC           | Yes            |     5 |        1   |          1    |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Failed   | eMBB    | eMBB           |                |     7 |        6   |          7    |             10 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Failed   | mMTC    | mMTC           |                |     4 |        1   |          0.5  |            150 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Failed   | mMTC    | mMTC           |                |     9 |        1   |          0.5  |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | eMBB    | eMBB           | Yes            |     5 |        5   |         11.5  |             15 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Failed   | URLLC   | mMTC           |                |    11 |        2   |          0.5  |             10 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 26/30 (86.7%)

Intent Understanding Evaluation:
Correctly identified intents: 23/26
Intent understanding rate: 88.5%

Workload Balancing Statistics:
Users with workload balancing: 26/30
Workload balancing rate: 86.7%

Slice Utilization Statistics:
Average eMBB utilization: 58.59%
Average URLLC utilization: 51.80%
Average mMTC utilization: 49.42%

Results exported to F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_west_minimax-m2.5.csv

[OK] TJU_west 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_west_minimax-m2.5.csv