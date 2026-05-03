============================================================
场景 3/5: TJU_east
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_east.csv
输出: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_east_minimax-m2.5.csv
------------------------------------------------------------
Starting prompt-based network slice management system...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I need to check weather forecasts"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~344, completion~941, total~1285

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 200.0

Intent Analysis: Web browsing - weather forecast access
Recommended Slice: eMBB - Weather forecast checking is a typical data-intensive web browsing application. eMBB slice provides appropriate bandwidth and data rates for this use case. The request does not require ultra-low latency (URLLC) or massive device connectivity (mMTC).
Bandwidth Allocation: 10.0 MHz
Data Rate: 200.0 Mbps
Latency: 25.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-04-05 23:35:14
Total Users: 1
Average Resource Utilization: 7.69%
eMBB Total Rate: 200.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  10.0/90 MHz       11.11%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → eMBB Slice
CQI: 15, Bandwidth: 10.0 MHz, Rate: 200.00 Mbps, Latency: 25.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         10 |           200 |             25 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need to stream 8K video content"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~349, completion~2307, total~2656

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 100.0

Intent Analysis: stream_8k_video
Recommended Slice: eMBB - CQI=4 corresponds to a moderate‑low channel quality (approx. 16‑QAM). Using the maximum allowable eMBB bandwidth of 20 MHz and a conservative spectral efficiency of 5 bits/s/Hz yields an achievable rate of ~100 Mbps, which satisfies the lower bound of the eMBB slice rate range (100‑400 Mbps) and stays within its latency budget (10‑100 ms).
Bandwidth Allocation: 20.0 MHz
Data Rate: 100.0 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-04-05 23:35:52
Total Users: 2
Average Resource Utilization: 23.08%
eMBB Total Rate: 300.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  30.0/90 MHz       33.33%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 100.00 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         10 |           200 |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |           100 |             20 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I want to listen to low-quality audio streaming"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~349, completion~1244, total~1593

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 6.0, rate: 64.0

Intent Analysis: Low-quality audio streaming requires minimal bandwidth and has moderate latency tolerance. The user needs consistent data flow for continuous audio playback but does not require ultra-low latency or extremely high data rates.
Recommended Slice: eMBB - Audio streaming is a mobile broadband use case. While URLLC could technically support it, the low-latency features are unnecessary for audio playback which inherently includes buffering. mMTC is unsuitable as it only supports up to 1 Mbps which may be insufficient and is designed for massive IoT device connectivity. eMBB provides the appropriate balance of bandwidth and latency for streaming services.
Bandwidth Allocation: 6.0 MHz
Data Rate: 64.0 Mbps
Latency: 50.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-04-05 23:36:12
Total Users: 3
Average Resource Utilization: 27.69%
eMBB Total Rate: 364.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  36.0/90 MHz       40.00%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 3 → eMBB Slice
CQI: 15, Bandwidth: 6.0 MHz, Rate: 64.00 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         10 |           200 |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |           100 |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |            64 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I want to browse websites and check email"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~348, completion~2089, total~2437

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 100.0

Intent Analysis: The user requires moderate bandwidth for web browsing and email access, with relatively low‑latency expectations. These characteristics align with the eMBB (enhanced Mobile Broadband) slice, which supports typical Internet services at rates of 100‑400 Mbps and latencies of 10‑100 ms.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 20.0 MHz
Data Rate: 100.0 Mbps
Latency: 30.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-04-05 23:36:51
Total Users: 4
Average Resource Utilization: 43.08%
eMBB Total Rate: 464.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  56.0/90 MHz       62.22%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 4 → eMBB Slice
CQI: 9, Bandwidth: 20.0 MHz, Rate: 100.00 Mbps, Latency: 30.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         10 |           200 |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |           100 |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |            64 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |           100 |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~349, completion~3032, total~3381

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 20.0

Intent Analysis: Remote video streaming from home security cameras
Recommended Slice: eMBB - The eMBB slice offers higher bandwidth (6‑20 MHz) and moderate latency (10‑100 ms), which are well‑suited for video streaming. It can support the required data rates while staying within its defined operational envelope.
Bandwidth Allocation: 5.0 MHz
Data Rate: 20.0 Mbps
Latency: 100.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-04-05 23:37:41
Total Users: 5
Average Resource Utilization: 46.92%
eMBB Total Rate: 484.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  61.0/90 MHz       67.78%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 5 → eMBB Slice
CQI: 11, Bandwidth: 5.0 MHz, Rate: 20.00 Mbps, Latency: 100.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         10 |           200 |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |           100 |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |            64 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |           100 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |            20 |            100 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I need to participate in a video conference meeting"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~349, completion~2541, total~2890

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 50.0

Intent Analysis: Participate in a video conference meeting. Typical video conference traffic requires a moderate‑to‑high data rate (several tens of Mbps) with tolerable latency on the order of tens of milliseconds.
Recommended Slice: eMBB - eMBB provides the necessary high data‑rate capability (100‑400 Mbps) and latency range (10‑100 ms) that comfortably accommodates a video conference. The slice still has ample free resources (≈29 MHz) to allocate a suitable bandwidth block without jeopardizing existing users.
Bandwidth Allocation: 20.0 MHz
Data Rate: 50.0 Mbps
Latency: 100.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-04-05 23:38:34
Total Users: 6
Average Resource Utilization: 62.31%
eMBB Total Rate: 534.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  81.0/90 MHz       90.00%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 6 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 50.00 Mbps, Latency: 100.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         10 |           200 |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |           100 |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |            64 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |           100 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |            20 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |            50 |            100 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to monitor water level in a reservoir"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~349, completion~1770, total~2119

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.0

Intent Analysis: The request involves low‑volume, periodic sensor data typical of IoT/machine‑type communications. The traffic does not require the high bandwidth or ultra‑low latency of eMBB or URLLC slices.
Recommended Slice: mMTC - mMTC slice supports low‑rate (0.1‑1 Mbps) and moderate‑latency (100‑1000 ms) IoT traffic. With CQI 6 the approximate spectral efficiency is 1.2 bits/s/Hz, so a 1 MHz allocation would yield ~1.2 Mbps. To stay within the mMTC rate limit, the effective rate is capped at the maximum allowed 1 Mbps. The mMTC slice currently has 0/10 MHz in use, leaving ample capacity (9 MHz remaining) for this and future IoT users.
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.0 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-04-05 23:39:10
Total Users: 7
Average Resource Utilization: 63.08%
eMBB Total Rate: 534.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  81.0/90 MHz       90.00%
URLLC          0  0/30 MHz          0%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 7 → mMTC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 1.00 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         10 |           200 |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |           100 |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |            64 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |           100 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |            20 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |            50 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |             1 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~350, completion~1988, total~2338

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 23.0

Intent Analysis: The user is a firefighter requiring reliable, low‑latency communication inside buildings. This implies mission‑critical data exchange (voice, telemetry, video) with stringent reliability and latency constraints rather than bulk throughput.
Recommended Slice: URLLC - URLLC provides the required latency (1‑10 ms) and reliability, and its spectrum is currently unused, allowing immediate allocation without impacting the heavily loaded eMBB slice.
Bandwidth Allocation: 5.0 MHz
Data Rate: 23.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-04-05 23:39:44
Total Users: 8
Average Resource Utilization: 66.92%
eMBB Total Rate: 534.00 Mbps, URLLC Total Rate: 23.00 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  81.0/90 MHz       90.00%
URLLC          1  5.0/30 MHz        16.67%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 9, Bandwidth: 5.0 MHz, Rate: 23.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         8 | URLLC   |     9 |          5 |            23 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |           200 |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |           100 |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |            64 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |           100 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |            20 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |            50 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |             1 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to update my social media status"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~2324, total~2676

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 9.0

Intent Analysis: N/A
Recommended Slice: URLLC - The request is a low‑to‑moderate data interactive update. eMBB is near capacity (90% utilization) leaving only 9 MHz spare, which would be insufficient to meet the minimum 100 Mbps eMBB rate while maintaining the required bandwidth. URLLC is lightly loaded (16.67% utilization) and offers ample headroom (25 MHz free). Assigning 2 MHz of URLLC spectrum yields ~9 Mbps—well within the 1‑100 Mbps URLLC rate window—while delivering low latency (≈5 ms), satisfying the user experience for an interactive status update. This choice also preserves eMBB resources for high‑throughput eMBB services.
Bandwidth Allocation: 2.0 MHz
Data Rate: 9.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-04-05 23:40:25
Total Users: 9
Average Resource Utilization: 68.46%
eMBB Total Rate: 534.00 Mbps, URLLC Total Rate: 32.00 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  81.0/90 MHz       90.00%
URLLC          2  7.0/30 MHz        23.33%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 9 → URLLC Slice
CQI: 12, Bandwidth: 2.0 MHz, Rate: 9.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         8 | URLLC   |     9 |          5 |            23 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |             9 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |           200 |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |           100 |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |            64 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |           100 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |            20 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |            50 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |             1 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I want to update my social media status"
CQI: 13
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~3428, total~3780

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 26.25

Intent Analysis: User request is a typical social‑media status update – a small payload (≈ 100‑500 KB) that can be transmitted with moderate data rate and is latency‑tolerant.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 26.25 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-04-05 23:41:27
Total Users: 10
Average Resource Utilization: 72.31%
eMBB Total Rate: 534.00 Mbps, URLLC Total Rate: 58.25 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  81.0/90 MHz       90.00%
URLLC          3  12.0/30 MHz       40.00%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 10 → URLLC Slice
CQI: 13, Bandwidth: 5.0 MHz, Rate: 26.25 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need vehicle-to-vehicle collision avoidance systems"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2134, total~2487

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 26.0

Intent Analysis: The user needs a V2V collision‑avoidance system, which is a safety‑critical service requiring ultra‑reliable low‑latency communications (URLLC) to guarantee rapid exchange of sensor and control messages between vehicles.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 26.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-04-05 23:42:01
Total Users: 11
Average Resource Utilization: 76.15%
eMBB Total Rate: 534.00 Mbps, URLLC Total Rate: 84.25 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  81.0/90 MHz       90.00%
URLLC          4  17.0/30 MHz       56.67%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 14, Bandwidth: 5.0 MHz, Rate: 26.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |    14 |          5 |         26    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need to sync my calendar and contacts"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~1521, total~1873

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: N/A
Recommended Slice: eMBB - The sync request is low‑bandwidth and latency‑tolerant. The mMTC slice can accommodate the traffic while relieving the heavily utilized eMBB slice. The chosen 1 MHz bandwidth and 0.5 Mbps rate stay within the mMTC limits (1‑3 MHz, 0.1‑1 Mbps) and keep latency at an acceptable 200 ms.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-04-05 23:42:30
Total Users: 12
Average Resource Utilization: 76.92%
eMBB Total Rate: 534.50 Mbps, URLLC Total Rate: 84.25 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  82.0/90 MHz       91.11%
URLLC          4  17.0/30 MHz       56.67%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 12 → eMBB Slice
CQI: 5, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |    14 |          5 |         26    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |          1 |          0.5  |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~1481, total~1834

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 5.0

Intent Analysis: Safety-critical machine shutdown requiring ultra-low latency communication for immediate control response
Recommended Slice: URLLC - Immediate machine shutdown for safety incidents requires ultra-reliable, low-latency communication (URLLC). The 1-10ms latency window is essential for rapid safety response. Control commands for emergency shutdown are typically short data bursts with high reliability requirements.
Bandwidth Allocation: 2.0 MHz
Data Rate: 5.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-04-05 23:42:59
Total Users: 13
Average Resource Utilization: 78.46%
eMBB Total Rate: 534.50 Mbps, URLLC Total Rate: 89.25 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  82.0/90 MHz       91.11%
URLLC          5  19.0/30 MHz       63.33%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 4, Bandwidth: 2.0 MHz, Rate: 5.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |    14 |          5 |         26    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |          2 |          5    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I want to use holographic communication"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~4628, total~4979

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 8.0, rate: 10.6

Intent Analysis: N/A
Recommended Slice: eMBB - Holographic communication requires high data throughput, classifying it as an eMBB (Enhanced Mobile Broadband) service. The eMBB slice has 8 MHz of remaining capacity. With a CQI of 4 (indicating low SNR, mapping to 16-QAM with low code rate), the spectral efficiency is approximately 1.33 bits/Hz. The calculated achievable data rate is 10.6 Mbps. While this rate is below the ideal eMBB peak rates (100-400 Mbps), it represents the best-effort high-bandwidth connection possible under current channel and network load conditions.
Bandwidth Allocation: 8.0 MHz
Data Rate: 10.6 Mbps
Latency: 100.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-04-05 23:44:15
Total Users: 14
Average Resource Utilization: 84.62%
eMBB Total Rate: 545.10 Mbps, URLLC Total Rate: 89.25 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          5  19.0/30 MHz       63.33%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 14 → eMBB Slice
CQI: 4, Bandwidth: 8.0 MHz, Rate: 10.60 Mbps, Latency: 100.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |    14 |          5 |         26    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |     4 |          8 |         10.6  |            100 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use maps for basic navigation"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~1789, total~2141

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 11.5

Intent Analysis: N/A
Recommended Slice: eMBB - eMBB at 100% capacity. URLLC slice has available resources (11 MHz free) to accommodate moderate bandwidth needs via dynamic sharing.
Bandwidth Allocation: 5.0 MHz
Data Rate: 11.5 Mbps
Latency: 8.0 ms

CAPACITY CHECK FAILED:
- Required: 5.0 MHz, Available: 0.0 MHz in eMBB slice

----------------------------------------
ALLOCATION FAILED FOR USER 15
----------------------------------------
Request: I need to use maps for basic navigation
Slice type: eMBB
Reason: Insufficient capacity in eMBB slice. Required: 5.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I want to use remote surgery equipment"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~2626, total~2977

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 10.0

Intent Analysis: The user (ID 16) at location (91.72, -239.92, 1.5) has requested remote surgery equipment. Remote surgery demands very low latency and high reliability, which are hallmarks of the URLLC (Ultra‑Reliable Low‑Latency Communications) slice. The request does not involve massive machine‑type traffic (no IoT sensors) nor extremely high peak rates typical of eMBB, so URLLC is the optimal slice.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 10.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-04-05 23:45:33
Total Users: 15
Average Resource Utilization: 88.46%
eMBB Total Rate: 545.10 Mbps, URLLC Total Rate: 99.25 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          6  24.0/30 MHz       80.00%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 8, Bandwidth: 5.0 MHz, Rate: 10.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |    14 |          5 |         26    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |          5 |         10    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |     4 |          8 |         10.6  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to participate in a video conference meeting"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~1485, total~1838

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 50.0

Intent Analysis: Video conference meeting - real-time interactive communication requiring moderate bandwidth and low latency for smooth audio/video transmission
Recommended Slice: URLLC - Video conference requires real-time communication with latency <50ms; eMBB slice is 100% utilized (90/90 MHz) - no capacity available; URLLC slice has 6 MHz available (24/30 MHz utilized); mMTC slice has high latency (100-1000ms) unsuitable for video conferencing; URLLC latency (1-10ms) meets real-time communication requirements
Bandwidth Allocation: 2.0 MHz
Data Rate: 50.0 Mbps
Latency: 50.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-04-05 23:45:59
Total Users: 16
Average Resource Utilization: 90.0%
eMBB Total Rate: 545.10 Mbps, URLLC Total Rate: 149.25 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          7  26.0/30 MHz       86.67%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 17 → URLLC Slice
CQI: 7, Bandwidth: 2.0 MHz, Rate: 50.00 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |    14 |          5 |         26    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         50    |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |     4 |          8 |         10.6  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~356, completion~1395, total~1751

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 10.0

Intent Analysis: Control precision CNC machines with zero tolerance for delay
Recommended Slice: URLLC - Zero tolerance for delay -> requires URLLC's 1-10ms latency (vs eMBB's 10-100ms); CNC control commands are small data packets but require immediate delivery; URLLC provides the reliability and low latency required for precision control; Control signals typically require 1-10 Mbps, well within URLLC's 1-100 Mbps rate capability
Bandwidth Allocation: 2.0 MHz
Data Rate: 10.0 Mbps
Latency: 1.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-04-05 23:46:24
Total Users: 17
Average Resource Utilization: 91.54%
eMBB Total Rate: 545.10 Mbps, URLLC Total Rate: 159.25 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          8  28.0/30 MHz       93.33%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 4, Bandwidth: 2.0 MHz, Rate: 10.00 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |    14 |          5 |         26    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         50    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |          2 |         10    |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |     4 |          8 |         10.6  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~355, completion~1817, total~2172

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 5.0

Intent Analysis: N/A
Recommended Slice: eMBB - Image processing requires high bandwidth (typical eMBB characteristic); CQI of 11 indicates good channel conditions suitable for high-order modulation; Cloud AI services have moderate latency requirements (10-100ms) matching eMBB specs; Not ultra-reliable low latency traffic (excluded from URLLC); Not massive machine-type communication (excluded from mMTC)
Bandwidth Allocation: 10.0 MHz
Data Rate: 5.0 Mbps
Latency: 25.0 ms

CAPACITY CHECK FAILED:
- Required: 10.0 MHz, Available: 0.0 MHz in eMBB slice

----------------------------------------
ALLOCATION FAILED FOR USER 19
----------------------------------------
Request: I need to use cloud-based AI services for image processing
Slice type: eMBB
Reason: Insufficient capacity in eMBB slice. Required: 10.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I need instant alerts for life-threatening patient conditions"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2946, total~3299

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 5.0

Intent Analysis: The user requests instant alerts for life‑threatening patient conditions. This is a safety‑critical, latency‑sensitive message that requires ultra‑reliable low‑latency communication (URLLC).
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 2.0 MHz
Data Rate: 5.0 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 2.0 MHz, Available: 0.0 MHz in eMBB slice

----------------------------------------
ALLOCATION FAILED FOR USER 20
----------------------------------------
Request: I need instant alerts for life-threatening patient conditions
Slice type: eMBB
Reason: Insufficient capacity in eMBB slice. Required: 2.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2799, total~3152

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: The sensor will transmit small, periodic soil‑temperature readings, which is a classic Machine‑Type Communication (MTC) workload with modest data volume and latency tolerance.
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-04-05 23:48:32
Total Users: 18
Average Resource Utilization: 92.31%
eMBB Total Rate: 545.10 Mbps, URLLC Total Rate: 159.25 Mbps, mMTC Total Rate: 1.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          8  28.0/30 MHz       93.33%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 21 → mMTC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |    14 |          5 |         26    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         50    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |          2 |         10    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |     4 |          8 |         10.6  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |          1 |          0.5  |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "I need to synchronize multiple robots on a factory floor"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~2237, total~2591

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 5.0

Intent Analysis: The user requires real‑time, low‑latency communication to coordinate several robots. This is a typical URLLC use case where tight timing and reliability are more critical than sheer throughput.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 2.0 MHz
Data Rate: 5.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-04-05 23:49:12
Total Users: 19
Average Resource Utilization: 93.85%
eMBB Total Rate: 545.10 Mbps, URLLC Total Rate: 164.25 Mbps, mMTC Total Rate: 1.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 22 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 5.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |    14 |          5 |         26    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         50    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |          2 |         10    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |          2 |          5    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |     4 |          8 |         10.6  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to control a robotic arm in real time"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~3048, total~3402

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 4.0

Intent Analysis: Low‑latency, high‑reliability command/feedback loop for motion control
Recommended Slice: URLLC - Robotic‑arm control requires end‑to‑end latency ≤10 ms and a reliable data path, which is the defining characteristic of the URLLC slice
Bandwidth Allocation: 2.0 MHz
Data Rate: 4.0 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 2.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 23
----------------------------------------
Request: I need to control a robotic arm in real time
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 2.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~3896, total~4250

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 5.5

Intent Analysis: N/A
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 3.0 MHz
Data Rate: 5.5 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 3.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 24
----------------------------------------
Request: I need to detect and isolate power grid faults instantly
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 3.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 2
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~1885, total~2239

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: Environmental sensors sending periodic, small‑volume air‑quality reports represent a classic IoT/machine‑type traffic pattern. This type of traffic is best served by the mMTC slice, which is designed for massive machine communications with low bandwidth, low data rates, and tolerance for higher latency. The current network state shows the eMBB and URLLC slices are fully saturated (100 % utilization), while the mMTC slice is only at 20 % utilization, leaving ample room for the new user.
Recommended Slice: mMTC - Matches the low‑rate, low‑bandwidth, and latency‑tolerant nature of sensor data; leverages the only slice with available capacity.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 300.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-04-05 23:51:47
Total Users: 20
Average Resource Utilization: 94.62%
eMBB Total Rate: 545.10 Mbps, URLLC Total Rate: 164.25 Mbps, mMTC Total Rate: 2.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 25 → mMTC Slice
CQI: 2, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 300.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |    14 |          5 |         26    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         50    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |          2 |         10    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |     4 |          8 |         10.6  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |          1 |          0.5  |            300 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 1
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~356, completion~1208, total~1564

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.15

Intent Analysis: N/A
Recommended Slice: mMTC - The request is for low‑rate telemetry from a drone fleet, which best matches the mMTC slice (low‑rate, high‑latency tolerance). The eMBB and URLLC slices are fully utilized, while mMTC has 7 MHz of free capacity. A CQI of 1 indicates poor channel quality, so a conservative 1 MHz allocation with a modest data rate (~0.15 Mbps) is appropriate. This respects the mMTC constraints (bandwidth 1‑3 MHz, rate 0.1‑1 Mbps, latency 100‑1000 ms) and leaves sufficient resources for other mMTC users.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.15 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-04-05 23:52:12
Total Users: 21
Average Resource Utilization: 95.38%
eMBB Total Rate: 545.10 Mbps, URLLC Total Rate: 164.25 Mbps, mMTC Total Rate: 2.15 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           4  4.0/10 MHz        40.00%

New User Allocation:
User 26 → mMTC Slice
CQI: 1, Bandwidth: 1.0 MHz, Rate: 0.15 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |    14 |          5 |         26    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         50    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |          2 |         10    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |     4 |          8 |         10.6  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |          1 |          0.5  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |          1 |          0.15 |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "My smart meter needs to report its reading"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~4055, total~4407

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.4766

Intent Analysis: Low‑volume, periodic IoT telemetry from a smart meter
Recommended Slice: mMTC - Raw rate (1.48 Mbps) exceeds mMTC maximum of 1 Mbps; cap applied to meet slice constraints
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.4766 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-04-05 23:53:13
Total Users: 22
Average Resource Utilization: 96.15%
eMBB Total Rate: 545.10 Mbps, URLLC Total Rate: 164.25 Mbps, mMTC Total Rate: 3.63 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           5  5.0/10 MHz        50.00%

New User Allocation:
User 27 → mMTC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 1.48 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |    14 |          5 |         26    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         50    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |          2 |         10    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |     4 |          8 |         10.6  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |          1 |          0.5  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |          1 |          0.15 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |          1 |          1.48 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~349, completion~2164, total~2513

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.377

Intent Analysis: Low‑rate, periodic sensor data from industrial equipment → IoT/mMTC‑type traffic
Recommended Slice: mMTC - 1 MHz provides robust coverage for CQI 3, stays within mMTC rate limits (0.1‑1 Mbps) and latency constraints (100‑1000 ms), and leaves headroom for future devices
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.377 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-04-05 23:53:43
Total Users: 23
Average Resource Utilization: 96.92%
eMBB Total Rate: 545.10 Mbps, URLLC Total Rate: 164.25 Mbps, mMTC Total Rate: 4.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           6  6.0/10 MHz        60.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 0.38 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |    14 |          5 |         26    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         50    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |          2 |         10    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |     4 |          8 |         10.6  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |          1 |          0.5  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |          1 |          0.15 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |          1 |          1.48 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     3 |          1 |          0.38 |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~1184, total~1537

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: N/A
Recommended Slice: mMTC - Sensor data transmission with low data volume; High latency tolerance (agricultural monitoring); Periodic reporting pattern typical of IoT devices; mMTC slice designed for massive machine-type communications; Cost-effective solution for low-bandwidth IoT applications
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 100.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-04-05 23:54:03
Total Users: 24
Average Resource Utilization: 97.69%
eMBB Total Rate: 545.10 Mbps, URLLC Total Rate: 164.25 Mbps, mMTC Total Rate: 4.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           7  7.0/10 MHz        70.00%

New User Allocation:
User 29 → mMTC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 100.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    13 |          5 |         26.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |    14 |          5 |         26    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         50    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |          2 |         10    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |          5 |         23    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |    12 |          2 |          9    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        200    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |     4 |          8 |         10.6  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |          6 |         64    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        100    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |    11 |          5 |         20    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |         50    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |          1 |          0.5  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |          1 |          0.15 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |          1 |          1.48 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     3 |          1 |          0.38 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | mMTC    |     9 |          1 |          0.5  |            100 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I want to use remote surgery equipment"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~3296, total~3647

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 25.0

Intent Analysis: Remote surgery equipment requires ultra‑reliable, low‑latency communication with sufficient bandwidth to carry high‑definition video, haptic feedback and control signals.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 25.0 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 5.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 30
----------------------------------------
Request: I want to use remote surgery equipment
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 5.0 MHz, Available: 0.0 MHz

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |    15 |         10 |      200      |             25 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | eMBB    | eMBB           | Yes            |     4 |         20 |      100      |             20 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | eMBB    | eMBB           | Yes            |    15 |          6 |       64      |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | eMBB    | eMBB           | Yes            |     9 |         20 |      100      |             30 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | eMBB    | eMBB           | Yes            |    11 |          5 |       20      |            100 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | eMBB    | eMBB           | Yes            |    12 |         20 |       50      |            100 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | mMTC    | mMTC           | Yes            |     6 |          1 |        1      |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | URLLC          | Yes            |     9 |          5 |       23      |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | URLLC   | eMBB           | No             |    12 |          2 |        9      |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | URLLC   | eMBB           | No             |    13 |          5 |       26.25   |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |    14 |          5 |       26      |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | eMBB    | eMBB           | Yes            |     5 |          1 |        0.5    |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | URLLC          | Yes            |     4 |          2 |        5      |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | eMBB    | eMBB           | Yes            |     4 |          8 |       10.6    |            100 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Failed   | eMBB    | eMBB           |                |     7 |          5 |       11.5    |              8 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |     8 |          5 |       10      |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | URLLC   | eMBB           | No             |     7 |          2 |       50      |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     4 |          2 |       10      |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Failed   | eMBB    | eMBB           |                |    11 |         10 |        5      |             25 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Failed   | eMBB    | URLLC          |                |     9 |          2 |        5      |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | mMTC    | mMTC           | Yes            |     7 |          1 |        0.5    |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | URLLC   | URLLC          | Yes            |     9 |          2 |        5      |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Failed   | URLLC   | URLLC          |                |     4 |          2 |        4      |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Failed   | URLLC   | URLLC          |                |     5 |          3 |        5.5    |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | mMTC    | mMTC           | Yes            |     2 |          1 |        0.5    |            300 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | mMTC    | mMTC           | Yes            |     1 |          1 |        0.15   |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | mMTC    | mMTC           | Yes            |     7 |          1 |        1.4766 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | mMTC    | mMTC           | Yes            |     3 |          1 |        0.377  |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | mMTC    | mMTC           | Yes            |     9 |          1 |        0.5    |            100 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Failed   | URLLC   | URLLC          |                |     8 |          5 |       25      |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 24/30 (80.0%)

Intent Understanding Evaluation:
Correctly identified intents: 21/24
Intent understanding rate: 87.5%

Workload Balancing Statistics:
Users with workload balancing: 24/30
Workload balancing rate: 80.0%

Slice Utilization Statistics:
Average eMBB utilization: 84.86%
Average URLLC utilization: 53.06%
Average mMTC utilization: 16.67%

Results exported to F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_east_minimax-m2.5.csv

[OK] TJU_east 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_east_minimax-m2.5.csv