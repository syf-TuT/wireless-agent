============================================================
场景 5/5: TJU_gym
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_gym.csv
输出: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_gym_minimax-m2.5.csv
------------------------------------------------------------
Starting prompt-based network slice management system...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~348, completion~1345, total~1693

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 0.3

Intent Analysis: Environmental sensor networks typically transmit low‑volume data (e.g., temperature, humidity, pollutant concentrations) and can tolerate higher latency. This traffic pattern aligns perfectly with the characteristics of the mMTC (massive Machine‑Type Communication) slice, which is designed for many low‑power IoT devices with relaxed latency requirements.
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 2.0 MHz
Data Rate: 0.3 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-04-06 00:12:55
Total Users: 1
Average Resource Utilization: 1.54%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.30 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          0  0/30 MHz          0%
mMTC           1  2.0/10 MHz        20.00%

New User Allocation:
User 1 → mMTC Slice
CQI: 3, Bandwidth: 2.0 MHz, Rate: 0.30 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | mMTC    |     3 |          2 |           0.3 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I want to browse websites and check email"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~348, completion~1319, total~1667

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 200.0

Intent Analysis: N/A
Recommended Slice: eMBB - The user’s activity (web browsing and email) requires moderate bandwidth and latency within the eMBB range. A 10 MHz allocation yields an estimated 200 Mbps data rate, satisfying the 100‑400 Mbps requirement while keeping latency around 30 ms, well within the 10‑100 ms eMBB window. The eMBB slice currently has sufficient idle capacity (0/90 MHz), so the allocation leaves ample resources for future users and maintains balanced load across slices.
Bandwidth Allocation: 10.0 MHz
Data Rate: 200.0 Mbps
Latency: 30.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-04-06 00:13:16
Total Users: 2
Average Resource Utilization: 9.23%
eMBB Total Rate: 200.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.30 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  10.0/90 MHz       11.11%
URLLC          0  0/30 MHz          0%
mMTC           1  2.0/10 MHz        20.00%

New User Allocation:
User 2 → eMBB Slice
CQI: 11, Bandwidth: 10.0 MHz, Rate: 200.00 Mbps, Latency: 30.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |         10 |         200   |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |           0.3 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to monitor water level in a reservoir"
CQI: 13
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~2503, total~2854

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: Monitor water level in a reservoir
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-04-06 00:13:52
Total Users: 3
Average Resource Utilization: 10.0%
eMBB Total Rate: 200.50 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.30 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  11.0/90 MHz       12.22%
URLLC          0  0/30 MHz          0%
mMTC           1  2.0/10 MHz        20.00%

New User Allocation:
User 3 → eMBB Slice
CQI: 13, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |         10 |         200   |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |           0.5 |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |           0.3 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~2304, total~2655

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.877

Intent Analysis: The user’s request describes a typical IoT health‑monitoring wearable that periodically uploads small amounts of telemetry (e.g., heart rate, SpO2). This traffic is low‑rate, latency‑tolerant, and best served by an mMTC‑type slice rather than high‑speed eMBB or ultra‑reliable URLLC services.
Recommended Slice: mMTC - A 1 MHz grant keeps the data rate below the 1 Mbps mMTC ceiling while providing enough capacity for periodic health‑data uploads. The latency of ≈200 ms is well within the 100‑1000 ms mMTC tolerance.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.877 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-04-06 00:14:31
Total Users: 4
Average Resource Utilization: 10.77%
eMBB Total Rate: 200.50 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 1.18 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  11.0/90 MHz       12.22%
URLLC          0  0/30 MHz          0%
mMTC           2  3.0/10 MHz        30.00%

New User Allocation:
User 4 → mMTC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 0.88 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~6038, total~6391

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 48.12

Intent Analysis: cloud‑based AI services for image processing – high data rate, moderate latency (≈30 ms)
Recommended Slice: eMBB - eMBB provides the bandwidth (6‑20 MHz) and latency (10‑100 ms) needed for cloud‑based AI image processing. The URLLC slice targets ultra‑reliable low‑latency traffic (<10 ms) and the mMTC slice is optimised for massive machine‑type connectivity, both of which are unnecessary for this request.
Bandwidth Allocation: 20.0 MHz
Data Rate: 48.12 Mbps
Latency: 100.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-04-06 00:16:02
Total Users: 5
Average Resource Utilization: 26.15%
eMBB Total Rate: 248.62 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 1.18 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  31.0/90 MHz       34.44%
URLLC          0  0/30 MHz          0%
mMTC           2  3.0/10 MHz        30.00%

New User Allocation:
User 5 → eMBB Slice
CQI: 8, Bandwidth: 20.0 MHz, Rate: 48.12 Mbps, Latency: 100.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I need to stream 8K video content"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~3006, total~3357

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 12.0, rate: 120.0

Intent Analysis: Stream 8K video content
Recommended Slice: eMBB - CQI 5 indicates moderate channel quality; allocating the middle of the eMBB per‑user bandwidth range (12 MHz) provides the minimum required 100 Mbps for 8K streaming while preserving capacity for other eMBB users.
Bandwidth Allocation: 12.0 MHz
Data Rate: 120.0 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-04-06 00:16:50
Total Users: 6
Average Resource Utilization: 35.38%
eMBB Total Rate: 368.62 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 1.18 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  43.0/90 MHz       47.78%
URLLC          0  0/30 MHz          0%
mMTC           2  3.0/10 MHz        30.00%

New User Allocation:
User 6 → eMBB Slice
CQI: 5, Bandwidth: 12.0 MHz, Rate: 120.00 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to track the location of a shipping container"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~2022, total~2374

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.1

Intent Analysis: The user requires periodic transmission of geographic position data for a shipping container. This is a low‑volume, latency‑tolerant IoT use case.
Recommended Slice: mMTC - Tracking a shipping container generates infrequent, small packets → ideal for massive Machine‑Type Communications.; The required data rate (0.1–1 Mbps) fits the mMTC slice spec (0.1–1 Mbps).; Latency tolerance (100–1000 ms) aligns with mMTC capabilities.; Allocating to mMTC leaves eMBB resources for high‑bandwidth services and preserves URLLC for ultra‑reliable low‑latency traffic.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.1 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-04-06 00:17:20
Total Users: 7
Average Resource Utilization: 36.15%
eMBB Total Rate: 368.62 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 1.28 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  43.0/90 MHz       47.78%
URLLC          0  0/30 MHz          0%
mMTC           3  4.0/10 MHz        40.00%

New User Allocation:
User 7 → mMTC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 0.10 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~2648, total~2999

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: The request comes from a wearable health‑monitoring device that periodically uploads small amounts of data (e.g., heart rate, steps). The traffic is low‑rate, delay‑tolerant, and originates from an IoT‑type device, which best matches the characteristics of the mMTC slice.
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-04-06 00:17:57
Total Users: 8
Average Resource Utilization: 36.92%
eMBB Total Rate: 368.62 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 1.78 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  43.0/90 MHz       47.78%
URLLC          0  0/30 MHz          0%
mMTC           4  5.0/10 MHz        50.00%

New User Allocation:
User 8 → mMTC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I need reliable connectivity for implanted medical devices"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~350, completion~2338, total~2688

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 10.0

Intent Analysis: The user requires highly reliable connectivity for implanted medical devices, emphasizing low latency and robust performance rather than high throughput.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 10.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-04-06 00:18:37
Total Users: 9
Average Resource Utilization: 40.77%
eMBB Total Rate: 368.62 Mbps, URLLC Total Rate: 10.00 Mbps, mMTC Total Rate: 1.78 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  43.0/90 MHz       47.78%
URLLC          1  5.0/30 MHz        16.67%
mMTC           4  5.0/10 MHz        50.00%

New User Allocation:
User 9 → URLLC Slice
CQI: 7, Bandwidth: 5.0 MHz, Rate: 10.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         9 | URLLC   |     7 |          5 |         10    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to check weather forecasts"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~350, completion~2922, total~3272

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 120.0

Intent Analysis: Data‑centric service (weather data retrieval) with moderate latency tolerance
Recommended Slice: eMBB - Weather‑forecast access is a broadband data service that fits the eMBB profile (typical bandwidth 6‑20 MHz, rate 100‑400 Mbps, latency 10‑100 ms).; The URLLC slice targets ultra‑reliable low‑latency traffic and would over‑provision for this request.; The mMTC slice is designed for massive machine‑type connectivity with very low data rates (0.1‑1 Mbps) and high latency (100‑1000 ms), which is insufficient for timely weather updates.
Bandwidth Allocation: 10.0 MHz
Data Rate: 120.0 Mbps
Latency: 30.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-04-06 00:19:22
Total Users: 10
Average Resource Utilization: 48.46%
eMBB Total Rate: 488.62 Mbps, URLLC Total Rate: 10.00 Mbps, mMTC Total Rate: 1.78 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  53.0/90 MHz       58.89%
URLLC          1  5.0/30 MHz        16.67%
mMTC           4  5.0/10 MHz        50.00%

New User Allocation:
User 10 → eMBB Slice
CQI: 5, Bandwidth: 10.0 MHz, Rate: 120.00 Mbps, Latency: 30.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         9 | URLLC   |     7 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |        120    |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~356, completion~1680, total~2036

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 0.8

Intent Analysis: IoT sensor data collection
Recommended Slice: mMTC - The request involves monitoring city-wide smart streetlights, which are IoT devices requiring machine-to-machine communication. mMTC slice is designed for massive IoT connectivity with low-to-moderate data rates and can tolerate higher latency, making it ideal for this use case.
Bandwidth Allocation: 2.0 MHz
Data Rate: 0.8 Mbps
Latency: 250.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-04-06 00:19:50
Total Users: 11
Average Resource Utilization: 50.0%
eMBB Total Rate: 488.62 Mbps, URLLC Total Rate: 10.00 Mbps, mMTC Total Rate: 2.58 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  53.0/90 MHz       58.89%
URLLC          1  5.0/30 MHz        16.67%
mMTC           5  7.0/10 MHz        70.00%

New User Allocation:
User 11 → mMTC Slice
CQI: 6, Bandwidth: 2.0 MHz, Rate: 0.80 Mbps, Latency: 250.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         9 | URLLC   |     7 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |          0.8  |            250 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need microsecond-level latency for high-frequency tradin"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~355, completion~1798, total~2153

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 25.0

Intent Analysis: critical
Recommended Slice: URLLC - The calculated 25 Mbps (5 MHz × 5 bits/Hz) fits comfortably within the URLLC rate window (1‑100 Mbps) and provides sufficient throughput for typical high‑frequency trading payloads while keeping queuing latency minimal.
Bandwidth Allocation: 5.0 MHz
Data Rate: 25.0 Mbps
Latency: 1.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-04-06 00:20:23
Total Users: 12
Average Resource Utilization: 53.85%
eMBB Total Rate: 488.62 Mbps, URLLC Total Rate: 35.00 Mbps, mMTC Total Rate: 2.58 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  53.0/90 MHz       58.89%
URLLC          2  10.0/30 MHz       33.33%
mMTC           5  7.0/10 MHz        70.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 3, Bandwidth: 5.0 MHz, Rate: 25.00 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |         25    |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |          0.8  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I want to use holographic communication"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~3671, total~4022

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 100.0

Intent Analysis: High‑bandwidth, real‑time holographic communication
Recommended Slice: eMBB - eMBB provides the high throughput (100‑400 Mbps) and latency (10‑100 ms) needed for holographic communication, while CQI 9 still supports adequate modulation.
Bandwidth Allocation: 10.0 MHz
Data Rate: 100.0 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-04-06 00:21:23
Total Users: 13
Average Resource Utilization: 61.54%
eMBB Total Rate: 588.62 Mbps, URLLC Total Rate: 35.00 Mbps, mMTC Total Rate: 2.58 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  63.0/90 MHz       70.00%
URLLC          2  10.0/30 MHz       33.33%
mMTC           5  7.0/10 MHz        70.00%

New User Allocation:
User 13 → eMBB Slice
CQI: 9, Bandwidth: 10.0 MHz, Rate: 100.00 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |         25    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |         10 |        100    |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |          0.8  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 12
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2992, total~3345

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 15.0

Intent Analysis: Requires ultra‑low latency, high reliability, and minimal interruption; best served by the URLLC slice.
Recommended Slice: eMBB - Bandwidth set to the maximum allowed for URLLC (5 MHz) to maximise reliability. Data rate derived from CQI 12 (≈3 bits/Hz) × 5 MHz ≈ 15 Mbps, well within the URLLC 1‑100 Mbps range. Latency set to 2 ms (within 1‑10 ms constraint).
Bandwidth Allocation: 5.0 MHz
Data Rate: 15.0 Mbps
Latency: 2.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-04-06 00:22:09
Total Users: 14
Average Resource Utilization: 65.38%
eMBB Total Rate: 603.62 Mbps, URLLC Total Rate: 35.00 Mbps, mMTC Total Rate: 2.58 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  68.0/90 MHz       75.56%
URLLC          2  10.0/30 MHz       33.33%
mMTC           5  7.0/10 MHz        70.00%

New User Allocation:
User 14 → eMBB Slice
CQI: 12, Bandwidth: 5.0 MHz, Rate: 15.00 Mbps, Latency: 2.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |         25    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |         10 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |    12 |          5 |         15    |              2 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |          0.8  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need real-time fraud detection for financial transactions"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~1878, total~2231

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 15.0

Intent Analysis: Low‑latency, high‑reliability processing of streaming transaction data for fraud detection
Recommended Slice: eMBB - URLLC slice provides ultra‑low latency (1‑10 ms) and can support the required data rate, whereas eMBB targets higher rates with higher latency and mMTC targets massive IoT with much higher latency.
Bandwidth Allocation: 5.0 MHz
Data Rate: 15.0 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-04-06 00:22:47
Total Users: 15
Average Resource Utilization: 69.23%
eMBB Total Rate: 618.62 Mbps, URLLC Total Rate: 35.00 Mbps, mMTC Total Rate: 2.58 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  73.0/90 MHz       81.11%
URLLC          2  10.0/30 MHz       33.33%
mMTC           5  7.0/10 MHz        70.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 7, Bandwidth: 5.0 MHz, Rate: 15.00 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |         25    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |         10 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |    12 |          5 |         15    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |          5 |         15    |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |          0.8  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need instant facial recognition for public security threats"
CQI: 10
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~1683, total~2036

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 50.0

Intent Analysis: N/A
Recommended Slice: URLLC - Security applications require ultra-low latency for real-time threat response. Facial recognition involves video capture, transmission, processing, and matching - all needing minimal delay to be effective in threat scenarios.
Bandwidth Allocation: 10.0 MHz
Data Rate: 50.0 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-04-06 00:23:13
Total Users: 16
Average Resource Utilization: 76.92%
eMBB Total Rate: 618.62 Mbps, URLLC Total Rate: 85.00 Mbps, mMTC Total Rate: 2.58 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  73.0/90 MHz       81.11%
URLLC          3  20.0/30 MHz       66.67%
mMTC           5  7.0/10 MHz        70.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 10, Bandwidth: 10.0 MHz, Rate: 50.00 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |         25    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |         10 |         50    |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |         10 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |    12 |          5 |         15    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |          5 |         15    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |          0.8  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "My smart trash can needs to signal that it's full"
CQI: 13
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~355, completion~2020, total~2375

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 3.0

Intent Analysis: The request comes from a smart trash can that periodically needs to transmit a tiny status message (e.g., 'full'). This is a low‑throughput, delay‑tolerant IoT use case that fits best with massive Machine‑Type Communications (mMTC).
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 3.0 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-04-06 00:23:51
Total Users: 17
Average Resource Utilization: 77.69%
eMBB Total Rate: 618.62 Mbps, URLLC Total Rate: 85.00 Mbps, mMTC Total Rate: 5.58 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  73.0/90 MHz       81.11%
URLLC          3  20.0/30 MHz       66.67%
mMTC           6  8.0/10 MHz        80.00%

New User Allocation:
User 17 → mMTC Slice
CQI: 13, Bandwidth: 1.0 MHz, Rate: 3.00 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |         25    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |         10 |         50    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |         10 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |    12 |          5 |         15    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |          5 |         15    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |          0.8  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |          1 |          3    |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~1690, total~2042

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 1.2

Intent Analysis: The request originates from a firefighter operating inside a building. Such mission‑critical communications require ultra‑reliable, low‑latency connectivity to guarantee safety and coordination. This aligns with the characteristics of a URLLC (Ultra‑Reliable Low‑Latency Communications) slice rather than eMBB (high‑throughput) or mMTC (massive machine‑type).
Recommended Slice: URLLC - CQI = 3 denotes a poor radio channel. Using a narrower bandwidth (2 MHz) reduces scheduling complexity and improves reliability while still delivering a data rate above the minimum URLLC requirement (1 Mbps). The chosen latency (5 ms) stays comfortably within the 1‑10 ms URLLC window.
Bandwidth Allocation: 2.0 MHz
Data Rate: 1.2 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-04-06 00:24:19
Total Users: 18
Average Resource Utilization: 79.23%
eMBB Total Rate: 618.62 Mbps, URLLC Total Rate: 86.20 Mbps, mMTC Total Rate: 5.58 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  73.0/90 MHz       81.11%
URLLC          4  22.0/30 MHz       73.33%
mMTC           6  8.0/10 MHz        80.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 3, Bandwidth: 2.0 MHz, Rate: 1.20 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |         25    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |         10 |         50    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |          2 |          1.2  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |         10 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |    12 |          5 |         15    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |          5 |         15    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |          0.8  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |          1 |          3    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to monitor water level in a reservoir"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2014, total~2367

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.33

Intent Analysis: User requires remote monitoring of a reservoir water level. This is a typical IoT sensor application that transmits small, periodic measurement packets and does not need high throughput or ultra‑low latency.
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.33 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-04-06 00:24:52
Total Users: 19
Average Resource Utilization: 80.0%
eMBB Total Rate: 618.62 Mbps, URLLC Total Rate: 86.20 Mbps, mMTC Total Rate: 5.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  73.0/90 MHz       81.11%
URLLC          4  22.0/30 MHz       73.33%
mMTC           7  9.0/10 MHz        90.00%

New User Allocation:
User 19 → mMTC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 0.33 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |         25    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |         10 |         50    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |          2 |          1.2  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |         10 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |    12 |          5 |         15    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |          5 |         15    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |          0.8  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |          1 |          3    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |          1 |          0.33 |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I need to make a high-quality voice call"
CQI: 10
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2461, total~2814

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 7.2

Intent Analysis: high-quality voice call
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 2.0 MHz
Data Rate: 7.2 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-04-06 00:25:33
Total Users: 20
Average Resource Utilization: 81.54%
eMBB Total Rate: 625.82 Mbps, URLLC Total Rate: 86.20 Mbps, mMTC Total Rate: 5.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           9  75.0/90 MHz       83.33%
URLLC          4  22.0/30 MHz       73.33%
mMTC           7  9.0/10 MHz        90.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 10, Bandwidth: 2.0 MHz, Rate: 7.20 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |         25    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |         10 |         50    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |          2 |          1.2  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |         10 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |    12 |          5 |         15    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |          5 |         15    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |          2 |          7.2  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |          0.8  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |          1 |          3    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |          1 |          0.33 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to use holographic communication"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~1258, total~1609

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 45.0

Intent Analysis: N/A
Recommended Slice: eMBB - Holographic communication requires real-time transmission of 3D visual data with high throughput for immersive experience
Bandwidth Allocation: 10.0 MHz
Data Rate: 45.0 Mbps
Latency: 15.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-04-06 00:25:54
Total Users: 21
Average Resource Utilization: 89.23%
eMBB Total Rate: 670.82 Mbps, URLLC Total Rate: 86.20 Mbps, mMTC Total Rate: 5.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB          10  85.0/90 MHz       94.44%
URLLC          4  22.0/30 MHz       73.33%
mMTC           7  9.0/10 MHz        90.00%

New User Allocation:
User 21 → eMBB Slice
CQI: 12, Bandwidth: 10.0 MHz, Rate: 45.00 Mbps, Latency: 15.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |         25    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |         10 |         50    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |          2 |          1.2  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |         10 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |    12 |          5 |         15    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |          5 |         15    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |          2 |          7.2  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |         10 |         45    |             15 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |          0.8  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |          1 |          3    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |          1 |          0.33 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "I need to check the status of my smart home sensors"
CQI: 11
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~355, completion~6671, total~7026

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 2.902

Intent Analysis: Check status of smart home sensors
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 2.902 Mbps
Latency: 150.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-04-06 00:27:41
Total Users: 22
Average Resource Utilization: 90.0%
eMBB Total Rate: 670.82 Mbps, URLLC Total Rate: 86.20 Mbps, mMTC Total Rate: 8.81 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB          10  85.0/90 MHz       94.44%
URLLC          4  22.0/30 MHz       73.33%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 11, Bandwidth: 1.0 MHz, Rate: 2.90 Mbps, Latency: 150.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |         25    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |         10 |         50    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |          2 |          1.2  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |        120    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |         10 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |    12 |          5 |         15    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |          5 |         15    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |        200    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |          2 |          7.2  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |         10 |         45    |             15 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |         48.12 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |          0.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |          0.8  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |          1 |          3    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |          1 |          0.33 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |          1 |          2.9  |            150 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |          0.88 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 1
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~1227, total~1581

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 4.0, rate: 4000000.0

Intent Analysis: N/A
Recommended Slice: URLLC - Selected 4 MHz to compensate for low CQI=1, ensuring reliable transmission within URLLC constraints
Bandwidth Allocation: 4.0 MHz
Data Rate: 4000000.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-04-06 00:28:01
Total Users: 23
Average Resource Utilization: 93.08%
eMBB Total Rate: 670.82 Mbps, URLLC Total Rate: 4000086.20 Mbps, mMTC Total Rate: 8.81 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB          10  85.0/90 MHz       94.44%
URLLC          5  26.0/30 MHz       86.67%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 23 → URLLC Slice
CQI: 1, Bandwidth: 4.0 MHz, Rate: 4000000.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |        25     |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |         10 |        50     |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |          2 |         1.2   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |          4 |         4e+06 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |        10     |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |       120     |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |         10 |       100     |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |    12 |          5 |        15     |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |          5 |        15     |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |       200     |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |          2 |         7.2   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |         10 |        45     |             15 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |         0.5   |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |        48.12  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |       120     |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |         0.3   |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |         0.8   |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |          1 |         3     |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |          1 |         0.33  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |          1 |         2.9   |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |         0.88  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |         0.1   |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |         0.5   |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I want to use remote surgery equipment"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~2057, total~2408

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 6.996

Intent Analysis: User request is for remote surgery equipment, which demands ultra‑reliable, low‑latency communication with sufficient bandwidth for control and high‑definition video/telemetry. This matches the characteristics of the URLLC slice.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 3.0 MHz
Data Rate: 6.996 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-04-06 00:28:33
Total Users: 24
Average Resource Utilization: 95.38%
eMBB Total Rate: 670.82 Mbps, URLLC Total Rate: 4000093.20 Mbps, mMTC Total Rate: 8.81 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB          10  85.0/90 MHz       94.44%
URLLC          6  29.0/30 MHz       96.67%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 5, Bandwidth: 3.0 MHz, Rate: 7.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |        25     |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |         10 |        50     |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |          2 |         1.2   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |          4 |         4e+06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |          3 |         7     |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |        10     |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |       120     |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |         10 |       100     |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |    12 |          5 |        15     |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |          5 |        15     |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |       200     |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |          2 |         7.2   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |         10 |        45     |             15 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |         0.5   |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |        48.12  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |       120     |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |         0.3   |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |         0.8   |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |          1 |         3     |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |          1 |         0.33  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |          1 |         2.9   |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |         0.88  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |         0.1   |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |         0.5   |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I want to stream music while browsing social media"
CQI: 2
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~3010, total~3363

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 3.3

Intent Analysis: The user is requesting activities that require moderate to high data rates and low latency, specifically streaming audio content and loading interactive web pages. This classifies the traffic as high-throughput data, aligning with the eMBB slice profile.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 3.3 Mbps
Latency: 50.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-04-06 00:29:25
Total Users: 25
Average Resource Utilization: 99.23%
eMBB Total Rate: 674.12 Mbps, URLLC Total Rate: 4000093.20 Mbps, mMTC Total Rate: 8.81 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB          11  90.0/90 MHz       100.00%
URLLC          6  29.0/30 MHz       96.67%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 25 → eMBB Slice
CQI: 2, Bandwidth: 5.0 MHz, Rate: 3.30 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |        25     |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |         10 |        50     |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |          2 |         1.2   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |          4 |         4e+06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |          3 |         7     |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |        10     |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |       120     |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |         10 |       100     |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |    12 |          5 |        15     |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |          5 |        15     |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |       200     |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |          2 |         7.2   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |         10 |        45     |             15 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     2 |          5 |         3.3   |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |         0.5   |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |        48.12  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |       120     |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |         0.3   |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |         0.8   |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |          1 |         3     |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |          1 |         0.33  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |          1 |         2.9   |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |         0.88  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |         0.1   |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |         0.5   |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~357, completion~4915, total~5272

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.76

Intent Analysis: Real‑time control of distributed electrical microgrids
Recommended Slice: URLLC - The request demands sub‑10 ms latency and high reliability, which fits URLLC’s 1‑10 ms latency budget. The moderate bandwidth (≈1 MHz) is also within URLLC’s 1‑5 MHz allocation window.
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.76 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-04-06 00:30:40
Total Users: 26
Average Resource Utilization: 100.0%
eMBB Total Rate: 674.12 Mbps, URLLC Total Rate: 4000094.96 Mbps, mMTC Total Rate: 8.81 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB          11  90.0/90 MHz       100.00%
URLLC          7  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 1.76 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |          5 |        25     |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |         10 |        50     |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |          2 |         1.2   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |          4 |         4e+06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |          3 |         7     |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |          1 |         1.76  |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          5 |        10     |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |         10 |       120     |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |         10 |       100     |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | eMBB    |    12 |          5 |        15     |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |          5 |        15     |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |         10 |       200     |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |          2 |         7.2   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |         10 |        45     |             15 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     2 |          5 |         3.3   |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    13 |          1 |         0.5   |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         20 |        48.12  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |         12 |       120     |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |         0.3   |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |          2 |         0.8   |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |          1 |         3     |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |          1 |         0.33  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |          1 |         2.9   |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |         0.88  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |          1 |         0.1   |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |          1 |         0.5   |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need to transmit real-time patient vital signs during critical care"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~356, completion~2441, total~2797

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 1.2

Intent Analysis: N/A
Recommended Slice: URLLC - 3 MHz compensates for the low CQI (≈0.38 bits/s/Hz) while staying within URLLC limits. The resulting ~1.2 Mbps meets the vital‑signs payload requirement, and the latency is well below the 10 ms URLLC ceiling.
Bandwidth Allocation: 3.0 MHz
Data Rate: 1.2 Mbps
Latency: 3.0 ms

CAPACITY CHECK FAILED:
- Required: 3.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 27
----------------------------------------
Request: I need to transmit real-time patient vital signs during critical care
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 3.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~3567, total~3919

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.0

Intent Analysis: N/A
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.0 Mbps
Latency: 1.0 ms

CAPACITY CHECK FAILED:
- Required: 1.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 28
----------------------------------------
Request: I need reliable communication for firefighters inside buildings
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 1.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need to participate in an online multiplayer game"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~3709, total~4062

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 57.0

Intent Analysis: The user wants to join an online multiplayer game. This activity requires moderate to high data rates (typically 5‑20 Mbps) and low latency (ideally <50 ms). These requirements align best with the eMBB slice, which offers higher bandwidth and latency in the 10‑100 ms range.
Recommended Slice: eMBB - Free 10 MHz for User 29 while keeping total eMBB usage at the 90 MHz limit
Bandwidth Allocation: 10.0 MHz
Data Rate: 57.0 Mbps
Latency: 20.0 ms

CAPACITY CHECK FAILED:
- Required: 10.0 MHz, Available: 0.0 MHz in eMBB slice

----------------------------------------
ALLOCATION FAILED FOR USER 29
----------------------------------------
Request: I need to participate in an online multiplayer game
Slice type: eMBB
Reason: Insufficient capacity in eMBB slice. Required: 10.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I want to browse websites and check email"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~3918, total~4270

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 6.0, rate: 10.2

Intent Analysis: The user requires moderate‑speed internet access for browsing and e‑mail. This is a typical eMBB (enhanced Mobile Broadband) workload, not latency‑critical and not massive‑machine type.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 6.0 MHz
Data Rate: 10.2 Mbps
Latency: 50.0 ms

CAPACITY CHECK FAILED:
- Required: 6.0 MHz, Available: 0.0 MHz in eMBB slice

----------------------------------------
ALLOCATION FAILED FOR USER 30
----------------------------------------
Request: I want to browse websites and check email
Slice type: eMBB
Reason: Insufficient capacity in eMBB slice. Required: 6.0 MHz, Available: 0.0 MHz

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | mMTC    | mMTC           | Yes            |     3 |          2 |         0.3   |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | eMBB    | eMBB           | Yes            |    11 |         10 |       200     |             30 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | eMBB    | mMTC           | No             |    13 |          1 |         0.5   |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | mMTC    | mMTC           | Yes            |     4 |          1 |         0.877 |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | eMBB    | eMBB           | Yes            |     8 |         20 |        48.12  |            100 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | eMBB    | eMBB           | Yes            |     5 |         12 |       120     |             20 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | mMTC    | mMTC           | Yes            |     6 |          1 |         0.1   |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | mMTC    | mMTC           | Yes            |    15 |          1 |         0.5   |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | URLLC   | URLLC          | Yes            |     7 |          5 |        10     |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | eMBB    | eMBB           | Yes            |     5 |         10 |       120     |             30 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | mMTC    | mMTC           | Yes            |     6 |          2 |         0.8   |            250 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     3 |          5 |        25     |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | eMBB    | eMBB           | Yes            |     9 |         10 |       100     |             10 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | eMBB    | URLLC          | No             |    12 |          5 |        15     |              2 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | URLLC          | No             |     7 |          5 |        15     |             10 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |    10 |         10 |        50     |             10 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | mMTC    | mMTC           | Yes            |    13 |          1 |         3     |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     3 |          2 |         1.2   |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | mMTC    | mMTC           | Yes            |     3 |          1 |         0.33  |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    10 |          2 |         7.2   |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | eMBB    | eMBB           | Yes            |    12 |         10 |        45     |             15 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | mMTC    | mMTC           | Yes            |    11 |          1 |         2.902 |            150 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | URLLC   | URLLC          | Yes            |     1 |          4 |         4e+06 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     5 |          3 |         6.996 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | eMBB    | eMBB           | Yes            |     2 |          5 |         3.3   |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | URLLC          | Yes            |     6 |          1 |         1.76  |             10 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Failed   | URLLC   | URLLC          |                |     3 |          3 |         1.2   |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Failed   | URLLC   | URLLC          |                |     4 |          1 |         1     |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Failed   | eMBB    | URLLC          |                |    15 |         10 |        57     |             20 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Failed   | eMBB    | eMBB           |                |     8 |          6 |        10.2   |             50 | No         |
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
Average eMBB utilization: 63.46%
Average URLLC utilization: 40.90%
Average mMTC utilization: 65.00%

Results exported to F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_gym_minimax-m2.5.csv

[OK] TJU_gym 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_gym_minimax-m2.5.csv