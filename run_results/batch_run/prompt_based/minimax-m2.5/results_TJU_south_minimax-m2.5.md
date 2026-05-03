============================================================
场景 2/5: TJU_south
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_south.csv
输出: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_south_minimax-m2.5.csv
------------------------------------------------------------
Starting prompt-based network slice management system...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 2
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~347, completion~1368, total~1715

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 2000000.0

Intent Analysis: N/A
Recommended Slice: URLLC - Safety-critical machine shutdown requires ultra-reliable, low-latency communication (URLLC). This is a mission-critical control function demanding sub-10ms latency and high reliability, not bandwidth-intensive eMBB or massive connectivity mMTC.
Bandwidth Allocation: 2.0 MHz
Data Rate: 2000000.0 Mbps
Latency: 3.5 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-04-05 23:15:42
Total Users: 1
Average Resource Utilization: 1.54%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 2000000.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          1  2.0/30 MHz        6.67%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → URLLC Slice
CQI: 2, Bandwidth: 2.0 MHz, Rate: 2000000.00 Mbps, Latency: 3.5 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need my autonomous vehicle to communicate in real time"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~350, completion~1175, total~1525
Error parsing LLM response: No JSON found in response

----------------------------------------
ALLOCATION FAILED FOR USER 2
----------------------------------------
Request: I need my autonomous vehicle to communicate in real time
Slice type: Failed
Reason: Unknown error

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~349, completion~946, total~1295

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: N/A
Recommended Slice: mMTC - Smart agriculture sensors are IoT devices transmitting small amounts of data with low bandwidth requirements and moderate latency tolerance. mMTC slice is optimized for massive machine-type communications with appropriate bandwidth (1-3 MHz) and data rate (0.1-1 Mbps) ranges.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 150.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-04-05 23:16:17
Total Users: 2
Average Resource Utilization: 2.31%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 2000000.00 Mbps, mMTC Total Rate: 0.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          1  2.0/30 MHz        6.67%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 3 → mMTC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 150.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to participate in an online multiplayer game"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~1931, total~2282

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 28.0

Intent Analysis: online multiplayer gaming
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 28.0 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-04-05 23:16:50
Total Users: 3
Average Resource Utilization: 6.15%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 2000028.00 Mbps, mMTC Total Rate: 0.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          2  7.0/30 MHz        23.33%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 4 → URLLC Slice
CQI: 15, Bandwidth: 5.0 MHz, Rate: 28.00 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~1345, total~1699

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: The user requests network resources for a fleet of delivery drones sending low-rate telemetry data. This is characterized by small data payloads, periodic transmission, and potentially a large number of devices (fleet). The requirement is for continuous but minimal data exchange rather than high-bandwidth streaming or ultra-low latency control.
Recommended Slice: mMTC - mMTC (massive Machine-Type Communications) is specifically designed for IoT and sensor applications with low data rates. Telemetry data from drones fits this profile perfectly - it requires modest bandwidth, tolerates higher latency (100-1000ms range is acceptable for status updates), and benefits from mMTC's efficient handling of numerous connected devices. The low-rate nature (not video or large files) makes eMBB unnecessary, while URLLC's ultra-low latency (1-10ms) is excessive and wasteful for routine telemetry.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-04-05 23:17:13
Total Users: 4
Average Resource Utilization: 6.92%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 2000028.00 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          2  7.0/30 MHz        23.33%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 5 → mMTC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I want to play competitive mobile games with ultra-low latency"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~1077, total~1430

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 4.5

Intent Analysis: Competitive mobile gaming with ultra-low latency
Recommended Slice: URLLC - Ultra-low latency requirement (1-10ms) is critical for competitive gaming. URLLC slice provides the necessary latency guarantees.
Bandwidth Allocation: 3.0 MHz
Data Rate: 4.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-04-05 23:17:30
Total Users: 5
Average Resource Utilization: 9.23%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 2000032.50 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          3  10.0/30 MHz       33.33%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 6 → URLLC Slice
CQI: 4, Bandwidth: 3.0 MHz, Rate: 4.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to participate in an online multiplayer game"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~3944, total~4295

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 22.5

Intent Analysis: Online multiplayer gaming requires low latency (<10 ms) and a moderate, stable data rate (≈10‑30 Mbps). The user’s CQI = 14 indicates a high‑quality radio link (≈4.5 bits/Hz spectral efficiency).
Recommended Slice: eMBB - With CQI = 14, the spectral efficiency ≈4.5 bits/Hz. Allocating the maximum 5 MHz bandwidth for URLLC yields ≈22.5 Mbps, comfortably within the URLLC rate envelope (1‑100 Mbps) and meeting the <10 ms latency requirement.
Bandwidth Allocation: 5.0 MHz
Data Rate: 22.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-04-05 23:18:36
Total Users: 6
Average Resource Utilization: 13.08%
eMBB Total Rate: 22.50 Mbps, URLLC Total Rate: 2000032.50 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  5.0/90 MHz        5.56%
URLLC          3  10.0/30 MHz       33.33%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 7 → eMBB Slice
CQI: 14, Bandwidth: 5.0 MHz, Rate: 22.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I want to browse websites and check email"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~1958, total~2310

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: Typical HTTP web traffic with moderate bandwidth needs and latency tolerance.
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-04-05 23:19:07
Total Users: 7
Average Resource Utilization: 13.85%
eMBB Total Rate: 22.50 Mbps, URLLC Total Rate: 2000032.50 Mbps, mMTC Total Rate: 1.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  5.0/90 MHz        5.56%
URLLC          3  10.0/30 MHz       33.33%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 8 → mMTC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~4913, total~5266

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 7.5

Intent Analysis: N/A
Recommended Slice: URLLC - User intent is remote monitoring of home security cameras, which requires moderate bandwidth and relatively low latency. The URLLC slice supports the needed data rate (1‑100 Mbps) and latency (1‑10 ms). With CQI 6 the assumed spectral efficiency is ~1.5 bits/Hz, yielding ≈7.5 Mbps on a 5 MHz allocation, well within URLLC's allowed range and balancing workload across slices.
Bandwidth Allocation: 5.0 MHz
Data Rate: 7.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-04-05 23:20:30
Total Users: 8
Average Resource Utilization: 17.69%
eMBB Total Rate: 22.50 Mbps, URLLC Total Rate: 2000040.00 Mbps, mMTC Total Rate: 1.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  5.0/90 MHz        5.56%
URLLC          4  15.0/30 MHz       50.00%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 9 → URLLC Slice
CQI: 6, Bandwidth: 5.0 MHz, Rate: 7.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to make a high-quality voice call"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~5866, total~6219

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 6.0, rate: 128.0

Intent Analysis: high-quality voice call
Recommended Slice: eMBB - Voice traffic fits the eMBB slice which supports latency in the range 10‑100 ms. With CQI 4 the spectral efficiency is about 0.60 bits/s/Hz, yielding ~3.6 Mbps on a 6 MHz allocation – far exceeding voice requirements while keeping utilization low.
Bandwidth Allocation: 6.0 MHz
Data Rate: 128.0 Mbps
Latency: 100.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-04-05 23:22:07
Total Users: 9
Average Resource Utilization: 22.31%
eMBB Total Rate: 150.50 Mbps, URLLC Total Rate: 2000040.00 Mbps, mMTC Total Rate: 1.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  11.0/90 MHz       12.22%
URLLC          4  15.0/30 MHz       50.00%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 10 → eMBB Slice
CQI: 4, Bandwidth: 6.0 MHz, Rate: 128.00 Mbps, Latency: 100.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to send text messages and use messaging apps"
CQI: 3
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~2108, total~2462

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 1.0

Intent Analysis: Send text messages and use messaging apps – low‑volume, intermittent traffic with modest latency tolerance.
Recommended Slice: mMTC - mMTC is designed for massive machine‑type traffic with low to moderate data rates (0.1‑1 Mbps) and can tolerate latency up to 1000 ms.; The requested traffic (text & messaging) fits well within mMTC capabilities, avoiding over‑allocation of high‑capacity eMBB resources.; Allocating to URLLC would provide excess latency (1‑10 ms) and higher bandwidth (1‑5 MHz), which is unnecessary and could lead to inefficient resource use.; mMTC slice currently has sufficient idle bandwidth (7 MHz) to accommodate the new user without impacting existing services.
Bandwidth Allocation: 2.0 MHz
Data Rate: 1.0 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-04-05 23:22:45
Total Users: 10
Average Resource Utilization: 23.85%
eMBB Total Rate: 150.50 Mbps, URLLC Total Rate: 2000040.00 Mbps, mMTC Total Rate: 2.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  11.0/90 MHz       12.22%
URLLC          4  15.0/30 MHz       50.00%
mMTC           4  5.0/10 MHz        50.00%

New User Allocation:
User 11 → mMTC Slice
CQI: 3, Bandwidth: 2.0 MHz, Rate: 1.00 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     3 |          2 |         1     |          200   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I want to play competitive mobile games with ultra-low latency"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~355, completion~2197, total~2552

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 3.34

Intent Analysis: The user requests ultra‑low latency for competitive mobile gaming, which aligns with the URLLC slice characteristics (latency 1‑10 ms).
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 3.34 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-04-05 23:23:20
Total Users: 11
Average Resource Utilization: 27.69%
eMBB Total Rate: 150.50 Mbps, URLLC Total Rate: 2000043.34 Mbps, mMTC Total Rate: 2.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  11.0/90 MHz       12.22%
URLLC          5  20.0/30 MHz       66.67%
mMTC           4  5.0/10 MHz        50.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 3, Bandwidth: 5.0 MHz, Rate: 3.34 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          5 |         3.34  |            5   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     3 |          2 |         1     |          200   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need real-time traffic updates for navigation"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~1364, total~1716

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 64.0

Intent Analysis: Real-time traffic updates for navigation - requires low latency and reliable connectivity for timely traffic information delivery
Recommended Slice: URLLC - Real-time traffic updates require low latency (1-10ms) which is best served by URLLC slice. Navigation updates are time-critical and benefit from ultra-reliable low-latency communication. The moderate bandwidth requirement (1-100 Mbps) aligns with URLLC capabilities.
Bandwidth Allocation: 2.0 MHz
Data Rate: 64.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-04-05 23:23:44
Total Users: 12
Average Resource Utilization: 29.23%
eMBB Total Rate: 150.50 Mbps, URLLC Total Rate: 2000107.34 Mbps, mMTC Total Rate: 2.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  11.0/90 MHz       12.22%
URLLC          6  22.0/30 MHz       73.33%
mMTC           4  5.0/10 MHz        50.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 64.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          5 |         3.34  |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |        64     |            5   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     3 |          2 |         1     |          200   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2082, total~2435

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: N/A
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-04-05 23:24:14
Total Users: 13
Average Resource Utilization: 30.0%
eMBB Total Rate: 150.50 Mbps, URLLC Total Rate: 2000107.34 Mbps, mMTC Total Rate: 3.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  11.0/90 MHz       12.22%
URLLC          6  22.0/30 MHz       73.33%
mMTC           5  6.0/10 MHz        60.00%

New User Allocation:
User 14 → mMTC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          5 |         3.34  |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |        64     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     3 |          2 |         1     |          200   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |         0.5   |          500   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 13
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~355, completion~1419, total~1774

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 22.5

Intent Analysis: Cloud-based AI image processing requires high bandwidth and substantial data rates to handle large image files, upload to cloud services, and receive processed results. This is a typical eMBB use case.
Recommended Slice: eMBB - Image processing workloads demand high bandwidth (6-20 MHz range) and high data rates (100-400 Mbps). The eMBB slice is designed for such enhanced mobile broadband services. The current eMBB utilization is only 12.22%, leaving ample capacity for allocation.
Bandwidth Allocation: 10.0 MHz
Data Rate: 22.5 Mbps
Latency: 1.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-04-05 23:24:35
Total Users: 14
Average Resource Utilization: 37.69%
eMBB Total Rate: 173.00 Mbps, URLLC Total Rate: 2000107.34 Mbps, mMTC Total Rate: 3.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  21.0/90 MHz       23.33%
URLLC          6  22.0/30 MHz       73.33%
mMTC           5  6.0/10 MHz        60.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 13, Bandwidth: 10.0 MHz, Rate: 22.50 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          5 |         3.34  |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |        64     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         10 |        22.5   |            1   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     3 |          2 |         1     |          200   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need instant facial recognition for public security threats"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2780, total~3133

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 100.0

Intent Analysis: The user requires immediate facial recognition for public‑security threats, which demands ultra‑reliable low‑latency communication (URLLC) with minimal delay to enable real‑time decision making.
Recommended Slice: URLLC - Bandwidth set to the maximum allowed for URLLC (5 MHz) to maximize throughput while staying within the 1‑5 MHz range. Data rate set to the slice maximum (100 Mbps) to support high‑resolution video analytics, and latency of 5 ms satisfies the <10 ms requirement.
Bandwidth Allocation: 5.0 MHz
Data Rate: 100.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-04-05 23:25:17
Total Users: 15
Average Resource Utilization: 41.54%
eMBB Total Rate: 173.00 Mbps, URLLC Total Rate: 2000207.34 Mbps, mMTC Total Rate: 3.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  21.0/90 MHz       23.33%
URLLC          7  27.0/30 MHz       90.00%
mMTC           5  6.0/10 MHz        60.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 9, Bandwidth: 5.0 MHz, Rate: 100.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          5 |         3.34  |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |        64     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          5 |       100     |            5   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         10 |        22.5   |            1   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     3 |          2 |         1     |          200   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to monitor and control critical manufacturing processes in real-time"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~356, completion~2061, total~2417

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 3000000.0

Intent Analysis: Real-time monitoring and control of critical manufacturing processes
Recommended Slice: URLLC - Real-time monitoring and control requires ultra-low latency (1-10ms); Critical manufacturing processes demand high reliability; CQI 15 supports URLLC's reliability requirements; Moderate data rates (1-100 Mbps) sufficient for sensor data and control signals
Bandwidth Allocation: 3.0 MHz
Data Rate: 3000000.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-04-05 23:25:49
Total Users: 16
Average Resource Utilization: 43.85%
eMBB Total Rate: 173.00 Mbps, URLLC Total Rate: 5000207.34 Mbps, mMTC Total Rate: 3.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  21.0/90 MHz       23.33%
URLLC          8  30.0/30 MHz       100.00%
mMTC           5  6.0/10 MHz        60.00%

New User Allocation:
User 17 → URLLC Slice
CQI: 15, Bandwidth: 3.0 MHz, Rate: 3000000.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          5 |         3.34  |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |        64     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          5 |       100     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |          3 |         3e+06 |            5   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         10 |        22.5   |            1   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     3 |          2 |         1     |          200   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need to check the status of my smart home sensors"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~355, completion~1395, total~1750

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 0.5

Intent Analysis: IoT_status_query
Recommended Slice: mMTC - Smart home sensors are IoT devices classified under mMTC; Low bandwidth requirement aligns with mMTC constraints (1-3 MHz); URLLC slice at 100% utilization - no capacity available; mMTC slice at 60% utilization - sufficient capacity (4 MHz available); CQI of 3 indicates challenging RF conditions - mMTC's robust coding suitable
Bandwidth Allocation: 2.0 MHz
Data Rate: 0.5 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-04-05 23:26:15
Total Users: 17
Average Resource Utilization: 45.38%
eMBB Total Rate: 173.00 Mbps, URLLC Total Rate: 5000207.34 Mbps, mMTC Total Rate: 3.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  21.0/90 MHz       23.33%
URLLC          8  30.0/30 MHz       100.00%
mMTC           6  8.0/10 MHz        80.00%

New User Allocation:
User 18 → mMTC Slice
CQI: 3, Bandwidth: 2.0 MHz, Rate: 0.50 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          5 |         3.34  |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |        64     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          5 |       100     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |          3 |         3e+06 |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         10 |        22.5   |            1   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     3 |          2 |         1     |          200   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |          2 |         0.5   |          500   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 2
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~349, completion~1040, total~1389

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 0.5

Intent Analysis: N/A
Recommended Slice: mMTC - Industrial sensor data is classic mMTC use case; Requires machine-type communication capabilities; Low CQI (2) acceptable for mMTC with robust coding; URLLC slice is 100% utilized - no capacity available; eMBB would be excessive allocation for sensor data
Bandwidth Allocation: 2.0 MHz
Data Rate: 0.5 Mbps
Latency: 100.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-04-05 23:26:34
Total Users: 18
Average Resource Utilization: 46.92%
eMBB Total Rate: 173.00 Mbps, URLLC Total Rate: 5000207.34 Mbps, mMTC Total Rate: 4.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  21.0/90 MHz       23.33%
URLLC          8  30.0/30 MHz       100.00%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 19 → mMTC Slice
CQI: 2, Bandwidth: 2.0 MHz, Rate: 0.50 Mbps, Latency: 100.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          5 |         3.34  |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |        64     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          5 |       100     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |          3 |         3e+06 |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         10 |        22.5   |            1   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     3 |          2 |         1     |          200   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |          2 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |          2 |         0.5   |          100   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I want to download a big game file"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~2050, total~2402

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 200.0

Intent Analysis: Download a large game file, requiring high throughput and moderate latency.
Recommended Slice: eMBB - eMBB is designed for high‑bandwidth, high‑rate services (100‑400 Mbps) which match the need for a big file download. The URLLC and mMTC slices are already at 100 % utilization and cannot accommodate additional traffic without violating their strict latency/throughput constraints.
Bandwidth Allocation: 20.0 MHz
Data Rate: 200.0 Mbps
Latency: 25.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-04-05 23:27:07
Total Users: 19
Average Resource Utilization: 62.31%
eMBB Total Rate: 373.00 Mbps, URLLC Total Rate: 5000207.34 Mbps, mMTC Total Rate: 4.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  41.0/90 MHz       45.56%
URLLC          8  30.0/30 MHz       100.00%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 11, Bandwidth: 20.0 MHz, Rate: 200.00 Mbps, Latency: 25.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          5 |         3.34  |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |        64     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          5 |       100     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |          3 |         3e+06 |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         10 |        22.5   |            1   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |       200     |           25   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     3 |          2 |         1     |          200   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |          2 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |          2 |         0.5   |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to update my social media status"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~324, total~676
Error parsing LLM response: No JSON found in response

----------------------------------------
ALLOCATION FAILED FOR USER 21
----------------------------------------
Request: I want to update my social media status
Slice type: Failed
Reason: Unknown error

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 2
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~1583, total~1936

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 15.0

Intent Analysis: Critical infrastructure control requiring ultra-reeliable, low-latency connectivity with zero downtime tolerance
Recommended Slice: URLLC - User explicitly requires 'zero downtime' - characteristic of URLLC; Critical infrastructure control demands ultra-reliable connectivity; Low latency (<10ms) is mandatory for control applications; URLLC provides necessary reliability mechanisms (HARQ, robust retransmission)
Bandwidth Allocation: 2.0 MHz
Data Rate: 15.0 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 2.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 22
----------------------------------------
Request: I need to control critical infrastructure with zero downtime
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 2.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I want to stream a webinar with interactive features"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~1293, total~1646

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 100.0

Intent Analysis: N/A
Recommended Slice: eMBB - The requested service (high‑definition webinar streaming with interactive elements) requires high bandwidth and low‑to‑moderate latency, fitting the eMBB slice profile. The user's CQI of 12 supports a high‑order modulation (≈64‑QAM) and a spectral efficiency of ~5 bits/Hz, enabling a 20 MHz allocation to meet the minimum eMBB data rate of 100 Mbps while staying within the allowed bandwidth range. The eMBB slice currently has ample free resources (≈49 MHz) and the added allocation will raise its utilization to ~68 %, well within safe operating limits.
Bandwidth Allocation: 20.0 MHz
Data Rate: 100.0 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-04-05 23:28:10
Total Users: 20
Average Resource Utilization: 77.69%
eMBB Total Rate: 473.00 Mbps, URLLC Total Rate: 5000207.34 Mbps, mMTC Total Rate: 4.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  61.0/90 MHz       67.78%
URLLC          8  30.0/30 MHz       100.00%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 23 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 100.00 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          5 |         3.34  |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |        64     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          5 |       100     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |          3 |         3e+06 |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         10 |        22.5   |            1   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |       200     |           25   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | eMBB    |    12 |         20 |       100     |           20   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     3 |          2 |         1     |          200   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |          2 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |          2 |         0.5   |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~355, completion~3147, total~3502

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 200.0

Intent Analysis: Cloud‑based AI services for image processing
Recommended Slice: eMBB - Image‑processing workloads require sustained high data rates (eMBB). The eMBB slice can provide 6‑20 MHz per user; 10 MHz is chosen to stay within the free pool while achieving a data rate of ~200 Mbps (well inside the 100‑400 Mbps eMBB range). Latency of 30 ms satisfies the 10‑100 ms eMBB requirement.
Bandwidth Allocation: 10.0 MHz
Data Rate: 200.0 Mbps
Latency: 30.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-04-05 23:29:05
Total Users: 21
Average Resource Utilization: 85.38%
eMBB Total Rate: 673.00 Mbps, URLLC Total Rate: 5000207.34 Mbps, mMTC Total Rate: 4.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  71.0/90 MHz       78.89%
URLLC          8  30.0/30 MHz       100.00%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 24 → eMBB Slice
CQI: 4, Bandwidth: 10.0 MHz, Rate: 200.00 Mbps, Latency: 30.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          5 |         3.34  |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |        64     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          5 |       100     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |          3 |         3e+06 |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         10 |        22.5   |            1   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |       200     |           25   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | eMBB    |    12 |         20 |       100     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | eMBB    |     4 |         10 |       200     |           30   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     3 |          2 |         1     |          200   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |          2 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |          2 |         0.5   |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I need to synchronize distributed financial ledgers instantly"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~4875, total~5228

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 17.0, rate: 102.0

Intent Analysis: Low‑latency, high‑reliability data exchange for financial ledger updates
Recommended Slice: eMBB - eMBB slice minimum rate 100 Mbps; 17 MHz × 6 bps/Hz ≈ 102 Mbps satisfies this while keeping latency on the low end of eMBB (≈10 ms).
Bandwidth Allocation: 17.0 MHz
Data Rate: 102.0 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-04-05 23:30:27
Total Users: 22
Average Resource Utilization: 98.46%
eMBB Total Rate: 775.00 Mbps, URLLC Total Rate: 5000207.34 Mbps, mMTC Total Rate: 4.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  88.0/90 MHz       97.78%
URLLC          8  30.0/30 MHz       100.00%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 25 → eMBB Slice
CQI: 14, Bandwidth: 17.0 MHz, Rate: 102.00 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          5 |         3.34  |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |        64     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          5 |       100     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |          3 |         3e+06 |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         10 |        22.5   |            1   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |       200     |           25   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | eMBB    |    12 |         20 |       100     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | eMBB    |     4 |         10 |       200     |           30   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |    14 |         17 |       102     |           10   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     3 |          2 |         1     |          200   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |          2 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |          2 |         0.5   |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to control a robotic arm in real time"
CQI: 2
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~3999, total~4353

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 1.172

Intent Analysis: The request involves controlling a robotic arm in real time, which demands low latency and moderate reliability. These characteristics map best to the URLLC slice (latency 1‑10 ms, moderate bandwidth).
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 1.172 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 5.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 26
----------------------------------------
Request: I need to control a robotic arm in real time
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 5.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need emergency response coordination during a disaste"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~4708, total~5061

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 50.0

Intent Analysis: Emergency response coordination during a disaster requires low‑latency, highly reliable connectivity with sufficient bandwidth for control, voice and possibly video traffic.
Recommended Slice: URLLC - 5 MHz is the maximum allowed for URLLC and provides enough capacity for the emergency traffic while staying within the 1‑10 ms latency bound.
Bandwidth Allocation: 5.0 MHz
Data Rate: 50.0 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 5.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 27
----------------------------------------
Request: I need emergency response coordination during a disaste
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 5.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I want to use augmented reality navigation"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~1615, total~1966

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 10.0

Intent Analysis: N/A
Recommended Slice: eMBB - AR navigation requires high bandwidth and moderate latency. eMBB slice is designed for high-bandwidth applications. URLLC has lower latency but is 100% utilized. mMTC has excessive latency (100-1000ms) unsuitable for real-time AR.
Bandwidth Allocation: 2.0 MHz
Data Rate: 10.0 Mbps
Latency: 50.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-04-05 23:33:19
Total Users: 23
Average Resource Utilization: 100.0%
eMBB Total Rate: 785.00 Mbps, URLLC Total Rate: 5000207.34 Mbps, mMTC Total Rate: 4.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          8  30.0/30 MHz       100.00%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 28 → eMBB Slice
CQI: 6, Bandwidth: 2.0 MHz, Rate: 10.00 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          2 |         2e+06 |            3.5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          5 |         3.34  |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |        64     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          5 |       100     |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |          3 |         3e+06 |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |        28     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          3 |         4.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |          5 |         7.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |          6 |       128     |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         10 |        22.5   |            1   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |       200     |           25   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | eMBB    |    12 |         20 |       100     |           20   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | eMBB    |     4 |         10 |       200     |           30   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |    14 |         17 |       102     |           10   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | eMBB    |     6 |          2 |        10     |           50   | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | eMBB    |    14 |          5 |        22.5   |            5   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     3 |          2 |         1     |          200   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |          2 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |          2 |         0.5   |          100   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |         0.5   |          150   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |     4 |          1 |         0.5   |          500   |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~3603, total~3956

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 1.3

Intent Analysis: N/A
Recommended Slice: eMBB - Safety shutdown command requires ultra‑reliable low‑latency communication (URLLC). The request’s CQI of 3 indicates limited channel quality, so a robust 2 MHz allocation is chosen to provide adequate coverage while delivering ~1.3 Mbps, well above the minimum 1 Mbps required for a small control message. The latency of ~5 ms satisfies the 1–10 ms URLLC requirement. To meet the bandwidth demand, 2 MHz is temporarily borrowed from the eMBB slice, reducing its utilization from 100 % (90/90 MHz) to ~97.8 % (88/90 MHz) with minimal impact on eMBB users.
Bandwidth Allocation: 2.0 MHz
Data Rate: 1.3 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 2.0 MHz, Available: 0.0 MHz in eMBB slice

----------------------------------------
ALLOCATION FAILED FOR USER 29
----------------------------------------
Request: I need immediate machine shutdown capability for safety incidents
Slice type: eMBB
Reason: Insufficient capacity in eMBB slice. Required: 2.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 1
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~356, completion~2451, total~2807

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.3

Intent Analysis: Periodic transmission of a tiny status packet (few bytes) from an IoT sensor.
Recommended Slice: mMTC - The sensor generates extremely low data traffic, which aligns with the mMTC slice (0.1‑1 Mbps).; Latency tolerance (hundreds of ms) fits the mMTC latency window (100‑1000 ms).; mMTC bandwidth granularity (1‑3 MHz) is sufficient for a single sensor.
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.3 Mbps
Latency: 500.0 ms

CAPACITY CHECK FAILED:
- Required: 1.0 MHz, Available: 0.0 MHz in mMTC slice

----------------------------------------
ALLOCATION FAILED FOR USER 30
----------------------------------------
Request: My smart parking sensor needs to report if the spot is free
Slice type: mMTC
Reason: Insufficient capacity in mMTC slice. Required: 1.0 MHz, Available: 0.0 MHz

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI | BW (MHz)   | Rate (Mbps)   | Latency (ms)   | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | URLLC   | URLLC          | Yes            |     2 | 2.0        | 2000000.0     | 3.5            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Failed   | Failed  | URLLC          |                |     3 | N/A        | N/A           | N/A            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | mMTC    | mMTC           | Yes            |    15 | 1.0        | 0.5           | 150.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | URLLC   | URLLC          | Yes            |    15 | 5.0        | 28.0          | 20.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | mMTC    | mMTC           | Yes            |     3 | 1.0        | 0.5           | 500.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | URLLC   | URLLC          | Yes            |     4 | 3.0        | 4.5           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | eMBB    | URLLC          | No             |    14 | 5.0        | 22.5          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | mMTC    | eMBB           | No             |     4 | 1.0        | 0.5           | 500.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | URLLC   | eMBB           | No             |     6 | 5.0        | 7.5           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | eMBB    | eMBB           | Yes            |     4 | 6.0        | 128.0         | 100.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | mMTC    | eMBB           | No             |     3 | 2.0        | 1.0           | 200.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     3 | 5.0        | 3.34          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | eMBB           | No             |     9 | 2.0        | 64.0          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | mMTC    | mMTC           | Yes            |     6 | 1.0        | 0.5           | 500.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |    13 | 10.0       | 22.5          | 1.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |     9 | 5.0        | 100.0         | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | URLLC   | URLLC          | Yes            |    15 | 3.0        | 3000000.0     | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | mMTC    | mMTC           | Yes            |     3 | 2.0        | 0.5           | 500.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | mMTC    | mMTC           | Yes            |     2 | 2.0        | 0.5           | 100.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    11 | 20.0       | 200.0         | 25.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Failed   | Failed  | eMBB           |                |     6 | N/A        | N/A           | N/A            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Failed   | URLLC   | URLLC          |                |     2 | 2.0        | 15.0          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | eMBB    | eMBB           | Yes            |    12 | 20.0       | 100.0         | 20.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | eMBB    | eMBB           | Yes            |     4 | 10.0       | 200.0         | 30.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | eMBB    | URLLC          | No             |    14 | 17.0       | 102.0         | 10.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Failed   | URLLC   | URLLC          |                |     2 | 5.0        | 1.172         | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Failed   | URLLC   | URLLC          |                |    14 | 5.0        | 50.0          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | eMBB    | eMBB           | Yes            |     6 | 2.0        | 10.0          | 50.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Failed   | eMBB    | URLLC          |                |     3 | 2.0        | 1.3           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Failed   | mMTC    | mMTC           |                |     1 | 1.0        | 1.3           | 500.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 23/30 (76.7%)

Intent Understanding Evaluation:
Correctly identified intents: 17/23
Intent understanding rate: 73.9%

Workload Balancing Statistics:
Users with workload balancing: 23/30
Workload balancing rate: 76.7%

Slice Utilization Statistics:
Average eMBB utilization: 25.41%
Average URLLC utilization: 64.64%
Average mMTC utilization: 53.91%

Results exported to F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_south_minimax-m2.5.csv

[OK] TJU_south 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_south_minimax-m2.5.csv