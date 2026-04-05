============================================================
场景 1/5: TJU_north
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_north.csv
输出: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_north_minimax-m2.5.csv
------------------------------------------------------------
Starting prompt-based network slice management system...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I want to use augmented reality navigation"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 15.0, rate: 100.0

Intent Analysis: Augmented Reality Navigation
Recommended Slice: eMBB - AR navigation requires high bandwidth for video/texture streaming and moderate-to-low latency for real-time rendering. eMBB slice provides the necessary bandwidth (6-20 MHz) and data rates (100-400 Mbps) while meeting latency requirements (10-100ms) for AR applications.
Bandwidth Allocation: 15.0 MHz
Data Rate: 100.0 Mbps
Latency: 50.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-30 10:46:49
Total Users: 1
Average Resource Utilization: 11.54%
eMBB Total Rate: 100.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → eMBB Slice
CQI: 8, Bandwidth: 15.0 MHz, Rate: 100.00 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |     8 |         15 |           100 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 5.0

Intent Analysis: The user requires a safety‑critical control command to shut down a machine instantly. This is a low‑volume, latency‑sensitive transmission rather than a high‑throughput data flow.
Recommended Slice: URLLC - URLLC supports latency 1‑10 ms and data rates 1‑100 Mbps, matching the need for an immediate shutdown command. The request does not need the high bandwidth of eMBB nor the massive connectivity of mMTC.
Bandwidth Allocation: 2.0 MHz
Data Rate: 5.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-30 10:47:29
Total Users: 2
Average Resource Utilization: 13.08%
eMBB Total Rate: 100.00 Mbps, URLLC Total Rate: 5.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          1  2.0/30 MHz        6.67%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → URLLC Slice
CQI: 5, Bandwidth: 2.0 MHz, Rate: 5.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |             5 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |           100 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 11
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 12.5

Intent Analysis: Real‑time balancing of electrical load across microgrids. This is a control/automation task that needs very low latency and reliable, moderate‑rate communication.
Recommended Slice: URLLC - The request is latency‑critical (real‑time control) – URLLC supports 1‑10 ms latency, whereas eMBB starts at 10 ms.; Data volume for load‑balancing messages is moderate (typically a few megabits per second), well within the 1‑100 Mbps range of URLLC.; CQI = 11 indicates a decent channel, allowing the use of a higher‑order modulation (e.g., 16‑QAM) to achieve the required rate without excessive bandwidth.
Bandwidth Allocation: 5.0 MHz
Data Rate: 12.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-30 10:48:07
Total Users: 3
Average Resource Utilization: 16.92%
eMBB Total Rate: 100.00 Mbps, URLLC Total Rate: 17.50 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          2  7.0/30 MHz        23.33%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 3 → URLLC Slice
CQI: 11, Bandwidth: 5.0 MHz, Rate: 12.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          12.5 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         100   |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.0

Intent Analysis: User wants to check the status of city‑wide smart streetlights. This involves low‑volume, periodic IoT data typical of massive machine‑type communications (mMTC).
Recommended Slice: mMTC - 1 MHz bandwidth with BPSK yields exactly 1 Mbps, meeting the mMTC rate window (0.1‑1 Mbps). The latency of 200 ms is within the allowed 100‑1000 ms range.
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.0 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-30 10:49:12
Total Users: 4
Average Resource Utilization: 17.69%
eMBB Total Rate: 100.00 Mbps, URLLC Total Rate: 17.50 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          2  7.0/30 MHz        23.33%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 4 → mMTC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 1.00 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          12.5 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         100   |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           1   |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I want to use remote surgery equipment"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 14.0

Intent Analysis: N/A
Recommended Slice: URLLC - Remote surgery is a mission-critical URLLC use case requiring real-time haptic feedback and precise control, where any delay could endanger patient safety.
Bandwidth Allocation: 3.0 MHz
Data Rate: 14.0 Mbps
Latency: 1.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-30 10:49:34
Total Users: 5
Average Resource Utilization: 20.0%
eMBB Total Rate: 100.00 Mbps, URLLC Total Rate: 31.50 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          3  10.0/30 MHz       33.33%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 5 → URLLC Slice
CQI: 14, Bandwidth: 3.0 MHz, Rate: 14.00 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          12.5 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |          14   |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         100   |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           1   |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: IoT sensor reporting parking spot occupancy
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-30 10:50:09
Total Users: 6
Average Resource Utilization: 20.77%
eMBB Total Rate: 100.00 Mbps, URLLC Total Rate: 31.50 Mbps, mMTC Total Rate: 1.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          3  10.0/30 MHz       33.33%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 6 → mMTC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          12.5 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |          14   |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         100   |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           1   |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |           0.5 |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 2.0

Intent Analysis: N/A
Recommended Slice: URLLC - Safety shutdown requires 1-10ms latency (URLLC: 1-10ms vs eMBB: 10-100ms); Machine control commands are small data payloads; URLLC provides ultra-reliable low-latency communications for mission-critical applications; mMTC latency (100-1000ms) unsuitable for immediate safety shutdown
Bandwidth Allocation: 2.0 MHz
Data Rate: 2.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-30 10:50:39
Total Users: 7
Average Resource Utilization: 22.31%
eMBB Total Rate: 100.00 Mbps, URLLC Total Rate: 33.50 Mbps, mMTC Total Rate: 1.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          4  12.0/30 MHz       40.00%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 6, Bandwidth: 2.0 MHz, Rate: 2.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          12.5 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |          14   |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |           2   |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         100   |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           1   |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I need to transmit real-time patient vital signs during critical care"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 5.0

Intent Analysis: Real‑time transmission of patient vital signs during critical care
Recommended Slice: URLLC - The URLLC slice provides the required ultra‑low latency and high reliability for real‑time patient‑vital‑sign transmission. The allocated 2 MHz bandwidth and 5 Mbps data rate satisfy the user's data needs while staying within the URLLC slice's defined limits.
Bandwidth Allocation: 2.0 MHz
Data Rate: 5.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-30 10:51:19
Total Users: 8
Average Resource Utilization: 23.85%
eMBB Total Rate: 100.00 Mbps, URLLC Total Rate: 38.50 Mbps, mMTC Total Rate: 1.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          5  14.0/30 MHz       46.67%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 8, Bandwidth: 2.0 MHz, Rate: 5.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          12.5 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |          14   |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |           2   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |           5   |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         100   |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           1   |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: Low‑rate telemetry data from a fleet of delivery drones
Recommended Slice: mMTC - mMTC supports bandwidth 1‑3 MHz, data rates 0.1‑1 Mbps, and latency 100‑1000 ms, matching the low‑rate telemetry needs while preserving the low‑latency slices for more critical traffic.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-30 10:52:26
Total Users: 9
Average Resource Utilization: 24.62%
eMBB Total Rate: 100.00 Mbps, URLLC Total Rate: 38.50 Mbps, mMTC Total Rate: 2.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          5  14.0/30 MHz       46.67%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 9 → mMTC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          12.5 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |          14   |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |           2   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         100   |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           1   |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |           0.5 |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 0.5

Intent Analysis: The user requests connectivity for a network of environmental sensors to report air quality.
Recommended Slice: mMTC - Bandwidth of 2 MHz is allocated within the 1-3 MHz mMTC limit. A data rate of 0.5 Mbps is set within the 0.1-1 Mbps mMTC range, sufficient for periodic sensor reporting. Latency is set to 500ms, well within the 100-1000ms mMTC tolerance.
Bandwidth Allocation: 2.0 MHz
Data Rate: 0.5 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-30 10:53:20
Total Users: 10
Average Resource Utilization: 26.15%
eMBB Total Rate: 100.00 Mbps, URLLC Total Rate: 38.50 Mbps, mMTC Total Rate: 2.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          5  14.0/30 MHz       46.67%
mMTC           4  5.0/10 MHz        50.00%

New User Allocation:
User 10 → mMTC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 0.50 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          12.5 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |          14   |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |           2   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         100   |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |           0.5 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           1   |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 7.78

Intent Analysis: Power grid fault detection and isolation with 'instant' requirement
Recommended Slice: URLLC - The 'instant' requirement (sub-10ms latency) is the defining characteristic. URLLC slice is specifically designed for ultra-reliable low-latency communications, making it ideal for power grid protection and fault isolation applications.
Bandwidth Allocation: 2.0 MHz
Data Rate: 7.78 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-30 10:53:50
Total Users: 11
Average Resource Utilization: 27.69%
eMBB Total Rate: 100.00 Mbps, URLLC Total Rate: 46.28 Mbps, mMTC Total Rate: 2.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          6  16.0/30 MHz       53.33%
mMTC           4  5.0/10 MHz        50.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 15, Bandwidth: 2.0 MHz, Rate: 7.78 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need to check the status of my smart home sensors"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.0

Intent Analysis: N/A
Recommended Slice: mMTC - mMTC scheduling overhead and modest packet size result in ~180 ms end‑to‑end delay, well within the 100‑1000 ms window.
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.0 Mbps
Latency: 180.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-30 10:54:38
Total Users: 12
Average Resource Utilization: 28.46%
eMBB Total Rate: 100.00 Mbps, URLLC Total Rate: 46.28 Mbps, mMTC Total Rate: 3.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          6  16.0/30 MHz       53.33%
mMTC           5  6.0/10 MHz        60.00%

New User Allocation:
User 12 → mMTC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 1.00 Mbps, Latency: 180.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need to check weather forecasts"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 16.65

Intent Analysis: The user wants to retrieve a weather forecast, which involves a moderate‑size data payload (web page/text with images) and can tolerate a few milliseconds of latency. This is a typical information‑retrieval use case rather than a massive IoT sensor report or an ultra‑reliable control message, so a slice that offers reasonable bandwidth and latency is appropriate.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 16.65 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-30 10:55:31
Total Users: 13
Average Resource Utilization: 32.31%
eMBB Total Rate: 100.00 Mbps, URLLC Total Rate: 62.93 Mbps, mMTC Total Rate: 3.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          7  21.0/30 MHz       70.00%
mMTC           5  6.0/10 MHz        60.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 8, Bandwidth: 5.0 MHz, Rate: 16.65 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 2.0

Intent Analysis: Mission-Critical Communication
Recommended Slice: URLLC - Firefighter communications require ultra-reliable low-latency connectivity (URLLC) due to life-safety operations. The CQI of 9 indicates favorable channel conditions suitable for URLLC allocation. URLLC slice provides 1-10ms latency which is essential for real-time voice/data communication during emergency response. While eMBB offers higher bandwidth, it does not meet the latency requirements for critical firefighter operations.
Bandwidth Allocation: 3.0 MHz
Data Rate: 2.0 Mbps
Latency: 1.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-30 10:56:12
Total Users: 14
Average Resource Utilization: 34.62%
eMBB Total Rate: 100.00 Mbps, URLLC Total Rate: 64.93 Mbps, mMTC Total Rate: 3.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          8  24.0/30 MHz       80.00%
mMTC           5  6.0/10 MHz        60.00%

New User Allocation:
User 14 → URLLC Slice
CQI: 9, Bandwidth: 3.0 MHz, Rate: 2.00 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          3 |          2    |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use maps for basic navigation"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 30.0

Intent Analysis: Map navigation typically requires moderate data rates and latency below 100 ms. This is best served by the eMBB slice, which offers the necessary bandwidth and speed while staying within acceptable latency limits.
Recommended Slice: eMBB - Allocating the maximum eMBB bandwidth of 20 MHz ensures the slice's minimum data‑rate requirement of 100 Mbps is met, providing ample capacity for map tiles and navigation updates.
Bandwidth Allocation: 20.0 MHz
Data Rate: 30.0 Mbps
Latency: 50.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-30 10:56:56
Total Users: 15
Average Resource Utilization: 50.0%
eMBB Total Rate: 130.00 Mbps, URLLC Total Rate: 64.93 Mbps, mMTC Total Rate: 3.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  35.0/90 MHz       38.89%
URLLC          8  24.0/30 MHz       80.00%
mMTC           5  6.0/10 MHz        60.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 9, Bandwidth: 20.0 MHz, Rate: 30.00 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          3 |          2    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |         30    |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need to stream 8K video content"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 400.0

Intent Analysis: User requests streaming of 8K video content, which demands high bandwidth and relatively low latency. This use‑case aligns with the eMBB slice.
Recommended Slice: eMBB - 8K video typically requires 50‑100 Mbps for compressed streams; the allocated 400 Mbps provides headroom for variability and guarantees smooth playback.
Bandwidth Allocation: 20.0 MHz
Data Rate: 400.0 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-30 10:57:22
Total Users: 16
Average Resource Utilization: 65.38%
eMBB Total Rate: 530.00 Mbps, URLLC Total Rate: 64.93 Mbps, mMTC Total Rate: 3.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  55.0/90 MHz       61.11%
URLLC          8  24.0/30 MHz       80.00%
mMTC           5  6.0/10 MHz        60.00%

New User Allocation:
User 16 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 400.00 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          3 |          2    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |         30    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |         20 |        400    |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need instant alerts for life-threatening patient conditions"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 10.0

Intent Analysis: The user requires immediate, highly reliable notification of life‑threatening events. This is a critical, low‑latency service that must be delivered within a few milliseconds and with a very high probability of success – characteristics that align with the URLLC (Ultra‑Reliable Low‑Latency Communications) slice rather than eMBB or mMTC.
Recommended Slice: URLLC - URLLC provides the required latency range (1‑10 ms) and a data rate of 1‑100 Mbps, both of which satisfy the instant‑alert requirement.; CQI = 7 indicates a moderate channel quality, allowing a modest MCS (e.g., 16‑QAM ½). A 2 MHz allocation yields an achievable throughput of ~10 Mbps, which is sufficient for transmitting small alert payloads instantly.; Allocating 2 MHz keeps the URLLC slice utilization below the 90 % safety threshold while preserving headroom for other ultra‑reliable traffic.; The eMBB and mMTC slices remain largely under‑utilized and can absorb future growth without impact on this user’s service.
Bandwidth Allocation: 2.0 MHz
Data Rate: 10.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-30 10:58:09
Total Users: 17
Average Resource Utilization: 66.92%
eMBB Total Rate: 530.00 Mbps, URLLC Total Rate: 74.93 Mbps, mMTC Total Rate: 3.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  55.0/90 MHz       61.11%
URLLC          9  26.0/30 MHz       86.67%
mMTC           5  6.0/10 MHz        60.00%

New User Allocation:
User 17 → URLLC Slice
CQI: 7, Bandwidth: 2.0 MHz, Rate: 10.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          3 |          2    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         10    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |         30    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |         20 |        400    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "My smart meter needs to report its reading"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 3.3

Intent Analysis: The user wants to transmit a smart meter reading. This is a small, periodic, delay‑tolerant IoT data transfer that is best served by a massive Machine‑Type Communications (mMTC) slice.
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 3.3 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-30 10:58:46
Total Users: 18
Average Resource Utilization: 67.69%
eMBB Total Rate: 530.00 Mbps, URLLC Total Rate: 74.93 Mbps, mMTC Total Rate: 6.80 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  55.0/90 MHz       61.11%
URLLC          9  26.0/30 MHz       86.67%
mMTC           6  7.0/10 MHz        70.00%

New User Allocation:
User 18 → mMTC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 3.30 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          3 |          2    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |         30    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |         20 |        400    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |          1 |          3.3  |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to control a robotic arm in real time"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 20.0

Intent Analysis: N/A
Recommended Slice: URLLC - Robotic‑arm control demands ultra‑reliable low‑latency communications, which aligns with the URLLC slice profile (latency 1‑10 ms, bandwidth 1‑5 MHz, rate 1‑100 Mbps).
Bandwidth Allocation: 3.0 MHz
Data Rate: 20.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-30 10:59:32
Total Users: 19
Average Resource Utilization: 70.0%
eMBB Total Rate: 530.00 Mbps, URLLC Total Rate: 94.93 Mbps, mMTC Total Rate: 6.80 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  55.0/90 MHz       61.11%
URLLC         10  29.0/30 MHz       96.67%
mMTC           6  7.0/10 MHz        70.00%

New User Allocation:
User 19 → URLLC Slice
CQI: 7, Bandwidth: 3.0 MHz, Rate: 20.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          3 |          2    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |          3 |         20    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |         30    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |         20 |        400    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |          1 |          3.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I want to watch 4K video"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 130.0

Intent Analysis: N/A
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 20.0 MHz
Data Rate: 130.0 Mbps
Latency: 30.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-30 11:00:09
Total Users: 20
Average Resource Utilization: 85.38%
eMBB Total Rate: 660.00 Mbps, URLLC Total Rate: 94.93 Mbps, mMTC Total Rate: 6.80 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  75.0/90 MHz       83.33%
URLLC         10  29.0/30 MHz       96.67%
mMTC           6  7.0/10 MHz        70.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 11, Bandwidth: 20.0 MHz, Rate: 130.00 Mbps, Latency: 30.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          3 |          2    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |          3 |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |         30    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |         20 |        400    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |        130    |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |          1 |          3.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to stream music while browsing social media"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 6.0, rate: 100.0

Intent Analysis: User wants to stream music and browse social media – these are typical eMBB services requiring moderate to high data rates and acceptable latency (10‑100 ms).
Recommended Slice: eMBB - Using the maximum remaining eMBB bandwidth (15 MHz) with 256‑QAM (CQI 12 ≈ 3.5 bits/Hz) yields about 55 Mbps. This is the highest feasible rate given the current channel quality and slice limits.
Bandwidth Allocation: 6.0 MHz
Data Rate: 100.0 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-30 11:01:52
Total Users: 21
Average Resource Utilization: 90.0%
eMBB Total Rate: 760.00 Mbps, URLLC Total Rate: 94.93 Mbps, mMTC Total Rate: 6.80 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  81.0/90 MHz       90.00%
URLLC         10  29.0/30 MHz       96.67%
mMTC           6  7.0/10 MHz        70.00%

New User Allocation:
User 21 → eMBB Slice
CQI: 12, Bandwidth: 6.0 MHz, Rate: 100.00 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          3 |          2    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |          3 |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |         30    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |         20 |        400    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |        130    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |          6 |        100    |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |          1 |          3.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "My asset tracking device needs to send location update"
CQI: 12
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.8

Intent Analysis: N/A
Recommended Slice: eMBB - Asset tracking device sending a small location update fits the mMTC profile (low‑rate, moderate latency). The mMTC slice currently has 3 MHz free; allocating the minimum 1 MHz minimizes resource usage while satisfying the device’s needs and leaves headroom for future devices.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.8 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-30 11:02:49
Total Users: 22
Average Resource Utilization: 90.77%
eMBB Total Rate: 760.80 Mbps, URLLC Total Rate: 94.93 Mbps, mMTC Total Rate: 6.80 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  82.0/90 MHz       91.11%
URLLC         10  29.0/30 MHz       96.67%
mMTC           6  7.0/10 MHz        70.00%

New User Allocation:
User 22 → eMBB Slice
CQI: 12, Bandwidth: 1.0 MHz, Rate: 0.80 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          3 |          2    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |          3 |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |         30    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |         20 |        400    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |        130    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |          6 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | eMBB    |    12 |          1 |          0.8  |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |          1 |          3.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: Environmental sensor network reporting air quality data - IoT/periodic monitoring use case
Recommended Slice: mMTC - Environmental sensors generate small, periodic data transmissions; Low bandwidth requirements (0.1-1 Mbps sufficient for sensor data); Higher latency tolerance (100-1000ms acceptable for air quality monitoring); mMTC slice has 30% unused capacity (3.0 MHz available); CQI of 15 indicates good radio conditions enabling efficient mMTC transmission
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 250.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-30 11:03:19
Total Users: 23
Average Resource Utilization: 91.54%
eMBB Total Rate: 760.80 Mbps, URLLC Total Rate: 94.93 Mbps, mMTC Total Rate: 7.30 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  82.0/90 MHz       91.11%
URLLC         10  29.0/30 MHz       96.67%
mMTC           7  8.0/10 MHz        80.00%

New User Allocation:
User 23 → mMTC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 250.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          3 |          2    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |          3 |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |         30    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |         20 |        400    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |        130    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |          6 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | eMBB    |    12 |          1 |          0.8  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |          1 |          3.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |          1 |          0.5  |            250 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to deploy early warning systems for natural disasters"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 12.0

Intent Analysis: Critical Infrastructure - Emergency Services
Recommended Slice: URLLC - Early warning systems require ultra-reliable low-latency communication for real-time alerts and emergency notifications. URLLC slice provides the necessary latency (1-10ms) and priority handling for safety-critical applications. While eMBB offers higher bandwidth, its higher latency (10-100ms) is unsuitable for emergency alerts. mMTC's very high latency (100-1000ms) and low data rates are insufficient for real-time warning transmission.
Bandwidth Allocation: 2.0 MHz
Data Rate: 12.0 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 2.0 MHz, Available: 1.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 24
----------------------------------------
Request: I need to deploy early warning systems for natural disasters
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 2.0 MHz, Available: 1.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I want to update my social media status"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 6.0, rate: 48.0

Intent Analysis: The user wants to post a short status update to a social‑media platform. This involves sending a small payload (text, possibly a low‑resolution image) and receiving an acknowledgement. Such traffic is best served by the eMBB slice because it requires moderate bandwidth and latency that match typical smartphone usage.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 6.0 MHz
Data Rate: 48.0 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-30 11:06:27
Total Users: 24
Average Resource Utilization: 96.15%
eMBB Total Rate: 808.80 Mbps, URLLC Total Rate: 94.93 Mbps, mMTC Total Rate: 7.30 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  88.0/90 MHz       97.78%
URLLC         10  29.0/30 MHz       96.67%
mMTC           7  8.0/10 MHz        80.00%

New User Allocation:
User 25 → eMBB Slice
CQI: 7, Bandwidth: 6.0 MHz, Rate: 48.00 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          3 |          2    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |          3 |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |         30    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |         20 |        400    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |        130    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |          6 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | eMBB    |    12 |          1 |          0.8  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |          6 |         48    |             20 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |          1 |          3.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |          1 |          0.5  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 2.5

Intent Analysis: N/A
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 2.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-30 11:07:47
Total Users: 25
Average Resource Utilization: 96.92%
eMBB Total Rate: 808.80 Mbps, URLLC Total Rate: 97.43 Mbps, mMTC Total Rate: 7.30 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  88.0/90 MHz       97.78%
URLLC         11  30.0/30 MHz       100.00%
mMTC           7  8.0/10 MHz        80.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 2.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          3 |          2    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |          3 |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |          1 |          2.5  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |         30    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |         20 |        400    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |        130    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |          6 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | eMBB    |    12 |          1 |          0.8  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |          6 |         48    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |          1 |          3.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |          1 |          0.5  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need to deploy early warning systems for natural disasters"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.0

Intent Analysis: The user wants to deploy early‑warning systems for natural disasters. Such systems typically consist of many distributed sensors that transmit small measurement packets (e.g., seismic, weather, water‑level data). This traffic pattern aligns with massive Machine‑Type Communications (mMTC): many low‑rate, delay‑tolerant devices sending periodic updates. Although low latency is desirable for alerts, the current network’s URLLC slice is fully utilized, and the eMBB slice lacks sufficient bandwidth. Therefore, the mMTC slice is the most appropriate choice for the sensor network.
Recommended Slice: mMTC - The mMTC slice permits 1‑3 MHz bandwidth and 0.1‑1 Mbps rates, matching the low‑rate sensor traffic. Allocating 1 MHz stays within slice limits. The raw capacity derived from CQI 9 (≈2.5 bits/s/Hz) would exceed the slice’s 1 Mbps ceiling, so the effective rate is capped at the maximum allowed 1 Mbps.
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.0 Mbps
Latency: 0.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-30 11:09:13
Total Users: 26
Average Resource Utilization: 97.69%
eMBB Total Rate: 808.80 Mbps, URLLC Total Rate: 97.43 Mbps, mMTC Total Rate: 8.30 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  88.0/90 MHz       97.78%
URLLC         11  30.0/30 MHz       100.00%
mMTC           8  9.0/10 MHz        90.00%

New User Allocation:
User 27 → mMTC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 1.00 Mbps, Latency: 0.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          3 |          2    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |          3 |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |          1 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |         30    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |         20 |        400    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |        130    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |          6 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | eMBB    |    12 |          1 |          0.8  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |          6 |         48    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |          1 |          3.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |          1 |          0.5  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     9 |          1 |          1    |              0 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 2.4

Intent Analysis: The user wants to check the status of city-wide smart streetlights. This is a typical IoT monitoring task that involves low‑volume, periodic status reports from a large number of devices, aligning with massive Machine‑Type Communications (mMTC) traffic patterns.
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 2.4 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-30 11:10:18
Total Users: 27
Average Resource Utilization: 98.46%
eMBB Total Rate: 808.80 Mbps, URLLC Total Rate: 97.43 Mbps, mMTC Total Rate: 10.70 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  88.0/90 MHz       97.78%
URLLC         11  30.0/30 MHz       100.00%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 2.40 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          2 |          7.78 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |          5 |         16.65 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          3 |          2    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |          3 |         20    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |          1 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         12.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          3 |         14    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          2 |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |         30    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |         20 |        400    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |        130    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |          6 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | eMBB    |    12 |          1 |          0.8  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |          6 |         48    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          2 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          1    |            180 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |          1 |          3.3  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |          1 |          0.5  |            250 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     9 |          1 |          1    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     8 |          1 |          2.4  |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need to check the status of my smart home sensors"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.8

Intent Analysis: Check status of smart home sensors
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.8 Mbps
Latency: 150.0 ms

CAPACITY CHECK FAILED:
- Required: 1.0 MHz, Available: 0.0 MHz in mMTC slice

----------------------------------------
ALLOCATION FAILED FOR USER 29
----------------------------------------
Request: I need to check the status of my smart home sensors
Slice type: mMTC
Reason: Insufficient capacity in mMTC slice. Required: 1.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 5.0

Intent Analysis: N/A
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 5.0 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 1.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 30
----------------------------------------
Request: I need to balance electrical load in real-time across microgrids
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 1.0 MHz, Available: 0.0 MHz

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |     8 |         15 |        100    |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | URLLC   | URLLC          | Yes            |     5 |          2 |          5    |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | URLLC   | URLLC          | Yes            |    11 |          5 |         12.5  |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | mMTC    | mMTC           | Yes            |     8 |          1 |          1    |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | URLLC   | URLLC          | Yes            |    14 |          3 |         14    |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | mMTC    | mMTC           | Yes            |     7 |          1 |          0.5  |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |     6 |          2 |          2    |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | URLLC          | Yes            |     8 |          2 |          5    |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | mMTC    | mMTC           | Yes            |     7 |          1 |          0.5  |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | mMTC    | mMTC           | Yes            |     9 |          2 |          0.5  |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |    15 |          2 |          7.78 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | mMTC    | mMTC           | Yes            |     7 |          1 |          1    |            180 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | eMBB           | No             |     8 |          5 |         16.65 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | URLLC   | URLLC          | Yes            |     9 |          3 |          2    |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |     9 |         20 |         30    |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | eMBB    | eMBB           | Yes            |    12 |         20 |        400    |             10 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | URLLC   | URLLC          | Yes            |     7 |          2 |         10    |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | mMTC    | mMTC           | Yes            |     8 |          1 |          3.3  |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | URLLC   | URLLC          | Yes            |     7 |          3 |         20    |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    11 |         20 |        130    |             30 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | eMBB    | eMBB           | Yes            |    12 |          6 |        100    |             10 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | eMBB    | mMTC           | No             |    12 |          1 |          0.8  |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | mMTC    | mMTC           | Yes            |    15 |          1 |          0.5  |            250 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Failed   | URLLC   | URLLC          |                |     8 |          2 |         12    |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | eMBB    | eMBB           | Yes            |     7 |          6 |         48    |             20 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | URLLC          | Yes            |     9 |          1 |          2.5  |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | mMTC    | URLLC          | No             |     9 |          1 |          1    |              0 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | mMTC    | mMTC           | Yes            |     8 |          1 |          2.4  |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Failed   | mMTC    | mMTC           |                |     8 |          1 |          0.8  |            150 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Failed   | URLLC   | URLLC          |                |     7 |          1 |          5    |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 27/30 (90.0%)

Intent Understanding Evaluation:
Correctly identified intents: 24/27
Intent understanding rate: 88.9%

Workload Balancing Statistics:
Users with workload balancing: 27/30
Workload balancing rate: 90.0%

Slice Utilization Statistics:
Average eMBB utilization: 46.79%
Average URLLC utilization: 65.56%
Average mMTC utilization: 50.00%

Results exported to F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_north_minimax-m2.5.csv

[OK] TJU_north 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_north_minimax-m2.5.csv