
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
[Token] Prompt_Based: prompt~345, completion~1360, total~1705

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 15.0, rate: 250.0

Intent Analysis: User requires augmented reality navigation which demands high bandwidth for real-time 3D graphics rendering, low latency for responsive AR overlays, and consistent data flow for mapping updates and positioning data.
Recommended Slice: eMBB - AR navigation requires high data rates (50-200+ Mbps) for video streaming, 3D model loading, and texture mapping. eMBB slice provides the necessary bandwidth capacity (6-20 MHz) and data rates (100-400 Mbps) to support smooth AR experience. URLLC offers lower latency but insufficient bandwidth for AR graphics-intensive operations.
Bandwidth Allocation: 15.0 MHz
Data Rate: 250.0 Mbps
Latency: 25.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-04-05 22:56:26
Total Users: 1
Average Resource Utilization: 11.54%
eMBB Total Rate: 250.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → eMBB Slice
CQI: 8, Bandwidth: 15.0 MHz, Rate: 250.00 Mbps, Latency: 25.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |     8 |         15 |           250 |             25 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~349, completion~1312, total~1661

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 4.0

Intent Analysis: The user requests immediate machine shutdown capability for safety incidents. This is a safety‑critical, time‑sensitive command that requires ultra‑reliable low‑latency communications (URLLC) rather than high throughput or massive machine‑type connectivity.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 2.0 MHz
Data Rate: 4.0 Mbps
Latency: 2.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-04-05 22:56:49
Total Users: 2
Average Resource Utilization: 13.08%
eMBB Total Rate: 250.00 Mbps, URLLC Total Rate: 4.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          1  2.0/30 MHz        6.67%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → URLLC Slice
CQI: 5, Bandwidth: 2.0 MHz, Rate: 4.00 Mbps, Latency: 2.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |             4 |              2 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |           250 |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 11
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~355, completion~3190, total~3545

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 18.0

Intent Analysis: Real-time balancing of electrical load across microgrids requires low-latency control signaling and reliable data exchange. This is a closed-loop control application (telemetry and command execution) with moderate data volume but strict timing constraints.
Recommended Slice: URLLC - CQI 11 indicates good channel conditions allowing for 64-QAM modulation. Allocating 5 MHz (maximum URLLC bandwidth) provides sufficient data rate for telemetry while maintaining the low-latency characteristics (<10ms) essential for real-time grid stability control. The slice utilization (2.0/30 MHz) allows for this allocation without resource contention.
Bandwidth Allocation: 5.0 MHz
Data Rate: 18.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-04-05 22:57:44
Total Users: 3
Average Resource Utilization: 16.92%
eMBB Total Rate: 250.00 Mbps, URLLC Total Rate: 22.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          2  7.0/30 MHz        23.33%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 3 → URLLC Slice
CQI: 11, Bandwidth: 5.0 MHz, Rate: 18.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |             4 |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |            18 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |           250 |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~2747, total~3101

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.1

Intent Analysis: The user wants to check the status of a large set of city‑wide smart streetlights. This is a monitoring/IoT‑type request that involves gathering data from many distributed devices, which is characteristic of massive Machine‑Type Communications (mMTC). The traffic is not bandwidth‑intensive and does not require ultra‑low latency, making the mMTC slice the most appropriate choice.
Recommended Slice: mMTC - The requested status check does not need high throughput; a low‑order modulation (QPSK) with a modest code rate reduces spectral efficiency to meet the mMTC rate ceiling while staying well above the minimum rate.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.1 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-04-05 22:58:31
Total Users: 4
Average Resource Utilization: 17.69%
eMBB Total Rate: 250.00 Mbps, URLLC Total Rate: 22.00 Mbps, mMTC Total Rate: 0.10 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          2  7.0/30 MHz        23.33%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 4 → mMTC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 0.10 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           4   |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          18   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         250   |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           0.1 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I want to use remote surgery equipment"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~1998, total~2349

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 5.0

Intent Analysis: The user requests remote surgery equipment, which demands ultra‑reliable, low‑latency communication with sufficient bandwidth for high‑definition video, instrument control, and feedback. This aligns with the URLLC slice profile (latency 1‑10 ms, moderate bandwidth, rate up to 100 Mbps).
Recommended Slice: URLLC - Bandwidth set to the maximum allowed for URLLC (5 MHz) to provide enough throughput for remote surgery while keeping latency well below 10 ms.
Bandwidth Allocation: 5.0 MHz
Data Rate: 5.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-04-05 22:59:02
Total Users: 5
Average Resource Utilization: 21.54%
eMBB Total Rate: 250.00 Mbps, URLLC Total Rate: 27.00 Mbps, mMTC Total Rate: 0.10 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          3  12.0/30 MHz       40.00%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 5 → URLLC Slice
CQI: 14, Bandwidth: 5.0 MHz, Rate: 5.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           4   |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          18   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |           5   |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         250   |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           0.1 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~356, completion~1930, total~2286

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: The request is for a low‑rate, periodic status report from a smart parking sensor. Such traffic is characteristic of massive machine‑type communications (mMTC) which tolerate higher latency and require modest bandwidth. CQI 7 indicates moderate channel quality, allowing a spectral efficiency around 2‑3 bits/s/Hz, but the slice’s rate上限 limits the usable throughput. The mMTC slice currently uses 1 MHz of its 10 MHz spectrum (10 % utilization), leaving ample room for an additional sensor.
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-04-05 22:59:33
Total Users: 6
Average Resource Utilization: 22.31%
eMBB Total Rate: 250.00 Mbps, URLLC Total Rate: 27.00 Mbps, mMTC Total Rate: 0.60 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          3  12.0/30 MHz       40.00%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 6 → mMTC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           4   |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          18   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         250   |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           0.1 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |           0.5 |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2327, total~2680

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 11.5

Intent Analysis: Immediate machine shutdown capability for safety incidents
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 11.5 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-04-05 23:00:03
Total Users: 7
Average Resource Utilization: 26.15%
eMBB Total Rate: 250.00 Mbps, URLLC Total Rate: 38.50 Mbps, mMTC Total Rate: 0.60 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          4  17.0/30 MHz       56.67%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 6, Bandwidth: 5.0 MHz, Rate: 11.50 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           4   |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          18   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          5 |          11.5 |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         250   |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           0.1 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I need to transmit real-time patient vital signs during critical care"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~356, completion~1993, total~2349

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 5.0

Intent Analysis: Transmission of real‑time patient vital signs in a critical‑care setting requires ultra‑reliable low‑latency communication (URLLC). The request does not need high bandwidth but demands latency well below 10 ms and a stable data rate.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 2.0 MHz
Data Rate: 5.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-04-05 23:00:41
Total Users: 8
Average Resource Utilization: 27.69%
eMBB Total Rate: 250.00 Mbps, URLLC Total Rate: 43.50 Mbps, mMTC Total Rate: 0.60 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          5  19.0/30 MHz       63.33%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 8, Bandwidth: 2.0 MHz, Rate: 5.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           4   |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          18   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          5 |          11.5 |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |           5   |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         250   |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           0.1 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~356, completion~3259, total~3615

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 5.0

Intent Analysis: The user needs a modest amount of bandwidth to transmit occasional telemetry frames. The traffic is not bandwidth‑hungry, but it does require relatively low latency to keep the drone fleet manageable.
Recommended Slice: URLLC - URLLC supports ultra‑low latency (1‑10 ms) which is suitable for real‑time telemetry.; The required data rate (≈ 1‑5 Mbps) falls comfortably within the URLLC rate range (1‑100 Mbps).; Available bandwidth in the URLLC slice (≈ 11 MHz free) can accommodate the request without moving users to other slices.; Using URLLC rather than eMBB avoids over‑provisioning high‑speed resources that are not needed for low‑rate telemetry.; The mMTC slice, while also low‑rate, imposes latency of 100‑1000 ms, which is too high for drone telemetry.
Bandwidth Allocation: 2.0 MHz
Data Rate: 5.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-04-05 23:01:39
Total Users: 9
Average Resource Utilization: 29.23%
eMBB Total Rate: 250.00 Mbps, URLLC Total Rate: 48.50 Mbps, mMTC Total Rate: 0.60 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          6  21.0/30 MHz       70.00%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 9 → URLLC Slice
CQI: 7, Bandwidth: 2.0 MHz, Rate: 5.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           4   |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          18   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          5 |          11.5 |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          2 |           5   |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         250   |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           0.1 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~1656, total~2010

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: The request is for a network of environmental sensors reporting air‑quality data. This is a typical massive Machine‑Type Communication (mMTC) scenario, characterized by a large number of low‑throughput devices that tolerate higher latency and do not require ultra‑reliable low‑latency links.
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-04-05 23:02:07
Total Users: 10
Average Resource Utilization: 30.0%
eMBB Total Rate: 250.00 Mbps, URLLC Total Rate: 48.50 Mbps, mMTC Total Rate: 1.10 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          6  21.0/30 MHz       70.00%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 10 → mMTC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |           4   |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          18   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          5 |          11.5 |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         250   |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          1 |           0.5 |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           0.1 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~1906, total~2260

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 30.0

Intent Analysis: The request requires ultra‑reliable, low‑latency communication (URLLC) to transmit high‑frequency sensor data and receive control commands with near‑real‑time response. Any delay could lead to cascading failures in the power grid.
Recommended Slice: URLLC - URLLC slice provides the required low latency and high reliability. With a CQI of 15 (excellent channel quality) we can allocate the maximum permitted bandwidth (5 MHz) to achieve a comfortable data rate while keeping latency low. The allocation leaves 4 MHz of spare capacity in the URLLC slice, preserving room for additional URLLC users.
Bandwidth Allocation: 5.0 MHz
Data Rate: 30.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-04-05 23:02:41
Total Users: 11
Average Resource Utilization: 33.85%
eMBB Total Rate: 250.00 Mbps, URLLC Total Rate: 78.50 Mbps, mMTC Total Rate: 1.10 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          7  26.0/30 MHz       86.67%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 15, Bandwidth: 5.0 MHz, Rate: 30.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          5 |          30   |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |           4   |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          18   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          5 |          11.5 |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         250   |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           0.1 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need to check the status of my smart home sensors"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~355, completion~2675, total~3030

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: Check status of smart home sensors (IoT monitoring)
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-04-05 23:03:25
Total Users: 12
Average Resource Utilization: 34.62%
eMBB Total Rate: 250.00 Mbps, URLLC Total Rate: 78.50 Mbps, mMTC Total Rate: 1.60 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          7  26.0/30 MHz       86.67%
mMTC           4  4.0/10 MHz        40.00%

New User Allocation:
User 12 → mMTC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          5 |          30   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |           4   |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          18   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          5 |          11.5 |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         250   |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |           0.5 |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           0.1 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need to check weather forecasts"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~350, completion~1632, total~1982

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 100.0

Intent Analysis: Check weather forecasts
Recommended Slice: eMBB - Weather‑forecast data is moderate in volume and tolerant of latency. The eMBB slice provides the necessary bandwidth (10 MHz) and can deliver the required 100 Mbps rate within its latency window (10‑100 ms), while keeping utilization low.
Bandwidth Allocation: 10.0 MHz
Data Rate: 100.0 Mbps
Latency: 50.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-04-05 23:03:54
Total Users: 13
Average Resource Utilization: 42.31%
eMBB Total Rate: 350.00 Mbps, URLLC Total Rate: 78.50 Mbps, mMTC Total Rate: 1.60 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  25.0/90 MHz       27.78%
URLLC          7  26.0/30 MHz       86.67%
mMTC           4  4.0/10 MHz        40.00%

New User Allocation:
User 13 → eMBB Slice
CQI: 8, Bandwidth: 10.0 MHz, Rate: 100.00 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          5 |          30   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |           4   |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |          18   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          5 |          11.5 |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          2 |           5   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |         250   |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |         10 |         100   |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |           0.1 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |           0.5 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~1667, total~2019

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 19.25

Intent Analysis: Mission-critical public safety communication requiring ultra-reliable, low-latency connectivity for firefighters operating in challenging indoor environments. This is a life-safety application requiring real-time voice/video communication and situational awareness.
Recommended Slice: URLLC - Firefighter communication is a canonical URLLC use case requiring: (1) ultra-low latency (1-10ms) for real-time coordination, (2) ultra-high reliability for critical infrastructure, (3) moderate bandwidth for voice/video transmission. CQI=9 indicates good channel conditions which supports the reliability requirement.
Bandwidth Allocation: 2.0 MHz
Data Rate: 19.25 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-04-05 23:04:29
Total Users: 14
Average Resource Utilization: 43.85%
eMBB Total Rate: 350.00 Mbps, URLLC Total Rate: 97.75 Mbps, mMTC Total Rate: 1.60 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  25.0/90 MHz       27.78%
URLLC          8  28.0/30 MHz       93.33%
mMTC           4  4.0/10 MHz        40.00%

New User Allocation:
User 14 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 19.25 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          5 |         30    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          2 |         19.25 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          4    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         18    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          5 |         11.5  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        250    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |         10 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use maps for basic navigation"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~1991, total~2343

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 100.0

Intent Analysis: The user requests basic navigation maps, which involve low‑to‑moderate data traffic with tolerance for moderate latency. An eMBB slice is the most suitable, as it provides the required bandwidth and rate range while meeting the latency budget.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 20.0 MHz
Data Rate: 100.0 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-04-05 23:05:08
Total Users: 15
Average Resource Utilization: 59.23%
eMBB Total Rate: 450.00 Mbps, URLLC Total Rate: 97.75 Mbps, mMTC Total Rate: 1.60 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  45.0/90 MHz       50.00%
URLLC          8  28.0/30 MHz       93.33%
mMTC           4  4.0/10 MHz        40.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 9, Bandwidth: 20.0 MHz, Rate: 100.00 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          5 |         30    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          2 |         19.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          4    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         18    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          5 |         11.5  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        250    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |         10 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |        100    |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need to stream 8K video content"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~1335, total~1688

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 100.0, rate: 4.2

Intent Analysis: N/A
Recommended Slice: eMBB - 8K video streaming requires high data rates (100+ Mbps) which aligns with eMBB slice capabilities (100-400 Mbps). URLLC is designed for ultra-reliable low-latency communications but with lower data rates. mMTC is designed for massive machine-type communications with very low bandwidth requirements.
Bandwidth Allocation: 100.0 MHz
Data Rate: 4.2 Mbps
Latency: 30.0 ms

CAPACITY CHECK FAILED:
- Required: 100.0 MHz, Available: 45.0 MHz in eMBB slice

----------------------------------------
ALLOCATION FAILED FOR USER 16
----------------------------------------
Request: I need to stream 8K video content
Slice type: eMBB
Reason: Insufficient capacity in eMBB slice. Required: 100.0 MHz, Available: 45.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need instant alerts for life-threatening patient conditions"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~1871, total~2224

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 5.0

Intent Analysis: N/A
Recommended Slice: URLLC - The request 'instant alerts for life‑threatening patient conditions' requires ultra‑reliable low‑latency communication. The URLLC slice supports the needed latency of 1‑10 ms and is the only slice that meets the stringent delay constraint. With a CQI of 7, the corresponding spectral efficiency is ≈2.4 bits/Hz, giving a theoretical throughput of ≈4.8 Mbps on a 2 MHz allocation. Allocating 2 MHz and a data rate of 5 Mbps satisfies the alert traffic (a few kilobytes) while staying within the slice's allowed range (1‑5 MHz, 1‑100 Mbps). The URLLC slice currently uses 28 MHz out of its 30 MHz capacity, leaving exactly 2 MHz for this user; after allocation the slice reaches 100% utilization, leaving no headroom for additional URLLC traffic. The eMBB slice remains at 50% utilization and could be leveraged for non‑critical offloading if needed.
Bandwidth Allocation: 2.0 MHz
Data Rate: 5.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-04-05 23:06:04
Total Users: 16
Average Resource Utilization: 60.77%
eMBB Total Rate: 450.00 Mbps, URLLC Total Rate: 102.75 Mbps, mMTC Total Rate: 1.60 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  45.0/90 MHz       50.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           4  4.0/10 MHz        40.00%

New User Allocation:
User 17 → URLLC Slice
CQI: 7, Bandwidth: 2.0 MHz, Rate: 5.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          5 |         30    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          2 |         19.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |          5    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          4    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         18    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          5 |         11.5  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        250    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |         10 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "My smart meter needs to report its reading"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~3990, total~4342

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.914

Intent Analysis: Smart meter reporting low‑volume, delay‑tolerant data
Recommended Slice: mMTC - The request is a typical low‑rate IoT transmission best served by the massive Machine‑Type Communication (mMTC) slice. The mMTC slice currently uses 4 MHz of its 10 MHz capacity, leaving 6 MHz free, so allocating the minimum 1 MHz satisfies the bandwidth limits while preserving resources for future devices. By limiting the effective data rate to 0.9 Mbps we stay within the slice’s 0.1‑1 Mbps envelope, and the 150 ms latency comfortably fits the 100‑1000 ms mMTC latency window. Adding this user raises mMTC utilization to 50 %, keeping the slice well‑balanced and leaving eMBB at 50 % and URLLC already saturated.
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.914 Mbps
Latency: 150.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-04-05 23:07:10
Total Users: 17
Average Resource Utilization: 61.54%
eMBB Total Rate: 450.00 Mbps, URLLC Total Rate: 102.75 Mbps, mMTC Total Rate: 3.51 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  45.0/90 MHz       50.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           5  5.0/10 MHz        50.00%

New User Allocation:
User 18 → mMTC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 1.91 Mbps, Latency: 150.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          5 |         30    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          2 |         19.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          4    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         18    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          5 |         11.5  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        250    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |         10 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |          1 |          1.91 |            150 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to control a robotic arm in real time"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~1999, total~2353

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 7.0

Intent Analysis: URLLC - Ultra-Reliable Low-Latency Communications
Recommended Slice: URLLC - Real-time robotic arm control requires ultra-low latency (1-10ms) as specified in URLLC constraints. Control signals must be transmitted with minimal delay to ensure precise and safe operation of the robotic arm.
Bandwidth Allocation: 1.0 MHz
Data Rate: 7.0 Mbps
Latency: 2.5 ms

CAPACITY CHECK FAILED:
- Required: 1.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 19
----------------------------------------
Request: I need to control a robotic arm in real time
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 1.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I want to watch 4K video"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~1963, total~2315

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 8.0, rate: 160.0

Intent Analysis: N/A
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 8.0 MHz
Data Rate: 160.0 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-04-05 23:08:06
Total Users: 18
Average Resource Utilization: 67.69%
eMBB Total Rate: 610.00 Mbps, URLLC Total Rate: 102.75 Mbps, mMTC Total Rate: 3.51 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  53.0/90 MHz       58.89%
URLLC          9  30.0/30 MHz       100.00%
mMTC           5  5.0/10 MHz        50.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 11, Bandwidth: 8.0 MHz, Rate: 160.00 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          5 |         30    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          2 |         19.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          4    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         18    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          5 |         11.5  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        250    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |         10 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |          8 |        160    |             20 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |          1 |          1.91 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to stream music while browsing social media"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2614, total~2967

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 15.0, rate: 108.0

Intent Analysis: N/A
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 15.0 MHz
Data Rate: 108.0 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-04-05 23:08:53
Total Users: 19
Average Resource Utilization: 79.23%
eMBB Total Rate: 718.00 Mbps, URLLC Total Rate: 102.75 Mbps, mMTC Total Rate: 3.51 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  68.0/90 MHz       75.56%
URLLC          9  30.0/30 MHz       100.00%
mMTC           5  5.0/10 MHz        50.00%

New User Allocation:
User 21 → eMBB Slice
CQI: 12, Bandwidth: 15.0 MHz, Rate: 108.00 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |          5 |         30    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |          2 |         19.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |          2 |          4    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         18    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |          5 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          5 |         11.5  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         15 |        250    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |         10 |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |         20 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |          8 |        160    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |         15 |        108    |             20 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |          1 |          1.91 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |          1 |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |          1 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "My asset tracking device needs to send location update"
CQI: 12
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~1554, total~1907

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 0.3, rate: 0.78

Intent Analysis: Asset tracking device sending periodic location updates - typical IoT/IIoT use case with small payload size (coordinates, timestamp)
Recommended Slice: mMTC - Asset tracking devices are classified as IoT/MTC devices. Location updates are small data transmissions that do not require high bandwidth or low latency. The mMTC slice is designed for massive machine type communications with tolerance for higher latency.
Bandwidth Allocation: 0.3 MHz
Data Rate: 0.78 Mbps
Latency: 100.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-04-05 23:09:19
Total Users: 20
Average Resource Utilization: 79.46%
eMBB Total Rate: 718.00 Mbps, URLLC Total Rate: 102.75 Mbps, mMTC Total Rate: 4.29 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  68.0/90 MHz       75.56%
URLLC          9  30.0/30 MHz       100.00%
mMTC           6  5.3/10 MHz        53.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 12, Bandwidth: 0.3 MHz, Rate: 0.78 Mbps, Latency: 100.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         30    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |          4    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         18    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         11.5  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       15   |        250    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       10   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        8   |        160    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        1   |          1.91 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.3 |          0.78 |            100 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1   |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~1802, total~2156

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.0

Intent Analysis: N/A
Recommended Slice: mMTC - Environmental sensors generate low‑volume, periodic data. The mMTC slice is intended for massive machine‑type communications with low bandwidth (1‑3 MHz) and low data‑rate (0.1‑1 Mbps) requirements, well within the allowed latency (100‑1000 ms). CQI = 15 indicates a strong channel, allowing the maximum permissible rate of 1 Mbps on the allocated 1 MHz, while leaving ample headroom for other mMTC devices. This choice preserves the heavily loaded eMBB and fully saturated URLLC slices.
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.0 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-04-05 23:09:45
Total Users: 21
Average Resource Utilization: 80.23%
eMBB Total Rate: 718.00 Mbps, URLLC Total Rate: 102.75 Mbps, mMTC Total Rate: 5.29 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  68.0/90 MHz       75.56%
URLLC          9  30.0/30 MHz       100.00%
mMTC           7  6.3/10 MHz        63.00%

New User Allocation:
User 23 → mMTC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 1.00 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         30    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |          4    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         18    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         11.5  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       15   |        250    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       10   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        8   |        160    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        1   |          1.91 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.3 |          0.78 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        1   |          1    |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1   |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to deploy early warning systems for natural disasters"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~1338, total~1692

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 0.876

Intent Analysis: N/A
Recommended Slice: mMTC - Early warning systems primarily consist of IoT sensors (seismographs, weather stations, flood monitors) that require energy-efficient, massive machine-type connectivity. While URLLC would be ideal for latency-critical alerts, it is at 100% capacity. mMTC provides appropriate characteristics for sensor data aggregation with available capacity.
Bandwidth Allocation: 2.0 MHz
Data Rate: 0.876 Mbps
Latency: 150.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-04-05 23:10:08
Total Users: 22
Average Resource Utilization: 81.77%
eMBB Total Rate: 718.00 Mbps, URLLC Total Rate: 102.75 Mbps, mMTC Total Rate: 6.17 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  68.0/90 MHz       75.56%
URLLC          9  30.0/30 MHz       100.00%
mMTC           8  8.3/10 MHz        83.00%

New User Allocation:
User 24 → mMTC Slice
CQI: 8, Bandwidth: 2.0 MHz, Rate: 0.88 Mbps, Latency: 150.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         30    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |          4    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         18    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         11.5  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       15   |        250    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       10   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        8   |        160    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        1   |          1.91 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.3 |          0.78 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        1   |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | mMTC    |     8 |        2   |          0.88 |            150 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1   |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I want to update my social media status"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~3394, total~3746

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 48.0

Intent Analysis: User wants to post a short status update to a social‑media platform. This is a low‑volume, human‑generated data exchange that does not require ultra‑low latency or massive machine‑type connectivity.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 20.0 MHz
Data Rate: 48.0 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-04-05 23:11:07
Total Users: 23
Average Resource Utilization: 97.15%
eMBB Total Rate: 766.00 Mbps, URLLC Total Rate: 102.75 Mbps, mMTC Total Rate: 6.17 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  88.0/90 MHz       97.78%
URLLC          9  30.0/30 MHz       100.00%
mMTC           8  8.3/10 MHz        83.00%

New User Allocation:
User 25 → eMBB Slice
CQI: 7, Bandwidth: 20.0 MHz, Rate: 48.00 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         30    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |          4    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         18    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         11.5  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       15   |        250    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       10   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        8   |        160    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |         48    |             20 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        1   |          1.91 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.3 |          0.78 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        1   |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | mMTC    |     8 |        2   |          0.88 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1   |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~356, completion~5711, total~6067

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 6.0

Intent Analysis: Control precision CNC machines with zero tolerance for delay
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 2.0 MHz
Data Rate: 6.0 Mbps
Latency: 1.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-04-05 23:12:39
Total Users: 24
Average Resource Utilization: 98.69%
eMBB Total Rate: 772.00 Mbps, URLLC Total Rate: 102.75 Mbps, mMTC Total Rate: 6.17 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           8  8.3/10 MHz        83.00%

New User Allocation:
User 26 → eMBB Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 6.00 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         30    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |          4    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         18    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         11.5  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       15   |        250    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       10   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        8   |        160    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |         48    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     9 |        2   |          6    |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        1   |          1.91 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.3 |          0.78 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        1   |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | mMTC    |     8 |        2   |          0.88 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1   |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need to deploy early warning systems for natural disasters"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~4949, total~5303

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.5, rate: 6.78

Intent Analysis: The user intends to deploy early warning systems for natural disasters. Such applications require ultra‑reliable, low‑latency communication for rapid alert dissemination and reliable data collection from sensors, as well as sufficient bandwidth to handle moderate data rates from multiple devices.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 1.5 MHz
Data Rate: 6.78 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 1.5 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 27
----------------------------------------
Request: I need to deploy early warning systems for natural disasters
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 1.5 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~356, completion~1715, total~2071

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 2.5

Intent Analysis: The user wants to retrieve low‑volume status information from a large set of IoT‑enabled streetlights. This is a typical monitoring/management task that requires modest bandwidth, can tolerate higher latency, and does not need ultra‑reliable low‑latency guarantees.
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 2.5 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-04-05 23:14:28
Total Users: 25
Average Resource Utilization: 99.46%
eMBB Total Rate: 772.00 Mbps, URLLC Total Rate: 102.75 Mbps, mMTC Total Rate: 8.67 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           9  9.3/10 MHz        93.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 2.50 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         30    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.25 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |          4    |              2 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         18    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         11.5  |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       15   |        250    |             25 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       10   |        100    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        8   |        160    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |         48    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     9 |        2   |          6    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        1   |          1.91 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.3 |          0.78 |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        1   |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | mMTC    |     8 |        2   |          0.88 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     8 |        1   |          2.5  |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1   |          0.1  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        1   |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need to check the status of my smart home sensors"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~355, completion~1195, total~1550

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 0.7, rate: 0.5

Intent Analysis: N/A
Recommended Slice: mMTC - Smart home sensors are IoT devices requiring minimal bandwidth for periodic status updates. The mMTC slice is purpose-built for massive machine-type communications and perfectly matches this use case. Allocating 0.7 MHz of the available bandwidth provides sufficient capacity for sensor status polling while staying within mMTC constraints. eMBB and URLLC slices are at full capacity and are not suitable for this IoT use case.
Bandwidth Allocation: 0.7 MHz
Data Rate: 0.5 Mbps
Latency: 1.0 ms

CAPACITY CHECK FAILED:
- Required: 0.7 MHz, Available: 0.6999999999999993 MHz in mMTC slice

----------------------------------------
ALLOCATION FAILED FOR USER 29
----------------------------------------
Request: I need to check the status of my smart home sensors
Slice type: mMTC
Reason: Insufficient capacity in mMTC slice. Required: 0.7 MHz, Available: 0.6999999999999993 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~357, completion~1754, total~2111

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 2.5

Intent Analysis: N/A
Recommended Slice: URLLC - URLLC slice at 100% utilization - reallocation required
Bandwidth Allocation: 1.0 MHz
Data Rate: 2.5 Mbps
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
|         1 | Success  | eMBB    | eMBB           | Yes            |     8 |       15   |       250     |           25   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | URLLC   | URLLC          | Yes            |     5 |        2   |         4     |            2   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | URLLC   | URLLC          | Yes            |    11 |        5   |        18     |            5   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | mMTC    | mMTC           | Yes            |     8 |        1   |         0.1   |          500   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | URLLC   | URLLC          | Yes            |    14 |        5   |         5     |            5   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | mMTC    | mMTC           | Yes            |     7 |        1   |         0.5   |          200   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |     6 |        5   |        11.5   |           10   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | URLLC          | Yes            |     8 |        2   |         5     |            5   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | URLLC   | mMTC           | No             |     7 |        2   |         5     |            5   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | mMTC    | mMTC           | Yes            |     9 |        1   |         0.5   |          200   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |    15 |        5   |        30     |            5   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | mMTC    | mMTC           | Yes            |     7 |        1   |         0.5   |          200   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | eMBB    | eMBB           | Yes            |     8 |       10   |       100     |           50   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | URLLC   | URLLC          | Yes            |     9 |        2   |        19.25  |            5   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |     9 |       20   |       100     |           10   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Failed   | eMBB    | eMBB           |                |    12 |      100   |         4.2   |           30   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | URLLC   | URLLC          | Yes            |     7 |        2   |         5     |            5   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | mMTC    | mMTC           | Yes            |     8 |        1   |         1.914 |          150   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Failed   | URLLC   | URLLC          |                |     7 |        1   |         7     |            2.5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    11 |        8   |       160     |           20   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | eMBB    | eMBB           | Yes            |    12 |       15   |       108     |           20   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | mMTC    | mMTC           | Yes            |    12 |        0.3 |         0.78  |          100   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | mMTC    | mMTC           | Yes            |    15 |        1   |         1     |          200   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | mMTC    | URLLC          | No             |     8 |        2   |         0.876 |          150   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | eMBB    | eMBB           | Yes            |     7 |       20   |        48     |           20   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | eMBB    | URLLC          | No             |     9 |        2   |         6     |            1   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Failed   | URLLC   | URLLC          |                |     9 |        1.5 |         6.78  |            5   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | mMTC    | mMTC           | Yes            |     8 |        1   |         2.5   |          200   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Failed   | mMTC    | mMTC           |                |     8 |        0.7 |         0.5   |            1   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Failed   | URLLC   | URLLC          |                |     7 |        1   |         2.5   |            5   | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 25/30 (83.3%)

Intent Understanding Evaluation:
Correctly identified intents: 22/25
Intent understanding rate: 88.0%

Workload Balancing Statistics:
Users with workload balancing: 25/30
Workload balancing rate: 83.3%

Slice Utilization Statistics:
Average eMBB utilization: 42.58%
Average URLLC utilization: 73.60%
Average mMTC utilization: 38.72%

Results exported to F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_north_minimax-m2.5.csv

[OK] TJU_north 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_north_minimax-m2.5.csv