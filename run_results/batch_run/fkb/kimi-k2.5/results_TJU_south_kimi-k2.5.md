============================================================
场景 2/5: TJU_south
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_south.csv
输出: F:\code\wirelessagent\run_results\batch_run\fkb\kimi-k2.5\network_slicing_results_TJU_south_kimi-k2.5.csv
------------------------------------------------------------
Starting network slice management system with CSV-based user testing...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 2
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~778, total~2150
[Token] Slice_Type_Determination: prompt~2620, completion~77, total~2697
[Token] Beamforming_Bandwidth: prompt~214, completion~322, total~536
[Token] Allocate_Resources: prompt~2813, completion~331, total~3144

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-16 11:35:18
Total Users: 1
Average Resource Utilization: 0.77%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 4.12 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          1  1.0/30 MHz        3.33%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → URLLC Slice
CQI: 2, Bandwidth: 1.0 MHz, Rate: 4.12 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          1 |          4.12 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3387, completion~470, total~3857

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need my autonomous vehicle to communicate in real time"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need my autonomous vehicle to communicate in real time", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~771, total~2145
[Token] Slice_Type_Determination: prompt~2605, completion~110, total~2715
[Token] Beamforming_Bandwidth: prompt~217, completion~261, total~478
[Token] Allocate_Resources: prompt~2834, completion~347, total~3181

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-16 11:36:09
Total Users: 2
Average Resource Utilization: 4.62%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 27.94 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          2  6.0/30 MHz        20.00%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → URLLC Slice
CQI: 3, Bandwidth: 5.0 MHz, Rate: 23.82 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          1 |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |          5 |         23.82 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3413, completion~606, total~4019

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart agriculture sensor needs to report soil temperature", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~805, total~2177
[Token] Slice_Type_Determination: prompt~2624, completion~88, total~2712
[Token] Beamforming_Bandwidth: prompt~213, completion~4, total~217
[Token] Allocate_Resources: prompt~2862, completion~551, total~3413

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-16 11:37:00
Total Users: 3
Average Resource Utilization: 5.92%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 27.94 Mbps, mMTC Total Rate: 25.73 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          2  6.0/30 MHz        20.00%
mMTC           1  1.7/10 MHz        17.00%

New User Allocation:
User 3 → mMTC Slice
CQI: 15, Bandwidth: 1.7 MHz, Rate: 25.73 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3639, completion~641, total~4280

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to participate in an online multiplayer game"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to participate in an online multiplayer game", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~798, total~2170
[Token] Slice_Type_Determination: prompt~2630, completion~105, total~2735
WARNING: LLM didn't provide explicit slice recommendation. Using knowledge base recommendation.
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2857, completion~750, total~3607

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-16 11:38:00
Total Users: 4
Average Resource Utilization: 9.0%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 88.48 Mbps, mMTC Total Rate: 25.73 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          3  10.0/30 MHz       33.33%
mMTC           1  1.7/10 MHz        17.00%

New User Allocation:
User 4 → URLLC Slice
CQI: 15, Bandwidth: 4.0 MHz, Rate: 60.54 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3905, completion~696, total~4601

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a fleet of delivery drones needs to send low-rate telemetry data", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~942, total~2320
[Token] Slice_Type_Determination: prompt~2780, completion~104, total~2884
[Token] Beamforming_Bandwidth: prompt~218, completion~2, total~220
[Token] Allocate_Resources: prompt~3040, completion~540, total~3580

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-16 11:38:58
Total Users: 5
Average Resource Utilization: 10.46%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 88.48 Mbps, mMTC Total Rate: 34.78 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           0  0/90 MHz                   0%
URLLC          3  10.0/30 MHz                33.33%
mMTC           2  3.5999999999999996/10 MHz  36.00%

New User Allocation:
User 5 → mMTC Slice
CQI: 3, Bandwidth: 1.9 MHz, Rate: 9.05 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3808, completion~778, total~4586

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I want to play competitive mobile games with ultra-low latency"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to play competitive mobile games with ultra-low latency", "urllc"),)
[Token] Intent_Analysis: prompt~1376, completion~708, total~2084
[Token] Slice_Type_Determination: prompt~2539, completion~89, total~2628
[Token] Beamforming_Bandwidth: prompt~218, completion~2, total~220
[Token] Allocate_Resources: prompt~2744, completion~343, total~3087

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-16 11:39:39
Total Users: 6
Average Resource Utilization: 13.54%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 110.30 Mbps, mMTC Total Rate: 34.78 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           0  0/90 MHz                   0%
URLLC          4  14.0/30 MHz                46.67%
mMTC           2  3.5999999999999996/10 MHz  36.00%

New User Allocation:
User 6 → URLLC Slice
CQI: 4, Bandwidth: 4.0 MHz, Rate: 21.82 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3320, completion~635, total~3955

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to participate in an online multiplayer game"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to participate in an online multiplayer game", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~758, total~2130
[Token] Slice_Type_Determination: prompt~2590, completion~78, total~2668
[Token] Beamforming_Bandwidth: prompt~216, completion~232, total~448
[Token] Allocate_Resources: prompt~2784, completion~411, total~3195

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-16 11:40:36
Total Users: 7
Average Resource Utilization: 17.38%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 181.15 Mbps, mMTC Total Rate: 34.78 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           0  0/90 MHz                   0%
URLLC          5  19.0/30 MHz                63.33%
mMTC           2  3.5999999999999996/10 MHz  36.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 14, Bandwidth: 5.0 MHz, Rate: 70.85 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        5   |         70.85 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3434, completion~840, total~4274

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I want to browse websites and check email"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~816, total~2186
[Token] Slice_Type_Determination: prompt~2647, completion~131, total~2778
[Token] Beamforming_Bandwidth: prompt~213, completion~2, total~215
[Token] Allocate_Resources: prompt~2926, completion~332, total~3258

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-16 11:41:23
Total Users: 8
Average Resource Utilization: 32.77%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 181.15 Mbps, mMTC Total Rate: 34.78 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           1  20.0/90 MHz                22.22%
URLLC          5  19.0/30 MHz                63.33%
mMTC           2  3.5999999999999996/10 MHz  36.00%

New User Allocation:
User 8 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3512, completion~512, total~4024

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to monitor my home security cameras remotely", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~881, total~2253
[Token] Slice_Type_Determination: prompt~2722, completion~104, total~2826
LLM recommended eMBB but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~2897, completion~99, total~2996
[Token] Beamforming_Bandwidth: prompt~216, completion~402, total~618
[Token] Allocate_Resources: prompt~3111, completion~545, total~3656

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-16 11:42:25
Total Users: 9
Average Resource Utilization: 33.54%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 188.12 Mbps, mMTC Total Rate: 34.78 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           1  20.0/90 MHz                22.22%
URLLC          6  20.0/30 MHz                66.67%
mMTC           2  3.5999999999999996/10 MHz  36.00%

New User Allocation:
User 9 → URLLC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 6.97 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3960, completion~596, total~4556

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to make a high-quality voice call"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to make a high-quality voice call", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~687, total~2059
[Token] Slice_Type_Determination: prompt~2522, completion~116, total~2638
LLM recommended URLLC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~2707, completion~119, total~2826
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2972, completion~522, total~3494

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-16 11:43:12
Total Users: 10
Average Resource Utilization: 48.92%
eMBB Total Rate: 218.22 Mbps, URLLC Total Rate: 188.12 Mbps, mMTC Total Rate: 34.78 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           2  40.0/90 MHz                44.44%
URLLC          6  20.0/30 MHz                66.67%
mMTC           2  3.5999999999999996/10 MHz  36.00%

New User Allocation:
User 10 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 30.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3795, completion~753, total~4548

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to send text messages and use messaging apps"
CQI: 3
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to send text messages and use messaging apps", "embb"),)
[Token] Intent_Analysis: prompt~1374, completion~920, total~2294
[Token] Slice_Type_Determination: prompt~2752, completion~74, total~2826
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~2893, completion~63, total~2956
[Token] Beamforming_Bandwidth: prompt~217, completion~365, total~582
[Token] Allocate_Resources: prompt~3103, completion~586, total~3689

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-16 11:44:10
Total Users: 11
Average Resource Utilization: 64.31%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 188.12 Mbps, mMTC Total Rate: 34.78 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           3  60.0/90 MHz                66.67%
URLLC          6  20.0/30 MHz                66.67%
mMTC           2  3.5999999999999996/10 MHz  36.00%

New User Allocation:
User 11 → eMBB Slice
CQI: 3, Bandwidth: 20.0 MHz, Rate: 95.29 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3997, completion~966, total~4963

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I want to play competitive mobile games with ultra-low latency"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to play competitive mobile games with ultra-low latency", "urllc"),)
[Token] Intent_Analysis: prompt~1376, completion~844, total~2220
[Token] Slice_Type_Determination: prompt~2689, completion~108, total~2797
[Token] Beamforming_Bandwidth: prompt~218, completion~277, total~495
[Token] Allocate_Resources: prompt~2915, completion~369, total~3284

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-16 11:45:12
Total Users: 12
Average Resource Utilization: 66.62%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 202.41 Mbps, mMTC Total Rate: 34.78 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           3  60.0/90 MHz                66.67%
URLLC          7  23.0/30 MHz                76.67%
mMTC           2  3.5999999999999996/10 MHz  36.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 3, Bandwidth: 3.0 MHz, Rate: 14.29 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3530, completion~651, total~4181

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need real-time traffic updates for navigation"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need real-time traffic updates for navigation", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~1019, total~2389
[Token] Slice_Type_Determination: prompt~2853, completion~93, total~2946
LLM recommended eMBB but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~3017, completion~89, total~3106
[Token] Beamforming_Bandwidth: prompt~215, completion~275, total~490
[Token] Allocate_Resources: prompt~3222, completion~370, total~3592

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-16 11:46:13
Total Users: 13
Average Resource Utilization: 70.46%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 249.98 Mbps, mMTC Total Rate: 34.78 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           3  60.0/90 MHz                66.67%
URLLC          8  28.0/30 MHz                93.33%
mMTC           2  3.5999999999999996/10 MHz  36.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 9, Bandwidth: 5.0 MHz, Rate: 47.57 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        5   |         47.57 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3885, completion~630, total~4515

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my wearable device needs to upload health data periodically", "mmtc"),, ("my wearable device needs to upload health data periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~713, total~2085
[Token] Slice_Type_Determination: prompt~2538, completion~61, total~2599
[Token] Beamforming_Bandwidth: prompt~215, completion~277, total~492
[Token] Allocate_Resources: prompt~2752, completion~659, total~3411

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-16 11:47:08
Total Users: 14
Average Resource Utilization: 71.92%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 249.98 Mbps, mMTC Total Rate: 48.03 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          8  28.0/30 MHz       93.33%
mMTC           3  5.5/10 MHz        55.00%

New User Allocation:
User 14 → mMTC Slice
CQI: 6, Bandwidth: 1.9 MHz, Rate: 13.25 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        1.9 |         13.25 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3657, completion~632, total~4289

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 13
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[Token] Intent_Analysis: prompt~1376, completion~623, total~1999
[Token] Slice_Type_Determination: prompt~2458, completion~107, total~2565
[Token] Beamforming_Bandwidth: prompt~218, completion~2, total~220
[Token] Allocate_Resources: prompt~2682, completion~311, total~2993

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-16 11:47:45
Total Users: 15
Average Resource Utilization: 83.46%
eMBB Total Rate: 511.70 Mbps, URLLC Total Rate: 249.98 Mbps, mMTC Total Rate: 48.03 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  75.0/90 MHz       83.33%
URLLC          8  28.0/30 MHz       93.33%
mMTC           3  5.5/10 MHz        55.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 13, Bandwidth: 15.0 MHz, Rate: 198.19 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       15   |        198.19 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3232, completion~588, total~3820

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need instant facial recognition for public security threats"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need instant facial recognition for public security threats", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~814, total~2186
[Token] Slice_Type_Determination: prompt~2658, completion~100, total~2758
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2877, completion~307, total~3184

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-16 11:48:24
Total Users: 16
Average Resource Utilization: 85.0%
eMBB Total Rate: 511.70 Mbps, URLLC Total Rate: 269.01 Mbps, mMTC Total Rate: 48.03 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  75.0/90 MHz       83.33%
URLLC          9  30.0/30 MHz       100.00%
mMTC           3  5.5/10 MHz        55.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 19.03 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        2   |         19.03 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       15   |        198.19 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3421, completion~770, total~4191

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to monitor and control critical manufacturing processes in real-time"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to monitor and control critical manufacturing processes in real-time", "urllc"),)
[Token] Intent_Analysis: prompt~1378, completion~698, total~2076
[Token] Slice_Type_Determination: prompt~2531, completion~81, total~2612
[Token] Beamforming_Bandwidth: prompt~219, completion~4, total~223
[Token] Allocate_Resources: prompt~2776, completion~438, total~3214

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-16 11:49:15
Total Users: 17
Average Resource Utilization: 85.0%
eMBB Total Rate: 511.70 Mbps, URLLC Total Rate: 269.98 Mbps, mMTC Total Rate: 48.03 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  75.0/90 MHz       83.33%
URLLC         10  30.0/30 MHz       100.00%
mMTC           3  5.5/10 MHz        55.00%

New User Allocation:
User 17 → URLLC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 15.14 Mbps, Latency: 3.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 7: 5.0 → 4.0 MHz, Rate: 70.85 → 56.68 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        4   |         56.68 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       15   |        198.19 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3539, completion~895, total~4434

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need to check the status of my smart home sensors"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of my smart home sensors", "mmtc"),, ("i need to check the status of my smart home sensors", "mmtc"),)
[Token] Intent_Analysis: prompt~1376, completion~737, total~2113
[Token] Slice_Type_Determination: prompt~2564, completion~75, total~2639
[Token] Beamforming_Bandwidth: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~2792, completion~710, total~3502

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-16 11:50:27
Total Users: 18
Average Resource Utilization: 86.46%
eMBB Total Rate: 511.70 Mbps, URLLC Total Rate: 269.98 Mbps, mMTC Total Rate: 57.08 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  75.0/90 MHz       83.33%
URLLC         10  30.0/30 MHz       100.00%
mMTC           4  7.4/10 MHz        74.00%

New User Allocation:
User 18 → mMTC Slice
CQI: 3, Bandwidth: 1.9 MHz, Rate: 9.05 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       15   |        198.19 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        1.9 |          9.05 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3736, completion~836, total~4572

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 2
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("industrial equipment monitoring sensor data", "mmtc"))
[Token] Intent_Analysis: prompt~1364, completion~805, total~2169
[Token] Slice_Type_Determination: prompt~2623, completion~69, total~2692
WARNING: LLM didn't provide explicit slice recommendation. Using knowledge base recommendation.
[Token] Beamforming_Bandwidth: prompt~231, completion~2, total~233
[Token] Allocate_Resources: prompt~2845, completion~656, total~3501

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-16 11:51:29
Total Users: 19
Average Resource Utilization: 87.15%
eMBB Total Rate: 511.70 Mbps, URLLC Total Rate: 269.98 Mbps, mMTC Total Rate: 60.79 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  75.0/90 MHz       83.33%
URLLC         10  30.0/30 MHz       100.00%
mMTC           5  8.3/10 MHz        83.00%

New User Allocation:
User 19 → mMTC Slice
CQI: 2, Bandwidth: 0.9 MHz, Rate: 3.71 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       15   |        198.19 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3794, completion~740, total~4534

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I want to download a big game file"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to download a big game file", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~613, total~1983
[Token] Slice_Type_Determination: prompt~2448, completion~86, total~2534
[Token] Beamforming_Bandwidth: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~2652, completion~336, total~2988

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-16 11:52:19
Total Users: 20
Average Resource Utilization: 98.69%
eMBB Total Rate: 681.68 Mbps, URLLC Total Rate: 269.98 Mbps, mMTC Total Rate: 60.79 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         10  30.0/30 MHz       100.00%
mMTC           5  8.3/10 MHz        83.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 11, Bandwidth: 15.0 MHz, Rate: 169.98 Mbps, Latency: 80.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       15   |        198.19 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       15   |        169.98 |             80 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3233, completion~672, total~3905

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to update my social media status"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~808, total~2178
[Token] Slice_Type_Determination: prompt~2645, completion~77, total~2722
[Token] Beamforming_Bandwidth: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~2912, completion~799, total~3711
[Token] Failure_Evaluation: prompt~3820, completion~667, total~4487

----------------------------------------
ALLOCATION FAILED FOR USER 21
----------------------------------------
Request: I want to update my social media status
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 2
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control critical infrastructure with zero downtime", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~741, total~2113
[Token] Slice_Type_Determination: prompt~2568, completion~116, total~2684
[Token] Beamforming_Bandwidth: prompt~216, completion~4, total~220
[Token] Allocate_Resources: prompt~2849, completion~381, total~3230

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-16 11:54:05
Total Users: 21
Average Resource Utilization: 98.69%
eMBB Total Rate: 681.68 Mbps, URLLC Total Rate: 258.97 Mbps, mMTC Total Rate: 60.79 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         11  30.0/30 MHz       100.00%
mMTC           5  8.3/10 MHz        83.00%

New User Allocation:
User 22 → URLLC Slice
CQI: 2, Bandwidth: 1.0 MHz, Rate: 4.12 Mbps, Latency: 3.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 4: 4.0 → 3.0 MHz, Rate: 60.54 → 45.41 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        1   |          4.12 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       15   |        198.19 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       15   |        169.98 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3543, completion~551, total~4094

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I want to stream a webinar with interactive features"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to stream a webinar with interactive features", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~885, total~2257
[Token] Slice_Type_Determination: prompt~2732, completion~82, total~2814
[Token] Beamforming_Bandwidth: prompt~216, completion~4, total~220
[Token] Allocate_Resources: prompt~3005, completion~407, total~3412
[Token] Failure_Evaluation: prompt~3501, completion~408, total~3909

----------------------------------------
ALLOCATION FAILED FOR USER 23
----------------------------------------
Request: I want to stream a webinar with interactive features
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[Token] Intent_Analysis: prompt~1376, completion~588, total~1964
[Token] Slice_Type_Determination: prompt~2423, completion~64, total~2487
[Token] Beamforming_Bandwidth: prompt~218, completion~4, total~222
[Token] Allocate_Resources: prompt~2678, completion~610, total~3288
[Token] Failure_Evaluation: prompt~3378, completion~313, total~3691

----------------------------------------
ALLOCATION FAILED FOR USER 24
----------------------------------------
Request: I need to use cloud-based AI services for image processing
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I need to synchronize distributed financial ledgers instantly"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to synchronize distributed financial ledgers instantly", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~748, total~2120
[Token] Slice_Type_Determination: prompt~2586, completion~128, total~2714
[Token] Beamforming_Bandwidth: prompt~216, completion~4, total~220
[Token] Allocate_Resources: prompt~2880, completion~383, total~3263

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-16 11:56:36
Total Users: 22
Average Resource Utilization: 98.69%
eMBB Total Rate: 681.68 Mbps, URLLC Total Rate: 258.97 Mbps, mMTC Total Rate: 60.79 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         12  30.0/30 MHz       100.00%
mMTC           5  8.3/10 MHz        83.00%

New User Allocation:
User 25 → URLLC Slice
CQI: 14, Bandwidth: 1.0 MHz, Rate: 14.17 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 7: 4.0 → 3.0 MHz, Rate: 56.68 → 42.51 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        3   |         42.51 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       15   |        198.19 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       15   |        169.98 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3588, completion~678, total~4266

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to control a robotic arm in real time"
CQI: 2
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control a robotic arm in real time", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~799, total~2173
[Token] Slice_Type_Determination: prompt~2639, completion~120, total~2759
[Token] Beamforming_Bandwidth: prompt~217, completion~4, total~221
[Token] Allocate_Resources: prompt~2927, completion~525, total~3452

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-16 11:57:37
Total Users: 23
Average Resource Utilization: 98.69%
eMBB Total Rate: 681.68 Mbps, URLLC Total Rate: 253.58 Mbps, mMTC Total Rate: 60.79 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         13  30.0/30 MHz       100.00%
mMTC           5  8.3/10 MHz        83.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 2, Bandwidth: 1.0 MHz, Rate: 4.12 Mbps, Latency: 3.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 13: 5.0 → 4.0 MHz, Rate: 47.57 → 38.06 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        4   |         38.06 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     2 |        1   |          4.12 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        3   |         42.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       15   |        198.19 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       15   |        169.98 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3764, completion~709, total~4473

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need emergency response coordination during a disaste"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need emergency response coordination during a disaster", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~816, total~2188
[Token] Slice_Type_Determination: prompt~2649, completion~119, total~2768
[Token] Beamforming_Bandwidth: prompt~216, completion~4, total~220
[Token] Allocate_Resources: prompt~2936, completion~429, total~3365

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-16 11:58:28
Total Users: 24
Average Resource Utilization: 98.69%
eMBB Total Rate: 681.68 Mbps, URLLC Total Rate: 252.61 Mbps, mMTC Total Rate: 60.79 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         14  30.0/30 MHz       100.00%
mMTC           5  8.3/10 MHz        83.00%

New User Allocation:
User 27 → URLLC Slice
CQI: 14, Bandwidth: 1.0 MHz, Rate: 14.17 Mbps, Latency: 1.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 4: 3.0 → 2.0 MHz, Rate: 45.41 → 30.27 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        4   |         38.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |    14 |        1   |         14.17 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        3   |         42.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       15   |        198.19 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       15   |        169.98 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3676, completion~758, total~4434

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I want to use augmented reality navigation"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use augmented reality navigation", "embb"),)
[Token] Intent_Analysis: prompt~1368, completion~755, total~2123
[Token] Slice_Type_Determination: prompt~2589, completion~90, total~2679
[Token] Beamforming_Bandwidth: prompt~214, completion~4, total~218
[Token] Allocate_Resources: prompt~2872, completion~572, total~3444
[Token] Failure_Evaluation: prompt~3535, completion~419, total~3954

----------------------------------------
ALLOCATION FAILED FOR USER 28
----------------------------------------
Request: I want to use augmented reality navigation
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~879, total~2251
[Token] Slice_Type_Determination: prompt~2725, completion~84, total~2809
[Token] Beamforming_Bandwidth: prompt~216, completion~4, total~220
[Token] Allocate_Resources: prompt~2973, completion~468, total~3441

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-16 12:00:41
Total Users: 25
Average Resource Utilization: 98.69%
eMBB Total Rate: 681.68 Mbps, URLLC Total Rate: 243.20 Mbps, mMTC Total Rate: 60.79 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         15  30.0/30 MHz       100.00%
mMTC           5  8.3/10 MHz        83.00%

New User Allocation:
User 29 → URLLC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 4.76 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 7: 3.0 → 2.0 MHz, Rate: 42.51 → 28.34 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        4   |         38.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | URLLC   |     3 |        1   |          4.76 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       15   |        198.19 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       15   |        169.98 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3757, completion~724, total~4481

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 1
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart parking sensor needs to report if the spot is free", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~855, total~2233
[Token] Slice_Type_Determination: prompt~2689, completion~97, total~2786
[Token] Beamforming_Bandwidth: prompt~238, completion~2, total~240
[Token] Allocate_Resources: prompt~2938, completion~562, total~3500

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-16 12:01:56
Total Users: 26
Average Resource Utilization: 99.38%
eMBB Total Rate: 681.68 Mbps, URLLC Total Rate: 243.20 Mbps, mMTC Total Rate: 63.98 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         15  30.0/30 MHz               100.00%
mMTC           6  9.200000000000001/10 MHz  92.00%

New User Allocation:
User 30 → mMTC Slice
CQI: 1, Bandwidth: 0.9 MHz, Rate: 3.19 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        4   |         38.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       15   |        198.19 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       15   |        169.98 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        1.7 |         25.73 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | mMTC    |     1 |        0.9 |          3.19 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        1.9 |          9.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3742, completion~604, total~4346

Detailed Slice Utilization Values:
eMBB utils: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.22, 22.22, 44.44, 66.67, 66.67, 66.67, 66.67, 83.33, 83.33, 83.33, 83.33, 83.33, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [3.33, 20.0, 20.0, 33.33, 33.33, 46.67, 63.33, 63.33, 66.67, 66.67, 66.67, 76.67, 93.33, 93.33, 93.33, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
mMTC utils: [0.0, 0.0, 17.0, 17.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 55.0, 55.0, 55.0, 55.0, 74.0, 83.0, 83.0, 83.0, 83.0, 83.0, 83.0, 83.0, 92.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | URLLC   | URLLC          | Yes            |     2 |        1   |          4.12 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | URLLC   | URLLC          | Yes            |     3 |        5   |         23.82 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | mMTC    | mMTC           | Yes            |    15 |        1.7 |         25.73 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | URLLC   | URLLC          | Yes            |    15 |        4   |         60.54 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | mMTC    | mMTC           | Yes            |     3 |        1.9 |          9.05 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | URLLC   | URLLC          | Yes            |     4 |        4   |         21.82 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |    14 |        5   |         70.85 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | URLLC   | eMBB           | No             |     6 |        1   |          6.97 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             30 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | eMBB    | eMBB           | Yes            |     3 |       20   |         95.29 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     3 |        3   |         14.29 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | eMBB           | No             |     9 |        5   |         47.57 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | mMTC    | mMTC           | Yes            |     6 |        1.9 |         13.25 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |    13 |       15   |        198.19 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |     9 |        2   |         19.03 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | URLLC   | URLLC          | Yes            |    15 |        1   |         15.14 |              3 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | mMTC    | mMTC           | Yes            |     3 |        1.9 |          9.05 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | mMTC    | mMTC           | Yes            |     2 |        0.9 |          3.71 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    11 |       15   |        169.98 |             80 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Failed   | eMBB    | eMBB           | Yes            |     6 |       20   |        139.46 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | URLLC   | URLLC          | Yes            |     2 |        1   |          4.12 |              3 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Failed   | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Failed   | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | URLLC   | URLLC          | Yes            |    14 |        1   |         14.17 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | URLLC          | Yes            |     2 |        1   |          4.12 |              3 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | URLLC   | URLLC          | Yes            |    14 |        1   |         14.17 |              1 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Failed   | eMBB    | eMBB           | Yes            |     6 |       20   |        139.46 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | URLLC   | URLLC          | Yes            |     3 |        1   |          4.76 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Success  | mMTC    | mMTC           | Yes            |     1 |        0.9 |          3.19 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 26/30 (86.7%)

Intent Understanding Evaluation:
Correctly identified intents: 28/30
Intent understanding rate: 93.3%

Workload Balancing Statistics:
Users with workload balancing: 0/30
Workload balancing rate: 0.0%

Slice Utilization Statistics:
Average eMBB utilization: 56.62%
Average URLLC utilization: 74.61%
Average mMTC utilization: 50.96%

Weighted Average Utilization: 60.34%

Transmission Rate Statistics:
Final eMBB total rate: 681.68 Mbps
Final URLLC total rate: 243.20 Mbps
Final mMTC total rate: 63.98 Mbps

Resource Utilization:
Average resource utilization: 99.38%

Results exported to F:\code\wirelessagent\run_results\batch_run\fkb\kimi-k2.5\network_slicing_results_TJU_south_kimi-k2.5.csv

✓ TJU_south 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\fkb\kimi-k2.5\network_slicing_results_TJU_south_kimi-k2.5.csv