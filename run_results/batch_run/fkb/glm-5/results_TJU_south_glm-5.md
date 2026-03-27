============================================================
场景 2/5: TJU_south
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_south.csv
输出: F:\code\wirelessagent\run_results\batch_run\fkb\glm-5\network_slicing_results_TJU_south_glm-5.csv
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
[Token] Intent_Analysis: prompt~1372, completion~1326, total~2698
[Token] Slice_Type_Determination: prompt~2480, completion~340, total~2820
[Token] Beamforming_Bandwidth: prompt~214, completion~1419, total~1633
[Token] Allocate_Resources: prompt~2675, completion~1199, total~3874

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-18 11:47:07
Total Users: 1
Average Resource Utilization: 3.85%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 20.62 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          1  5.0/30 MHz        16.67%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → URLLC Slice
CQI: 2, Bandwidth: 5.0 MHz, Rate: 20.62 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          5 |         20.62 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3377, completion~983, total~4360

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need my autonomous vehicle to communicate in real time"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need my autonomous vehicle to communicate in real time", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~1384, total~2758
[Token] Slice_Type_Determination: prompt~2438, completion~346, total~2784
[Token] Beamforming_Bandwidth: prompt~217, completion~1086, total~1303
[Token] Allocate_Resources: prompt~2641, completion~962, total~3603

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-18 11:49:12
Total Users: 2
Average Resource Utilization: 7.69%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 44.44 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          2  10.0/30 MHz       33.33%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → URLLC Slice
CQI: 3, Bandwidth: 5.0 MHz, Rate: 23.82 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          5 |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |          5 |         23.82 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3360, completion~993, total~4353

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart agriculture sensor needs to report soil temperature", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~885, total~2257
[Token] Slice_Type_Determination: prompt~2334, completion~342, total~2676
[Token] Beamforming_Bandwidth: prompt~213, completion~687, total~900
[Token] Allocate_Resources: prompt~2572, completion~1599, total~4171

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-18 11:51:26
Total Users: 3
Average Resource Utilization: 8.38%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 44.44 Mbps, mMTC Total Rate: 13.62 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          2  10.0/30 MHz       33.33%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 3 → mMTC Slice
CQI: 15, Bandwidth: 0.9 MHz, Rate: 13.62 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3414, completion~976, total~4390

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to participate in an online multiplayer game"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to participate in an online multiplayer game", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~917, total~2289
[Token] Slice_Type_Determination: prompt~2305, completion~231, total~2536
LLM recommended eMBB but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~2478, completion~862, total~3340
[Token] Beamforming_Bandwidth: prompt~216, completion~2076, total~2292
[Token] Allocate_Resources: prompt~2994, completion~627, total~3621

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-18 11:54:08
Total Users: 4
Average Resource Utilization: 10.69%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 89.85 Mbps, mMTC Total Rate: 13.62 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          3  13.0/30 MHz       43.33%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 4 → URLLC Slice
CQI: 15, Bandwidth: 3.0 MHz, Rate: 45.41 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3645, completion~833, total~4478

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a fleet of delivery drones needs to send low-rate telemetry data", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~1791, total~3169
[Token] Slice_Type_Determination: prompt~2566, completion~360, total~2926
[Token] Beamforming_Bandwidth: prompt~218, completion~2090, total~2308
[Token] Allocate_Resources: prompt~2816, completion~1888, total~4704

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-18 12:00:32
Total Users: 5
Average Resource Utilization: 12.92%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 89.85 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          3  13.0/30 MHz       43.33%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 5 → mMTC Slice
CQI: 3, Bandwidth: 2.9 MHz, Rate: 13.82 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3571, completion~900, total~4471

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I want to play competitive mobile games with ultra-low latency"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to play competitive mobile games with ultra-low latency", "urllc"),)
[Token] Intent_Analysis: prompt~1376, completion~978, total~2354
[Token] Slice_Type_Determination: prompt~2279, completion~243, total~2522
[Token] Beamforming_Bandwidth: prompt~218, completion~1165, total~1383
[Token] Allocate_Resources: prompt~2489, completion~1030, total~3519

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-18 12:02:31
Total Users: 6
Average Resource Utilization: 16.77%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 117.13 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          4  18.0/30 MHz       60.00%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 6 → URLLC Slice
CQI: 4, Bandwidth: 5.0 MHz, Rate: 27.28 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3255, completion~1142, total~4397

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to participate in an online multiplayer game"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to participate in an online multiplayer game", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~1592, total~2964
[Token] Slice_Type_Determination: prompt~2309, completion~1821, total~4130
[Token] Beamforming_Bandwidth: prompt~216, completion~4013, total~4229
[Token] Allocate_Resources: prompt~2725, completion~978, total~3703

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-18 12:07:06
Total Users: 7
Average Resource Utilization: 19.08%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 159.64 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          5  21.0/30 MHz       70.00%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 14, Bandwidth: 3.0 MHz, Rate: 42.51 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        3   |         42.51 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3432, completion~1087, total~4519

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I want to browse websites and check email"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~1926, total~3296
[Token] Slice_Type_Determination: prompt~2469, completion~365, total~2834
[Token] Beamforming_Bandwidth: prompt~213, completion~1367, total~1580
[Token] Allocate_Resources: prompt~2736, completion~1625, total~4361

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-18 12:10:17
Total Users: 8
Average Resource Utilization: 34.46%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 159.64 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          5  21.0/30 MHz       70.00%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 8 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        3   |         42.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3450, completion~857, total~4307

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to monitor my home security cameras remotely", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~1487, total~2859
[Token] Slice_Type_Determination: prompt~2387, completion~216, total~2603
LLM recommended eMBB but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~2524, completion~751, total~3275
[Token] Beamforming_Bandwidth: prompt~216, completion~3244, total~3460
[Token] Allocate_Resources: prompt~2987, completion~809, total~3796

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-18 12:13:55
Total Users: 9
Average Resource Utilization: 38.31%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 194.51 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          6  26.0/30 MHz       86.67%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 9 → URLLC Slice
CQI: 6, Bandwidth: 5.0 MHz, Rate: 34.87 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        3   |         42.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3655, completion~894, total~4549

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to make a high-quality voice call"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to make a high-quality voice call", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~2579, total~3951
[Token] Slice_Type_Determination: prompt~2489, completion~1151, total~3640
LLM recommended URLLC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~2639, completion~791, total~3430
[Token] Beamforming_Bandwidth: prompt~216, completion~2537, total~2753
[Token] Allocate_Resources: prompt~3136, completion~954, total~4090

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-18 12:17:56
Total Users: 10
Average Resource Utilization: 53.69%
eMBB Total Rate: 218.22 Mbps, URLLC Total Rate: 194.51 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          6  26.0/30 MHz       86.67%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 10 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 30.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        3   |         42.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3940, completion~1046, total~4986

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to send text messages and use messaging apps"
CQI: 3
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to send text messages and use messaging apps", "embb"),)
[Token] Intent_Analysis: prompt~1374, completion~2950, total~4324
[Token] Slice_Type_Determination: prompt~2508, completion~858, total~3366
[Token] Beamforming_Bandwidth: prompt~217, completion~2031, total~2248
[Token] Allocate_Resources: prompt~2990, completion~1579, total~4569

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-18 12:21:48
Total Users: 11
Average Resource Utilization: 69.08%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 194.51 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          6  26.0/30 MHz       86.67%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 11 → eMBB Slice
CQI: 3, Bandwidth: 20.0 MHz, Rate: 95.29 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        3   |         42.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3818, completion~1139, total~4957

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I want to play competitive mobile games with ultra-low latency"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to play competitive mobile games with ultra-low latency", "urllc"),)
[Token] Intent_Analysis: prompt~1376, completion~1219, total~2595
[Token] Slice_Type_Determination: prompt~2420, completion~534, total~2954
[Token] Beamforming_Bandwidth: prompt~218, completion~2548, total~2766
[Token] Allocate_Resources: prompt~2621, completion~1026, total~3647

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-18 12:24:54
Total Users: 12
Average Resource Utilization: 72.15%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 213.57 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          7  30.0/30 MHz       100.00%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 3, Bandwidth: 4.0 MHz, Rate: 19.06 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        3   |         42.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3427, completion~1391, total~4818

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need real-time traffic updates for navigation"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need real-time traffic updates for navigation", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~3966, total~5336
[Token] Slice_Type_Determination: prompt~2600, completion~1692, total~4292
LLM recommended mMTC but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~2996, completion~618, total~3614
[Token] Beamforming_Bandwidth: prompt~215, completion~725, total~940
[Token] Allocate_Resources: prompt~3542, completion~1242, total~4784

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-18 12:29:14
Total Users: 13
Average Resource Utilization: 72.15%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 207.94 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          8  30.0/30 MHz       100.00%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 4: 3.0 → 2.0 MHz, Rate: 45.41 → 30.27 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        3   |         42.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4359, completion~827, total~5186

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my wearable device needs to upload health data periodically", "mmtc"),, ("my wearable device needs to upload health data periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~1347, total~2719
[Token] Slice_Type_Determination: prompt~2352, completion~192, total~2544
[Token] Beamforming_Bandwidth: prompt~215, completion~1231, total~1446
[Token] Allocate_Resources: prompt~2586, completion~1647, total~4233

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-18 12:31:45
Total Users: 14
Average Resource Utilization: 72.85%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 207.94 Mbps, mMTC Total Rate: 33.72 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          8  30.0/30 MHz       100.00%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 14 → mMTC Slice
CQI: 6, Bandwidth: 0.9 MHz, Rate: 6.28 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        3   |         42.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3258, completion~948, total~4206

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 13
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[Token] Intent_Analysis: prompt~1376, completion~1304, total~2680
[Token] Slice_Type_Determination: prompt~2495, completion~414, total~2909
[Token] Beamforming_Bandwidth: prompt~218, completion~1100, total~1318
[Token] Allocate_Resources: prompt~2692, completion~916, total~3608

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-18 12:34:13
Total Users: 15
Average Resource Utilization: 88.23%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 207.94 Mbps, mMTC Total Rate: 33.72 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          8  30.0/30 MHz       100.00%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 13, Bandwidth: 20.0 MHz, Rate: 264.25 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        3   |         42.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3308, completion~1013, total~4321

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need instant facial recognition for public security threats"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need instant facial recognition for public security threats", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~1466, total~2838
[Token] Slice_Type_Determination: prompt~2534, completion~280, total~2814
[Token] Beamforming_Bandwidth: prompt~216, completion~477, total~693
[Token] Allocate_Resources: prompt~2763, completion~919, total~3682

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-18 12:36:13
Total Users: 16
Average Resource Utilization: 88.23%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 203.28 Mbps, mMTC Total Rate: 33.72 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          9  30.0/30 MHz       100.00%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 7: 3.0 → 2.0 MHz, Rate: 42.51 → 28.34 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3473, completion~858, total~4331

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to monitor and control critical manufacturing processes in real-time"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to monitor and control critical manufacturing processes in real-time", "urllc"),)
[Token] Intent_Analysis: prompt~1378, completion~1014, total~2392
[Token] Slice_Type_Determination: prompt~2339, completion~263, total~2602
[Token] Beamforming_Bandwidth: prompt~219, completion~423, total~642
[Token] Allocate_Resources: prompt~2648, completion~1273, total~3921

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-18 12:38:05
Total Users: 17
Average Resource Utilization: 88.23%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 211.44 Mbps, mMTC Total Rate: 33.72 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC         10  30.0/30 MHz       100.00%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 17 → URLLC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 15.14 Mbps, Latency: 3.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 9: 5.0 → 4.0 MHz, Rate: 34.87 → 27.89 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        4   |         27.89 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3502, completion~1059, total~4561

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need to check the status of my smart home sensors"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of my smart home sensors", "mmtc"),, ("i need to check the status of my smart home sensors", "mmtc"),)
[Token] Intent_Analysis: prompt~1376, completion~1434, total~2810
[Token] Slice_Type_Determination: prompt~2465, completion~178, total~2643
[Token] Beamforming_Bandwidth: prompt~217, completion~1011, total~1228
[Token] Allocate_Resources: prompt~2694, completion~1351, total~4045

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-18 12:40:46
Total Users: 18
Average Resource Utilization: 88.92%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 211.44 Mbps, mMTC Total Rate: 38.01 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           4  80.0/90 MHz                88.89%
URLLC         10  30.0/30 MHz                100.00%
mMTC           4  5.6000000000000005/10 MHz  56.00%

New User Allocation:
User 18 → mMTC Slice
CQI: 3, Bandwidth: 0.9 MHz, Rate: 4.29 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        4   |         27.89 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3327, completion~984, total~4311

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 2
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("industrial equipment monitoring sensor data", "mmtc"))
[Token] Intent_Analysis: prompt~1364, completion~1209, total~2573
[Token] Slice_Type_Determination: prompt~2510, completion~215, total~2725
[Token] Beamforming_Bandwidth: prompt~216, completion~1777, total~1993
[Token] Allocate_Resources: prompt~2762, completion~1262, total~4024

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-18 12:43:39
Total Users: 19
Average Resource Utilization: 89.62%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 211.44 Mbps, mMTC Total Rate: 41.72 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           4  80.0/90 MHz               88.89%
URLLC         10  30.0/30 MHz               100.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 19 → mMTC Slice
CQI: 2, Bandwidth: 0.9 MHz, Rate: 3.71 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        4   |         27.89 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3438, completion~1021, total~4459

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I want to download a big game file"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to download a big game file", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~912, total~2282
[Token] Slice_Type_Determination: prompt~2265, completion~340, total~2605
[Token] Beamforming_Bandwidth: prompt~215, completion~690, total~905
[Token] Allocate_Resources: prompt~2493, completion~910, total~3403

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-18 12:45:35
Total Users: 20
Average Resource Utilization: 97.31%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 211.44 Mbps, mMTC Total Rate: 41.72 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         10  30.0/30 MHz               100.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 11, Bandwidth: 10.0 MHz, Rate: 113.32 Mbps, Latency: 80.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        4   |         27.89 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             80 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3192, completion~1013, total~4205

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to update my social media status"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~3227, total~4597
[Token] Slice_Type_Determination: prompt~2403, completion~418, total~2821
[Token] Beamforming_Bandwidth: prompt~215, completion~1703, total~1918
[Token] Allocate_Resources: prompt~2687, completion~1651, total~4338
[Token] Failure_Evaluation: prompt~3179, completion~838, total~4017

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
[Token] Intent_Analysis: prompt~1372, completion~1133, total~2505
[Token] Slice_Type_Determination: prompt~2343, completion~756, total~3099
[Token] Beamforming_Bandwidth: prompt~216, completion~831, total~1047
[Token] Allocate_Resources: prompt~2906, completion~1491, total~4397

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-18 12:51:43
Total Users: 21
Average Resource Utilization: 97.31%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 200.43 Mbps, mMTC Total Rate: 41.72 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         11  30.0/30 MHz               100.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 22 → URLLC Slice
CQI: 2, Bandwidth: 1.0 MHz, Rate: 4.12 Mbps, Latency: 3.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 4: 2.0 → 1.0 MHz, Rate: 30.27 → 15.14 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        1   |          4.12 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        4   |         27.89 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3672, completion~789, total~4461

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I want to stream a webinar with interactive features"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to stream a webinar with interactive features", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~1617, total~2989
[Token] Slice_Type_Determination: prompt~2580, completion~273, total~2853
[Token] Beamforming_Bandwidth: prompt~216, completion~357, total~573
[Token] Allocate_Resources: prompt~2881, completion~1906, total~4787
[Token] Failure_Evaluation: prompt~3397, completion~561, total~3958

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
[Token] Intent_Analysis: prompt~1376, completion~1105, total~2481
[Token] Slice_Type_Determination: prompt~2322, completion~223, total~2545
[Token] Beamforming_Bandwidth: prompt~218, completion~4026, total~4244
[Token] Allocate_Resources: prompt~2601, completion~943, total~3544
[Token] Failure_Evaluation: prompt~3096, completion~477, total~3573

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
[Token] Intent_Analysis: prompt~1372, completion~1079, total~2451
[Token] Slice_Type_Determination: prompt~2311, completion~298, total~2609
[Token] Beamforming_Bandwidth: prompt~216, completion~2868, total~3084
[Token] Allocate_Resources: prompt~2601, completion~1130, total~3731

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-18 13:00:59
Total Users: 22
Average Resource Utilization: 97.31%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 200.43 Mbps, mMTC Total Rate: 41.72 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         12  30.0/30 MHz               100.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 25 → URLLC Slice
CQI: 14, Bandwidth: 1.0 MHz, Rate: 14.17 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 7: 2.0 → 1.0 MHz, Rate: 28.34 → 14.17 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        4   |         27.89 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3374, completion~1174, total~4548

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to control a robotic arm in real time"
CQI: 2
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control a robotic arm in real time", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~1271, total~2645
[Token] Slice_Type_Determination: prompt~2478, completion~248, total~2726
[Token] Beamforming_Bandwidth: prompt~217, completion~703, total~920
[Token] Allocate_Resources: prompt~2714, completion~925, total~3639

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-18 13:03:02
Total Users: 23
Average Resource Utilization: 97.31%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 197.58 Mbps, mMTC Total Rate: 41.72 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         13  30.0/30 MHz               100.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 2, Bandwidth: 1.0 MHz, Rate: 4.12 Mbps, Latency: 3.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 9: 4.0 → 3.0 MHz, Rate: 27.89 → 20.92 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
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
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        3   |         20.92 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3504, completion~1120, total~4624

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need emergency response coordination during a disaste"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need emergency response coordination during a disaster", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~1311, total~2683
[Token] Slice_Type_Determination: prompt~2431, completion~247, total~2678
[Token] Beamforming_Bandwidth: prompt~216, completion~4065, total~4281
[Token] Allocate_Resources: prompt~2676, completion~1254, total~3930

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-18 13:07:16
Total Users: 24
Average Resource Utilization: 97.31%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 206.29 Mbps, mMTC Total Rate: 41.72 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         14  30.0/30 MHz               100.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 27 → URLLC Slice
CQI: 14, Bandwidth: 1.0 MHz, Rate: 14.17 Mbps, Latency: 1.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 6: 5.0 → 4.0 MHz, Rate: 27.28 → 21.82 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
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
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3453, completion~946, total~4399

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I want to use augmented reality navigation"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use augmented reality navigation", "embb"),)
[Token] Intent_Analysis: prompt~1368, completion~1157, total~2525
[Token] Slice_Type_Determination: prompt~2327, completion~453, total~2780
[Token] Beamforming_Bandwidth: prompt~214, completion~371, total~585
[Token] Allocate_Resources: prompt~2643, completion~1230, total~3873
[Token] Failure_Evaluation: prompt~3180, completion~644, total~3824

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
[Token] Intent_Analysis: prompt~1372, completion~1475, total~2847
[Token] Slice_Type_Determination: prompt~2577, completion~334, total~2911
[Token] Beamforming_Bandwidth: prompt~216, completion~655, total~871
[Token] Allocate_Resources: prompt~2841, completion~1109, total~3950

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-18 13:11:35
Total Users: 25
Average Resource Utilization: 97.31%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 206.29 Mbps, mMTC Total Rate: 41.72 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         15  30.0/30 MHz               100.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 29 → URLLC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 4.76 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 2: 5.0 → 4.0 MHz, Rate: 23.82 → 19.06 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        4   |         19.06 |              5 | ADJUSTED |
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
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3619, completion~955, total~4574

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 1
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart parking sensor needs to report if the spot is free", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~1153, total~2531
[Token] Slice_Type_Determination: prompt~2355, completion~450, total~2805
[Token] Beamforming_Bandwidth: prompt~222, completion~616, total~838
[Token] Allocate_Resources: prompt~2624, completion~1491, total~4115

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-18 13:13:51
Total Users: 26
Average Resource Utilization: 98.0%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 206.29 Mbps, mMTC Total Rate: 44.91 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         15  30.0/30 MHz               100.00%
mMTC           6  7.400000000000001/10 MHz  74.00%

New User Allocation:
User 30 → mMTC Slice
CQI: 1, Bandwidth: 0.9 MHz, Rate: 3.19 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        4   |         19.06 |              5 |          |
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
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | mMTC    |     1 |        0.9 |          3.19 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3296, completion~656, total~3952

Detailed Slice Utilization Values:
eMBB utils: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.22, 22.22, 44.44, 66.67, 66.67, 66.67, 66.67, 88.89, 88.89, 88.89, 88.89, 88.89, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [16.67, 33.33, 33.33, 43.33, 43.33, 60.0, 70.0, 70.0, 86.67, 86.67, 86.67, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
mMTC utils: [0.0, 0.0, 9.0, 9.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 47.0, 47.0, 47.0, 47.0, 56.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 74.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | URLLC   | URLLC          | Yes            |     2 |        5   |         20.62 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | URLLC   | URLLC          | Yes            |     3 |        5   |         23.82 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | mMTC    | mMTC           | Yes            |    15 |        0.9 |         13.62 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | URLLC   | URLLC          | Yes            |    15 |        3   |         45.41 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | mMTC    | mMTC           | Yes            |     3 |        2.9 |         13.82 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | URLLC   | URLLC          | Yes            |     4 |        5   |         27.28 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |    14 |        3   |         42.51 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | URLLC   | eMBB           | No             |     6 |        5   |         34.87 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             30 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | eMBB    | eMBB           | Yes            |     3 |       20   |         95.29 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     3 |        4   |         19.06 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | eMBB           | No             |     9 |        1   |          9.51 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | mMTC    | mMTC           | Yes            |     6 |        0.9 |          6.28 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |    13 |       20   |        264.25 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |     9 |        1   |          9.51 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | URLLC   | URLLC          | Yes            |    15 |        1   |         15.14 |              3 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | mMTC    | mMTC           | Yes            |     3 |        0.9 |          4.29 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | mMTC    | mMTC           | Yes            |     2 |        0.9 |          3.71 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    11 |       10   |        113.32 |             80 | No         |
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
Average eMBB utilization: 57.69%
Average URLLC utilization: 81.92%
Average mMTC utilization: 43.58%

Weighted Average Utilization: 62.20%

Transmission Rate Statistics:
Final eMBB total rate: 691.08 Mbps
Final URLLC total rate: 206.29 Mbps
Final mMTC total rate: 44.91 Mbps

Resource Utilization:
Average resource utilization: 98.00%

Results exported to F:\code\wirelessagent\run_results\batch_run\fkb\glm-5\network_slicing_results_TJU_south_glm-5.csv

✓ TJU_south 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\fkb\glm-5\network_slicing_results_TJU_south_glm-5.csv