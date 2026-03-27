============================================================
场景 2/5: TJU_east
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_east.csv
输出: F:\code\wirelessagent\run_results\batch_run\fkb\minimax-m2.1\network_slicing_results_TJU_east_minimax-m2.1.csv
------------------------------------------------------------
Starting network slice management system with CSV-based user testing...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I need to check weather forecasts"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to check weather forecasts", "embb"),)
[Token] Intent_Analysis: prompt~1366, completion~876, total~2242
[Token] Slice_Type_Determination: prompt~2693, completion~599, total~3292
[Token] Beamforming_Bandwidth: prompt~211, completion~448, total~659
[Token] Allocate_Resources: prompt~3440, completion~806, total~4246

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-15 20:45:01
Total Users: 1
Average Resource Utilization: 15.38%
eMBB Total Rate: 302.70 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → eMBB Slice
CQI: 15, Bandwidth: 20.0 MHz, Rate: 302.70 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         20 |         302.7 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4489, completion~673, total~5162

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need to stream 8K video content"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to stream 8k video content", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~880, total~2252
[Token] Slice_Type_Determination: prompt~2722, completion~226, total~2948
[Token] Beamforming_Bandwidth: prompt~216, completion~1498, total~1714
[Token] Allocate_Resources: prompt~3095, completion~769, total~3864

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-15 20:48:01
Total Users: 2
Average Resource Utilization: 30.77%
eMBB Total Rate: 411.81 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         20 |        302.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4106, completion~618, total~4724

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I want to listen to low-quality audio streaming"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to listen to low-quality audio streaming", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~960, total~2332
[Token] Slice_Type_Determination: prompt~2799, completion~342, total~3141
[Token] Beamforming_Bandwidth: prompt~216, completion~299, total~515
[Token] Allocate_Resources: prompt~3291, completion~830, total~4121

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-15 20:49:01
Total Users: 3
Average Resource Utilization: 46.15%
eMBB Total Rate: 714.51 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 3 → eMBB Slice
CQI: 15, Bandwidth: 20.0 MHz, Rate: 302.70 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         20 |        302.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4362, completion~831, total~5193

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I want to browse websites and check email"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~887, total~2257
[Token] Slice_Type_Determination: prompt~2720, completion~240, total~2960
[Token] Beamforming_Bandwidth: prompt~215, completion~322, total~537
[Token] Allocate_Resources: prompt~3106, completion~731, total~3837

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-15 20:50:22
Total Users: 4
Average Resource Utilization: 61.54%
eMBB Total Rate: 904.81 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 4 → eMBB Slice
CQI: 9, Bandwidth: 20.0 MHz, Rate: 190.30 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         20 |        302.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        190.3  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4075, completion~650, total~4725

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to monitor my home security cameras remotely", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~870, total~2242
[Token] Slice_Type_Determination: prompt~2714, completion~575, total~3289
LLM recommended eMBB but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~3367, completion~271, total~3638
[Token] Beamforming_Bandwidth: prompt~214, completion~418, total~632
[Token] Allocate_Resources: prompt~3763, completion~775, total~4538

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-15 20:52:21
Total Users: 5
Average Resource Utilization: 62.31%
eMBB Total Rate: 904.81 Mbps, URLLC Total Rate: 11.33 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          1  1.0/30 MHz        3.33%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 5 → URLLC Slice
CQI: 11, Bandwidth: 1.0 MHz, Rate: 11.33 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |          1 |         11.33 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         20 |        302.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4848, completion~864, total~5712

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I need to participate in a video conference meeting"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to participate in a video conference meeting", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~1057, total~2429
[Token] Slice_Type_Determination: prompt~2908, completion~236, total~3144
[Token] Beamforming_Bandwidth: prompt~216, completion~230, total~446
[Token] Allocate_Resources: prompt~3343, completion~777, total~4120

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-15 20:53:50
Total Users: 6
Average Resource Utilization: 70.0%
eMBB Total Rate: 998.77 Mbps, URLLC Total Rate: 11.33 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          1  1.0/30 MHz        3.33%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 6 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 245.31 Mbps, Latency: 50.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 10.0 MHz
  User 1: 20.0 → 10.0 MHz, Rate: 302.70 → 151.35 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |          1 |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        151.35 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |        245.31 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4441, completion~583, total~5024

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to monitor water level in a reservoir"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor water level in a reservoir", "mmtc"),, ("i need to monitor water level in a reservoir periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~1180, total~2552
[Token] Slice_Type_Determination: prompt~3017, completion~383, total~3400
[Token] Beamforming_Bandwidth: prompt~213, completion~619, total~832
[Token] Allocate_Resources: prompt~3557, completion~804, total~4361

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-15 20:55:14
Total Users: 7
Average Resource Utilization: 72.23%
eMBB Total Rate: 998.77 Mbps, URLLC Total Rate: 11.33 Mbps, mMTC Total Rate: 20.22 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          1  1.0/30 MHz        3.33%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 7 → mMTC Slice
CQI: 6, Bandwidth: 2.9 MHz, Rate: 20.22 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       20   |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |       20   |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4604, completion~766, total~5370

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~862, total~2232
[Token] Slice_Type_Determination: prompt~2725, completion~228, total~2953
[Token] Beamforming_Bandwidth: prompt~215, completion~467, total~682
[Token] Allocate_Resources: prompt~3078, completion~785, total~3863

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-15 20:56:55
Total Users: 8
Average Resource Utilization: 73.0%
eMBB Total Rate: 998.77 Mbps, URLLC Total Rate: 20.84 Mbps, mMTC Total Rate: 20.22 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          2  2.0/30 MHz        6.67%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       20   |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |       20   |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4126, completion~692, total~4818

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to update my social media status"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~976, total~2346
[Token] Slice_Type_Determination: prompt~2813, completion~308, total~3121
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~3187, completion~184, total~3371
[Token] Beamforming_Bandwidth: prompt~215, completion~491, total~706
[Token] Allocate_Resources: prompt~3595, completion~710, total~4305

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-15 20:58:05
Total Users: 9
Average Resource Utilization: 73.0%
eMBB Total Rate: 961.47 Mbps, URLLC Total Rate: 20.84 Mbps, mMTC Total Rate: 20.22 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          2  2.0/30 MHz        6.67%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 9 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 245.31 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 20.0 MHz
  User 3: 20.0 → 7.0 MHz, Rate: 302.70 → 105.95 Mbps, User 6: 20.0 → 13.0 MHz, Rate: 245.31 → 159.45 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       20   |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |       13   |        159.45 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |       20   |        245.31 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4723, completion~698, total~5421

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I want to update my social media status"
CQI: 13
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~929, total~2299
[Token] Slice_Type_Determination: prompt~2757, completion~262, total~3019
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~3088, completion~258, total~3346
[Token] Beamforming_Bandwidth: prompt~215, completion~517, total~732
[Token] Allocate_Resources: prompt~3571, completion~769, total~4340

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-15 20:59:20
Total Users: 10
Average Resource Utilization: 73.0%
eMBB Total Rate: 1005.16 Mbps, URLLC Total Rate: 20.84 Mbps, mMTC Total Rate: 20.22 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC          2  2.0/30 MHz        6.67%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 10 → eMBB Slice
CQI: 13, Bandwidth: 20.0 MHz, Rate: 264.25 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 20.0 MHz
  User 9: 20.0 → 9.0 MHz, Rate: 245.31 → 110.39 Mbps, User 4: 20.0 → 11.0 MHz, Rate: 190.30 → 104.66 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |       20   |        264.25 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       20   |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |       13   |        159.45 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4760, completion~762, total~5522

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need vehicle-to-vehicle collision avoidance systems"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need vehicle-to-vehicle collision avoidance systems", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~938, total~2310
[Token] Slice_Type_Determination: prompt~2798, completion~277, total~3075
[Token] Beamforming_Bandwidth: prompt~216, completion~610, total~826
[Token] Allocate_Resources: prompt~3206, completion~843, total~4049

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-15 21:00:43
Total Users: 11
Average Resource Utilization: 73.77%
eMBB Total Rate: 1005.16 Mbps, URLLC Total Rate: 35.01 Mbps, mMTC Total Rate: 20.22 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC          3  3.0/30 MHz        10.00%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 14, Bandwidth: 1.0 MHz, Rate: 14.17 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       20   |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |       13   |        159.45 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4318, completion~731, total~5049

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need to sync my calendar and contacts"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to sync my calendar and contacts", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~967, total~2337
[Token] Slice_Type_Determination: prompt~2800, completion~268, total~3068
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~3136, completion~323, total~3459
[Token] Beamforming_Bandwidth: prompt~215, completion~289, total~504
[Token] Allocate_Resources: prompt~3743, completion~801, total~4544

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-15 21:02:10
Total Users: 12
Average Resource Utilization: 73.77%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 35.01 Mbps, mMTC Total Rate: 20.22 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          3  3.0/30 MHz        10.00%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 12 → eMBB Slice
CQI: 5, Bandwidth: 20.0 MHz, Rate: 123.87 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 4, Bandwidth freed: 20.0 MHz
  User 10: 20.0 → 8.0 MHz, Rate: 264.25 → 105.70 Mbps, User 6: 13.0 → 9.0 MHz, Rate: 159.45 → 110.39 Mbps, User 1: 10.0 → 7.0 MHz, Rate: 151.35 → 105.95 Mbps, User 2: 20.0 → 19.0 MHz, Rate: 109.11 → 103.65 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5057, completion~617, total~5674

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~743, total~2115
[Token] Slice_Type_Determination: prompt~2598, completion~285, total~2883
[Token] Beamforming_Bandwidth: prompt~216, completion~4346, total~4562
[Token] Allocate_Resources: prompt~3017, completion~799, total~3816

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-15 21:05:04
Total Users: 13
Average Resource Utilization: 76.85%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 56.83 Mbps, mMTC Total Rate: 20.22 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          4  7.0/30 MHz        23.33%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 4, Bandwidth: 4.0 MHz, Rate: 21.82 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        4   |         21.82 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4097, completion~964, total~5061

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I want to use holographic communication"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use holographic communication", "embb"),)
[Token] Intent_Analysis: prompt~1368, completion~741, total~2109
[Token] Slice_Type_Determination: prompt~2577, completion~297, total~2874
[Token] Beamforming_Bandwidth: prompt~214, completion~568, total~782
[Token] Allocate_Resources: prompt~3070, completion~779, total~3849
[Token] Failure_Evaluation: prompt~3955, completion~500, total~4455

----------------------------------------
ALLOCATION FAILED FOR USER 14
----------------------------------------
Request: I want to use holographic communication
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use maps for basic navigation"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use maps for basic navigation", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~1167, total~2537
[Token] Slice_Type_Determination: prompt~3021, completion~689, total~3710
[Token] Beamforming_Bandwidth: prompt~215, completion~351, total~566
[Token] Allocate_Resources: prompt~3906, completion~847, total~4753
[Token] Failure_Evaluation: prompt~4847, completion~756, total~5603

----------------------------------------
ALLOCATION FAILED FOR USER 15
----------------------------------------
Request: I need to use maps for basic navigation
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I want to use remote surgery equipment"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to use remote surgery equipment", "urllc"),)
[Token] Intent_Analysis: prompt~1368, completion~923, total~2291
[Token] Slice_Type_Determination: prompt~2773, completion~252, total~3025
[Token] Beamforming_Bandwidth: prompt~214, completion~418, total~632
[Token] Allocate_Resources: prompt~3149, completion~821, total~3970

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-15 21:08:27
Total Users: 14
Average Resource Utilization: 80.69%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 100.02 Mbps, mMTC Total Rate: 20.22 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          5  12.0/30 MHz       40.00%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 8, Bandwidth: 5.0 MHz, Rate: 43.19 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4233, completion~726, total~4959

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to participate in a video conference meeting"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to participate in a video conference meeting", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~823, total~2195
[Token] Slice_Type_Determination: prompt~2671, completion~265, total~2936
[Token] Beamforming_Bandwidth: prompt~216, completion~481, total~697
[Token] Allocate_Resources: prompt~3127, completion~1063, total~4190
[Token] Failure_Evaluation: prompt~4300, completion~508, total~4808

----------------------------------------
ALLOCATION FAILED FOR USER 17
----------------------------------------
Request: I need to participate in a video conference meeting
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control precision cnc machines with zero tolerance for delay", "urllc"),)
[Token] Intent_Analysis: prompt~1378, completion~737, total~2115
[Token] Slice_Type_Determination: prompt~2592, completion~190, total~2782
[Token] Beamforming_Bandwidth: prompt~219, completion~458, total~677
[Token] Allocate_Resources: prompt~2904, completion~709, total~3613

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-15 21:11:01
Total Users: 15
Average Resource Utilization: 81.46%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 105.48 Mbps, mMTC Total Rate: 20.22 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          6  13.0/30 MHz       43.33%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3872, completion~704, total~4576

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[Token] Intent_Analysis: prompt~1376, completion~1010, total~2386
[Token] Slice_Type_Determination: prompt~2850, completion~430, total~3280
[Token] Beamforming_Bandwidth: prompt~218, completion~474, total~692
[Token] Allocate_Resources: prompt~3471, completion~793, total~4264
[Token] Failure_Evaluation: prompt~4370, completion~660, total~5030

----------------------------------------
ALLOCATION FAILED FOR USER 19
----------------------------------------
Request: I need to use cloud-based AI services for image processing
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I need instant alerts for life-threatening patient conditions"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need instant alerts for life-threatening patient conditions", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~740, total~2112
[Token] Slice_Type_Determination: prompt~2579, completion~296, total~2875
[Token] Beamforming_Bandwidth: prompt~216, completion~1093, total~1309
[Token] Allocate_Resources: prompt~3000, completion~747, total~3747

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-15 21:14:12
Total Users: 16
Average Resource Utilization: 85.31%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 153.05 Mbps, mMTC Total Rate: 20.22 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          7  18.0/30 MHz       60.00%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 20 → URLLC Slice
CQI: 9, Bandwidth: 5.0 MHz, Rate: 47.57 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         47.57 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4008, completion~851, total~4859

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart agriculture sensor needs to report soil temperature", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~915, total~2287
[Token] Slice_Type_Determination: prompt~2741, completion~293, total~3034
[Token] Beamforming_Bandwidth: prompt~215, completion~456, total~671
[Token] Allocate_Resources: prompt~3184, completion~935, total~4119

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-15 21:15:52
Total Users: 17
Average Resource Utilization: 87.54%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 153.05 Mbps, mMTC Total Rate: 42.81 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          7  18.0/30 MHz       60.00%
mMTC           2  5.8/10 MHz        58.00%

New User Allocation:
User 21 → mMTC Slice
CQI: 7, Bandwidth: 2.9 MHz, Rate: 22.59 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |        2.9 |         22.59 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4361, completion~704, total~5065

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "I need to synchronize multiple robots on a factory floor"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to synchronize multiple robots on a factory floor", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~811, total~2185
[Token] Slice_Type_Determination: prompt~2648, completion~391, total~3039
[Token] Beamforming_Bandwidth: prompt~217, completion~405, total~622
[Token] Allocate_Resources: prompt~3165, completion~853, total~4018

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-15 21:16:56
Total Users: 18
Average Resource Utilization: 91.38%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 200.62 Mbps, mMTC Total Rate: 42.81 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          8  23.0/30 MHz       76.67%
mMTC           2  5.8/10 MHz        58.00%

New User Allocation:
User 22 → URLLC Slice
CQI: 9, Bandwidth: 5.0 MHz, Rate: 47.57 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |        2.9 |         22.59 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4286, completion~972, total~5258

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to control a robotic arm in real time"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control a robotic arm in real time", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~922, total~2296
[Token] Slice_Type_Determination: prompt~2783, completion~235, total~3018
[Token] Beamforming_Bandwidth: prompt~217, completion~929, total~1146
[Token] Allocate_Resources: prompt~3146, completion~769, total~3915

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-15 21:19:00
Total Users: 19
Average Resource Utilization: 95.23%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 227.90 Mbps, mMTC Total Rate: 42.81 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          9  28.0/30 MHz       93.33%
mMTC           2  5.8/10 MHz        58.00%

New User Allocation:
User 23 → URLLC Slice
CQI: 4, Bandwidth: 5.0 MHz, Rate: 27.28 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        5   |         27.28 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |        2.9 |         22.59 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4175, completion~836, total~5011

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to detect and isolate power grid faults instantly", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~779, total~2153
[Token] Slice_Type_Determination: prompt~2631, completion~318, total~2949
[Token] Beamforming_Bandwidth: prompt~217, completion~395, total~612
[Token] Allocate_Resources: prompt~3074, completion~786, total~3860

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-15 21:20:02
Total Users: 20
Average Resource Utilization: 96.0%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 234.09 Mbps, mMTC Total Rate: 42.81 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         10  29.0/30 MHz       96.67%
mMTC           2  5.8/10 MHz        58.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 5, Bandwidth: 1.0 MHz, Rate: 6.19 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        5   |         27.28 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |        2.9 |         22.59 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4132, completion~701, total~4833

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 2
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~861, total~2235
[Token] Slice_Type_Determination: prompt~2691, completion~322, total~3013
[Token] Beamforming_Bandwidth: prompt~216, completion~1782, total~1998
[Token] Allocate_Resources: prompt~3168, completion~1058, total~4226

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-15 21:22:00
Total Users: 21
Average Resource Utilization: 97.46%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 234.09 Mbps, mMTC Total Rate: 50.65 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           8  90.0/90 MHz               100.00%
URLLC         10  29.0/30 MHz               96.67%
mMTC           3  7.699999999999999/10 MHz  77.00%

New User Allocation:
User 25 → mMTC Slice
CQI: 2, Bandwidth: 1.9 MHz, Rate: 7.84 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        5   |         27.28 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |        2.9 |         22.59 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4477, completion~904, total~5381

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 1
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a fleet of delivery drones needs to send low-rate telemetry data", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~1187, total~2565
[Token] Slice_Type_Determination: prompt~3019, completion~378, total~3397
[Token] Beamforming_Bandwidth: prompt~238, completion~1281, total~1519
[Token] Allocate_Resources: prompt~3548, completion~891, total~4439

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-15 21:24:42
Total Users: 22
Average Resource Utilization: 98.15%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 234.09 Mbps, mMTC Total Rate: 53.84 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         10  29.0/30 MHz       96.67%
mMTC           4  8.6/10 MHz        86.00%

New User Allocation:
User 26 → mMTC Slice
CQI: 1, Bandwidth: 0.9 MHz, Rate: 3.19 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        5   |         27.28 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |        2.9 |         22.59 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        0.9 |          3.19 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4693, completion~901, total~5594

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "My smart meter needs to report its reading"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart meter needs to report its reading", "mmtc"),)
[Token] Intent_Analysis: prompt~1370, completion~960, total~2330
[Token] Slice_Type_Determination: prompt~2782, completion~238, total~3020
[Token] Beamforming_Bandwidth: prompt~234, completion~541, total~775
[Token] Allocate_Resources: prompt~3175, completion~896, total~4071

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-15 21:26:02
Total Users: 23
Average Resource Utilization: 98.85%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 234.09 Mbps, mMTC Total Rate: 60.85 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         10  29.0/30 MHz       96.67%
mMTC           5  9.5/10 MHz        95.00%

New User Allocation:
User 27 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        5   |         27.28 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |        2.9 |         22.59 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        0.9 |          3.19 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4326, completion~754, total~5080

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("industrial equipment monitoring sensor data", "mmtc"))
[Token] Intent_Analysis: prompt~1364, completion~983, total~2347
[Token] Slice_Type_Determination: prompt~2809, completion~470, total~3279
[Token] Beamforming_Bandwidth: prompt~211, completion~872, total~1083
[Token] Allocate_Resources: prompt~3487, completion~910, total~4397

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-15 21:28:15
Total Users: 24
Average Resource Utilization: 99.23%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 234.09 Mbps, mMTC Total Rate: 62.03 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         10  29.0/30 MHz       96.67%
mMTC           6  10.0/10 MHz       100.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 3, Bandwidth: 0.9 MHz, Rate: 4.29 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.4 MHz
  User 21: 2.9 → 2.5 MHz, Rate: 22.59 → 19.48 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        5   |         27.28 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |        2.5 |         19.48 |            500 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        0.9 |          3.19 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     3 |        0.9 |          4.29 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4715, completion~856, total~5571

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart agriculture sensor needs to report soil temperature", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~878, total~2250
[Token] Slice_Type_Determination: prompt~2708, completion~239, total~2947
[Token] Beamforming_Bandwidth: prompt~215, completion~438, total~653
[Token] Allocate_Resources: prompt~3152, completion~940, total~4092

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-15 21:29:25
Total Users: 25
Average Resource Utilization: 99.23%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 234.09 Mbps, mMTC Total Rate: 64.32 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         10  29.0/30 MHz       96.67%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 29 → mMTC Slice
CQI: 9, Bandwidth: 0.9 MHz, Rate: 8.56 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.9 MHz
  User 7: 2.9 → 2.0 MHz, Rate: 20.22 → 13.95 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        5   |         27.28 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |        2.5 |         19.48 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        0.9 |          3.19 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | mMTC    |     9 |        0.9 |          8.56 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2   |         13.95 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4425, completion~1000, total~5425

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I want to use remote surgery equipment"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to use remote surgery equipment", "urllc"),)
[Token] Intent_Analysis: prompt~1368, completion~835, total~2203
[Token] Slice_Type_Determination: prompt~2682, completion~235, total~2917
[Token] Beamforming_Bandwidth: prompt~214, completion~337, total~551
[Token] Allocate_Resources: prompt~3041, completion~683, total~3724

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-15 21:31:15
Total Users: 26
Average Resource Utilization: 100.0%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 242.73 Mbps, mMTC Total Rate: 64.32 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         11  30.0/30 MHz       100.00%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 30 → URLLC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 8.64 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        5   |         27.28 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | URLLC   |     8 |        1   |          8.64 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     7 |        2.5 |         19.48 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        0.9 |          3.19 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2   |         13.95 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3979, completion~975, total~4954

Detailed Slice Utilization Values:
eMBB utils: [22.22, 44.44, 66.67, 88.89, 88.89, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 0.0, 0.0, 0.0, 3.33, 3.33, 3.33, 6.67, 6.67, 6.67, 10.0, 10.0, 23.33, 40.0, 43.33, 60.0, 60.0, 76.67, 93.33, 96.67, 96.67, 96.67, 96.67, 96.67, 96.67, 100.0]
mMTC utils: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 58.0, 58.0, 58.0, 58.0, 77.0, 86.0, 95.0, 100.0, 100.0, 100.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |    15 |       20   |        302.7  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | eMBB    | eMBB           | Yes            |    15 |       20   |        302.7  |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | eMBB    | eMBB           | Yes            |     9 |       20   |        190.3  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | URLLC   | eMBB           | No             |    11 |        1   |         11.33 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             50 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | mMTC    | mMTC           | Yes            |     6 |        2.9 |         20.22 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | URLLC          | Yes            |     9 |        1   |          9.51 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | eMBB    | eMBB           | Yes            |    13 |       20   |        264.25 |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |    14 |        1   |         14.17 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | eMBB    | eMBB           | Yes            |     5 |       20   |        123.87 |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | URLLC          | Yes            |     4 |        4   |         21.82 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Failed   | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Failed   | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |     8 |        5   |         43.19 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Failed   | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          5.46 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Failed   | eMBB    | eMBB           | Yes            |    11 |       20   |        226.64 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | URLLC   | URLLC          | Yes            |     9 |        5   |         47.57 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | mMTC    | mMTC           | Yes            |     7 |        2.9 |         22.59 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | URLLC   | URLLC          | Yes            |     9 |        5   |         47.57 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | URLLC   | URLLC          | Yes            |     4 |        5   |         27.28 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     5 |        1   |          6.19 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | mMTC    | mMTC           | Yes            |     2 |        1.9 |          7.84 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | mMTC    | mMTC           | Yes            |     1 |        0.9 |          3.19 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | mMTC    | mMTC           | Yes            |     3 |        0.9 |          4.29 |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | mMTC    | mMTC           | Yes            |     9 |        0.9 |          8.56 |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Success  | URLLC   | URLLC          | Yes            |     8 |        1   |          8.64 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 26/30 (86.7%)

Intent Understanding Evaluation:
Correctly identified intents: 29/30
Intent understanding rate: 96.7%

Workload Balancing Statistics:
Users with workload balancing: 0/30
Workload balancing rate: 0.0%

Slice Utilization Statistics:
Average eMBB utilization: 92.73%
Average URLLC utilization: 43.33%
Average mMTC utilization: 41.54%

Weighted Average Utilization: 77.40%

Transmission Rate Statistics:
Final eMBB total rate: 870.56 Mbps
Final URLLC total rate: 242.73 Mbps
Final mMTC total rate: 64.32 Mbps

Resource Utilization:
Average resource utilization: 100.00%

Results exported to F:\code\wirelessagent\run_results\batch_run\fkb\minimax-m2.1\network_slicing_results_TJU_east_minimax-m2.1.csv

✓ TJU_east 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\fkb\minimax-m2.1\network_slicing_results_TJU_east_minimax-m2.1.csv