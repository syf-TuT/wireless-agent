场景 2/5: TJU_west
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_west.csv
输出: F:\code\wirelessagent\run_results\batch_run\nkb\minimax-m2.1\network_slicing_results_TJU_west_minimax-m2.1.csv
------------------------------------------------------------
Starting network slice management system with CSV-based user testing...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I need to download large files"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1366, completion~696, total~2062
[Token] Slice_Type_Determination: prompt~2587, completion~306, total~2893
[Token] Bandwidth_Analysis: prompt~211, completion~337, total~548
[Token] Allocate_Resources: prompt~3040, completion~742, total~3782

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-15 20:43:53
Total Users: 1
Average Resource Utilization: 15.38%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 80.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |     4 |         20 |        109.11 |             80 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4021, completion~613, total~4634

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~805, total~2179
[Token] Slice_Type_Determination: prompt~2700, completion~291, total~2991
[Token] Bandwidth_Analysis: prompt~214, completion~956, total~1170
[Token] Allocate_Resources: prompt~3144, completion~952, total~4096

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-15 20:45:06
Total Users: 2
Average Resource Utilization: 17.62%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 15.82 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          0  0/30 MHz          0%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 2 → mMTC Slice
CQI: 4, Bandwidth: 2.9 MHz, Rate: 15.82 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4339, completion~790, total~5129

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1378, completion~821, total~2199
[Token] Slice_Type_Determination: prompt~2736, completion~383, total~3119
[Token] Bandwidth_Analysis: prompt~217, completion~1072, total~1289
[Token] Allocate_Resources: prompt~3247, completion~773, total~4020

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-15 20:47:05
Total Users: 3
Average Resource Utilization: 18.38%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 7.79 Mbps, mMTC Total Rate: 15.82 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          1  1.0/30 MHz        3.33%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 3 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4281, completion~656, total~4937

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to send text messages and use messaging apps"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~981, total~2355
[Token] Slice_Type_Determination: prompt~2885, completion~375, total~3260
[Token] Workload_Balance: prompt~3357, completion~731, total~4088
[Token] Bandwidth_Analysis: prompt~217, completion~490, total~707
[Token] Allocate_Resources: prompt~4218, completion~772, total~4990

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-15 20:48:23
Total Users: 4
Average Resource Utilization: 19.15%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 15.58 Mbps, mMTC Total Rate: 15.82 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          2  2.0/30 MHz        6.67%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 4 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5321, completion~798, total~6119

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I need to use maps for basic navigation"
CQI: 3
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~882, total~2252
[Token] Slice_Type_Determination: prompt~2787, completion~346, total~3133
[Token] Bandwidth_Analysis: prompt~215, completion~416, total~631
[Token] Allocate_Resources: prompt~3284, completion~872, total~4156

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-15 20:50:05
Total Users: 5
Average Resource Utilization: 34.54%
eMBB Total Rate: 204.40 Mbps, URLLC Total Rate: 15.58 Mbps, mMTC Total Rate: 15.82 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          2  2.0/30 MHz        6.67%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 5 → eMBB Slice
CQI: 3, Bandwidth: 20.0 MHz, Rate: 95.29 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4405, completion~866, total~5271

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~901, total~2275
[Token] Slice_Type_Determination: prompt~2787, completion~262, total~3049
[Token] Workload_Balance: prompt~3145, completion~474, total~3619
[Token] Bandwidth_Analysis: prompt~217, completion~1140, total~1357
[Token] Allocate_Resources: prompt~3746, completion~721, total~4467

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-15 20:52:25
Total Users: 6
Average Resource Utilization: 38.38%
eMBB Total Rate: 204.40 Mbps, URLLC Total Rate: 50.45 Mbps, mMTC Total Rate: 15.82 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          3  7.0/30 MHz        23.33%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 6 → URLLC Slice
CQI: 6, Bandwidth: 5.0 MHz, Rate: 34.87 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4813, completion~761, total~5574

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need reliable connectivity for implanted medical devices"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~1174, total~2544
[Token] Slice_Type_Determination: prompt~3086, completion~400, total~3486
[Token] Bandwidth_Analysis: prompt~215, completion~1782, total~1997
[Token] Allocate_Resources: prompt~3611, completion~784, total~4395

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-15 20:54:33
Total Users: 7
Average Resource Utilization: 39.15%
eMBB Total Rate: 204.40 Mbps, URLLC Total Rate: 57.42 Mbps, mMTC Total Rate: 15.82 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          4  8.0/30 MHz        26.67%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 6.97 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4662, completion~906, total~5568

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I want to remotely access my work computer"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~995, total~2365
[Token] Slice_Type_Determination: prompt~2914, completion~412, total~3326
[Token] Bandwidth_Analysis: prompt~215, completion~403, total~618
[Token] Allocate_Resources: prompt~3453, completion~775, total~4228

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-15 20:55:40
Total Users: 8
Average Resource Utilization: 39.92%
eMBB Total Rate: 204.40 Mbps, URLLC Total Rate: 62.88 Mbps, mMTC Total Rate: 15.82 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          5  9.0/30 MHz        30.00%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4501, completion~671, total~5172

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to use holographic communication"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1368, completion~854, total~2222
[Token] Slice_Type_Determination: prompt~2767, completion~329, total~3096
[Token] Bandwidth_Analysis: prompt~214, completion~291, total~505
[Token] Allocate_Resources: prompt~3222, completion~698, total~3920

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-15 20:57:00
Total Users: 9
Average Resource Utilization: 51.46%
eMBB Total Rate: 431.43 Mbps, URLLC Total Rate: 62.88 Mbps, mMTC Total Rate: 15.82 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  55.0/90 MHz       61.11%
URLLC          5  9.0/30 MHz        30.00%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 9 → eMBB Slice
CQI: 15, Bandwidth: 15.0 MHz, Rate: 227.03 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4182, completion~995, total~5177

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1378, completion~844, total~2222
[Token] Slice_Type_Determination: prompt~2764, completion~288, total~3052
[Token] Bandwidth_Analysis: prompt~219, completion~394, total~613
[Token] Allocate_Resources: prompt~3177, completion~948, total~4125

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-15 20:58:50
Total Users: 10
Average Resource Utilization: 52.23%
eMBB Total Rate: 431.43 Mbps, URLLC Total Rate: 78.02 Mbps, mMTC Total Rate: 15.82 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  55.0/90 MHz       61.11%
URLLC          6  10.0/30 MHz       33.33%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 10 → URLLC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 15.14 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4394, completion~864, total~5258

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need emergency response coordination during a disaste"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~769, total~2141
[Token] Slice_Type_Determination: prompt~2683, completion~358, total~3041
[Token] Bandwidth_Analysis: prompt~216, completion~361, total~577
[Token] Allocate_Resources: prompt~3170, completion~740, total~3910

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-15 21:00:04
Total Users: 11
Average Resource Utilization: 53.0%
eMBB Total Rate: 431.43 Mbps, URLLC Total Rate: 82.78 Mbps, mMTC Total Rate: 15.82 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  55.0/90 MHz       61.11%
URLLC          7  11.0/30 MHz       36.67%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 4.76 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4175, completion~601, total~4776

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I want to use remote surgery equipment"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1368, completion~836, total~2204
[Token] Slice_Type_Determination: prompt~2748, completion~264, total~3012
[Token] Bandwidth_Analysis: prompt~214, completion~515, total~729
[Token] Allocate_Resources: prompt~3138, completion~888, total~4026

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-15 21:02:10
Total Users: 12
Average Resource Utilization: 56.85%
eMBB Total Rate: 431.43 Mbps, URLLC Total Rate: 121.73 Mbps, mMTC Total Rate: 15.82 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  55.0/90 MHz       61.11%
URLLC          8  16.0/30 MHz       53.33%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 7, Bandwidth: 5.0 MHz, Rate: 38.95 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |         38.95 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4286, completion~1060, total~5346

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need to participate in an online multiplayer game"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~976, total~2348
[Token] Slice_Type_Determination: prompt~2884, completion~370, total~3254
[Token] Bandwidth_Analysis: prompt~216, completion~442, total~658
[Token] Allocate_Resources: prompt~3377, completion~685, total~4062

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-15 21:03:58
Total Users: 13
Average Resource Utilization: 57.62%
eMBB Total Rate: 431.43 Mbps, URLLC Total Rate: 129.52 Mbps, mMTC Total Rate: 15.82 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  55.0/90 MHz       61.11%
URLLC          9  17.0/30 MHz       56.67%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |         38.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4317, completion~628, total~4945

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need to track the location of a shipping container"
CQI: 14
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~861, total~2235
[Token] Slice_Type_Determination: prompt~2766, completion~322, total~3088
[Token] Bandwidth_Analysis: prompt~216, completion~423, total~639
[Token] Allocate_Resources: prompt~3245, completion~786, total~4031

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-15 21:05:39
Total Users: 14
Average Resource Utilization: 58.31%
eMBB Total Rate: 431.43 Mbps, URLLC Total Rate: 129.52 Mbps, mMTC Total Rate: 28.57 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  55.0/90 MHz       61.11%
URLLC          9  17.0/30 MHz       56.67%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 14 → mMTC Slice
CQI: 14, Bandwidth: 0.9 MHz, Rate: 12.75 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |         38.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4269, completion~719, total~4988

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1376, completion~986, total~2362
[Token] Slice_Type_Determination: prompt~2894, completion~381, total~3275
[Token] Bandwidth_Analysis: prompt~218, completion~308, total~526
[Token] Allocate_Resources: prompt~3420, completion~766, total~4186

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-15 21:07:28
Total Users: 15
Average Resource Utilization: 73.69%
eMBB Total Rate: 734.13 Mbps, URLLC Total Rate: 129.52 Mbps, mMTC Total Rate: 28.57 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  75.0/90 MHz       83.33%
URLLC          9  17.0/30 MHz       56.67%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 15, Bandwidth: 20.0 MHz, Rate: 302.70 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |         38.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        302.7  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4438, completion~611, total~5049

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need to monitor water level in a reservoir"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~879, total~2251
[Token] Slice_Type_Determination: prompt~2789, completion~386, total~3175
[Token] Bandwidth_Analysis: prompt~215, completion~335, total~550
[Token] Allocate_Resources: prompt~3334, completion~933, total~4267

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-15 21:09:03
Total Users: 16
Average Resource Utilization: 75.92%
eMBB Total Rate: 734.13 Mbps, URLLC Total Rate: 129.52 Mbps, mMTC Total Rate: 56.16 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           4  75.0/90 MHz               83.33%
URLLC          9  17.0/30 MHz               56.67%
mMTC           3  6.699999999999999/10 MHz  67.00%

New User Allocation:
User 16 → mMTC Slice
CQI: 9, Bandwidth: 2.9 MHz, Rate: 27.59 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |         38.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        302.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2.9 |         27.59 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4535, completion~930, total~5465

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to monitor soil moisture levels in a large farm"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1376, completion~929, total~2305
[Token] Slice_Type_Determination: prompt~2822, completion~410, total~3232
[Token] Bandwidth_Analysis: prompt~222, completion~838, total~1060
[Token] Allocate_Resources: prompt~3386, completion~879, total~4265

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-15 21:10:19
Total Users: 17
Average Resource Utilization: 78.15%
eMBB Total Rate: 734.13 Mbps, URLLC Total Rate: 129.52 Mbps, mMTC Total Rate: 71.98 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  75.0/90 MHz       83.33%
URLLC          9  17.0/30 MHz       56.67%
mMTC           4  9.6/10 MHz        96.00%

New User Allocation:
User 17 → mMTC Slice
CQI: 4, Bandwidth: 2.9 MHz, Rate: 15.82 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |         38.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        302.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2.9 |         27.59 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        2.9 |         15.82 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4526, completion~940, total~5466

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need instant alerts for life-threatening patient conditions"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~839, total~2211
[Token] Slice_Type_Determination: prompt~2753, completion~270, total~3023
[Token] Bandwidth_Analysis: prompt~216, completion~631, total~847
[Token] Allocate_Resources: prompt~3151, completion~798, total~3949

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-15 21:11:45
Total Users: 18
Average Resource Utilization: 82.0%
eMBB Total Rate: 734.13 Mbps, URLLC Total Rate: 164.39 Mbps, mMTC Total Rate: 71.98 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  75.0/90 MHz       83.33%
URLLC         10  22.0/30 MHz       73.33%
mMTC           4  9.6/10 MHz        96.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 6, Bandwidth: 5.0 MHz, Rate: 34.87 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |         38.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        5   |         34.87 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       20   |        302.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2.9 |         27.59 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        2.9 |         15.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4212, completion~748, total~4960

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to download large files"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1366, completion~1098, total~2464
[Token] Slice_Type_Determination: prompt~3006, completion~243, total~3249
[Token] Bandwidth_Analysis: prompt~213, completion~367, total~580
[Token] Allocate_Resources: prompt~3442, completion~916, total~4358

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-15 21:12:53
Total Users: 19
Average Resource Utilization: 93.54%
eMBB Total Rate: 831.24 Mbps, URLLC Total Rate: 164.39 Mbps, mMTC Total Rate: 71.98 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         10  22.0/30 MHz       73.33%
mMTC           4  9.6/10 MHz        96.00%

New User Allocation:
User 19 → eMBB Slice
CQI: 8, Bandwidth: 20.0 MHz, Rate: 172.78 Mbps, Latency: 80.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 5.0 MHz
  User 15: 20.0 → 15.0 MHz, Rate: 302.70 → 227.03 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |         38.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       15   |        227.03 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       20   |        172.78 |             80 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2.9 |         27.59 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        2.9 |         15.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4683, completion~987, total~5670

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I need to synchronize distributed financial ledgers instantly"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~814, total~2186
[Token] Slice_Type_Determination: prompt~2716, completion~287, total~3003
[Token] Bandwidth_Analysis: prompt~216, completion~322, total~538
[Token] Allocate_Resources: prompt~3129, completion~788, total~3917

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-15 21:14:07
Total Users: 20
Average Resource Utilization: 97.38%
eMBB Total Rate: 831.24 Mbps, URLLC Total Rate: 211.96 Mbps, mMTC Total Rate: 71.98 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         11  27.0/30 MHz       90.00%
mMTC           4  9.6/10 MHz        96.00%

New User Allocation:
User 20 → URLLC Slice
CQI: 9, Bandwidth: 5.0 MHz, Rate: 47.57 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |         38.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         47.57 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       20   |        172.78 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2.9 |         27.59 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        2.9 |         15.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4187, completion~731, total~4918

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~851, total~2221
[Token] Slice_Type_Determination: prompt~2761, completion~371, total~3132
[Token] Bandwidth_Analysis: prompt~215, completion~376, total~591
[Token] Allocate_Resources: prompt~3260, completion~898, total~4158

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-15 21:16:03
Total Users: 21
Average Resource Utilization: 99.69%
eMBB Total Rate: 831.24 Mbps, URLLC Total Rate: 232.88 Mbps, mMTC Total Rate: 71.98 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         12  30.0/30 MHz       100.00%
mMTC           4  9.6/10 MHz        96.00%

New User Allocation:
User 21 → URLLC Slice
CQI: 6, Bandwidth: 3.0 MHz, Rate: 20.92 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |         38.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        3   |         20.92 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       20   |        172.78 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2.9 |         27.59 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        2.9 |         15.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4427, completion~942, total~5369

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "Smart city parking sensor reporting availability"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1366, completion~871, total~2237
[Token] Slice_Type_Determination: prompt~2765, completion~270, total~3035
[Token] Bandwidth_Analysis: prompt~217, completion~541, total~758
[Token] Allocate_Resources: prompt~3248, completion~827, total~4075

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-15 21:17:15
Total Users: 22
Average Resource Utilization: 100.0%
eMBB Total Rate: 831.24 Mbps, URLLC Total Rate: 232.88 Mbps, mMTC Total Rate: 74.24 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         12  30.0/30 MHz       100.00%
mMTC           5  10.0/10 MHz       100.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.49999999999999967 MHz
  User 16: 2.9 → 2.4000000000000004 MHz, Rate: 27.59 → 22.84 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |         38.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       20   |        172.78 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2.4 |         22.84 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        2.9 |         15.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4418, completion~614, total~5032

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~708, total~2080
[Token] Slice_Type_Determination: prompt~2622, completion~229, total~2851
[Token] Bandwidth_Analysis: prompt~216, completion~325, total~541
[Token] Allocate_Resources: prompt~3023, completion~837, total~3860

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-15 21:19:07
Total Users: 23
Average Resource Utilization: 100.0%
eMBB Total Rate: 831.24 Mbps, URLLC Total Rate: 228.83 Mbps, mMTC Total Rate: 74.24 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         13  30.0/30 MHz       100.00%
mMTC           5  10.0/10 MHz       100.00%

New User Allocation:
User 23 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 3.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 20: 5.0 → 4.0 MHz, Rate: 47.57 → 38.06 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        5   |         38.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        4   |         38.06 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       20   |        172.78 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2.4 |         22.84 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        2.9 |         15.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4202, completion~575, total~4777

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to monitor IoT sensors in real-time"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1134, total~2506
[Token] Slice_Type_Determination: prompt~3043, completion~492, total~3535
[Token] Bandwidth_Analysis: prompt~216, completion~326, total~542
[Token] Allocate_Resources: prompt~3706, completion~819, total~4525

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-15 21:20:50
Total Users: 24
Average Resource Utilization: 100.0%
eMBB Total Rate: 831.24 Mbps, URLLC Total Rate: 226.50 Mbps, mMTC Total Rate: 74.24 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         14  30.0/30 MHz       100.00%
mMTC           5  10.0/10 MHz       100.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 12: 5.0 → 4.0 MHz, Rate: 38.95 → 31.16 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        4   |         38.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       20   |        172.78 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        2.4 |         22.84 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        2.9 |         15.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4852, completion~671, total~5523

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "My smart trash can needs to signal that it's full"
CQI: 5
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1376, completion~1118, total~2494
[Token] Slice_Type_Determination: prompt~3019, completion~301, total~3320
[Token] Bandwidth_Analysis: prompt~217, completion~441, total~658
[Token] Allocate_Resources: prompt~3533, completion~883, total~4416

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-15 21:22:23
Total Users: 25
Average Resource Utilization: 100.0%
eMBB Total Rate: 831.24 Mbps, URLLC Total Rate: 226.50 Mbps, mMTC Total Rate: 71.24 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         14  30.0/30 MHz       100.00%
mMTC           6  10.0/10 MHz       100.00%

New User Allocation:
User 25 → mMTC Slice
CQI: 5, Bandwidth: 0.9 MHz, Rate: 5.57 Mbps, Latency: 1000.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.9 MHz
  User 16: 2.4000000000000004 → 1.5000000000000004 MHz, Rate: 22.84 → 14.27 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        4   |         38.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       20   |        172.78 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       15   |        227.03 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        1.5 |         14.27 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        2.9 |         15.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4745, completion~644, total~5389

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~951, total~2323
[Token] Slice_Type_Determination: prompt~2850, completion~412, total~3262
[Token] Bandwidth_Analysis: prompt~216, completion~479, total~695
[Token] Allocate_Resources: prompt~3523, completion~773, total~4296

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-15 21:24:20
Total Users: 26
Average Resource Utilization: 100.0%
eMBB Total Rate: 710.32 Mbps, URLLC Total Rate: 226.50 Mbps, mMTC Total Rate: 71.24 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         14  30.0/30 MHz       100.00%
mMTC           6  10.0/10 MHz       100.00%

New User Allocation:
User 26 → eMBB Slice
CQI: 7, Bandwidth: 20.0 MHz, Rate: 155.80 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 3, Bandwidth freed: 20.0 MHz
  User 9: 15.0 → 7.0 MHz, Rate: 227.03 → 105.95 Mbps, User 15: 15.0 → 7.0 MHz, Rate: 227.03 → 105.95 Mbps, User 19: 20.0 → 16.0 MHz, Rate: 172.78 → 138.22 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        4   |         38.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |        7   |        105.95 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     7 |       20   |        155.8  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        7   |        105.95 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        1.5 |         14.27 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        2.9 |         15.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2.9 |         15.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4705, completion~606, total~5311

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1364, completion~904, total~2268
[Token] Slice_Type_Determination: prompt~2786, completion~423, total~3209
[Token] Bandwidth_Analysis: prompt~211, completion~495, total~706
[Token] Allocate_Resources: prompt~3415, completion~1046, total~4461

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-15 21:25:48
Total Users: 27
Average Resource Utilization: 100.0%
eMBB Total Rate: 710.32 Mbps, URLLC Total Rate: 226.50 Mbps, mMTC Total Rate: 71.24 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         14  30.0/30 MHz       100.00%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 27 → mMTC Slice
CQI: 4, Bandwidth: 0.9 MHz, Rate: 4.91 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.9 MHz
  User 2: 2.9 → 2.0 MHz, Rate: 15.82 → 10.91 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        4   |         38.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        1.5 |         14.27 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        2.9 |         15.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2   |         10.91 |            500 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     4 |        0.9 |          4.91 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4792, completion~640, total~5432

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1378, completion~766, total~2144
[Token] Slice_Type_Determination: prompt~2667, completion~316, total~2983
[Token] Bandwidth_Analysis: prompt~218, completion~611, total~829
[Token] Allocate_Resources: prompt~3185, completion~903, total~4088

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-15 21:26:43
Total Users: 28
Average Resource Utilization: 100.0%
eMBB Total Rate: 710.32 Mbps, URLLC Total Rate: 226.50 Mbps, mMTC Total Rate: 74.89 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         14  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 9, Bandwidth: 0.9 MHz, Rate: 8.56 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.9 MHz
  User 17: 2.9 → 2.0 MHz, Rate: 15.82 → 10.91 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        4   |         38.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        1.5 |         14.27 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        2   |         10.91 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        2   |         10.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     9 |        0.9 |          8.56 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4413, completion~873, total~5286

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I want to stream music while browsing social media"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~952, total~2324
[Token] Slice_Type_Determination: prompt~2857, completion~320, total~3177
[Token] Bandwidth_Analysis: prompt~216, completion~576, total~792
[Token] Allocate_Resources: prompt~3369, completion~858, total~4227
[Token] Failure_Evaluation: prompt~4338, completion~649, total~4987

----------------------------------------
ALLOCATION FAILED FOR USER 29
----------------------------------------
Request: I want to stream music while browsing social media
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 11
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1364, completion~954, total~2318
[Token] Slice_Type_Determination: prompt~2839, completion~467, total~3306
[Token] Bandwidth_Analysis: prompt~211, completion~353, total~564
[Token] Allocate_Resources: prompt~3552, completion~858, total~4410

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-15 21:29:56
Total Users: 29
Average Resource Utilization: 100.0%
eMBB Total Rate: 710.32 Mbps, URLLC Total Rate: 226.50 Mbps, mMTC Total Rate: 78.15 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         14  30.0/30 MHz       100.00%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 30 → mMTC Slice
CQI: 11, Bandwidth: 0.9 MHz, Rate: 10.20 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 0.9 MHz
  User 16: 1.5000000000000004 → 1.0 MHz, Rate: 14.27 → 9.51 Mbps, User 2: 2.0 → 1.6000000000000005 MHz, Rate: 10.91 → 8.73 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        4   |         38.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        1   |          9.51 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        2   |         10.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.6 |          8.73 |            500 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | mMTC    |    11 |        0.9 |         10.2  |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4804, completion~944, total~5748

Detailed Slice Utilization Values:
eMBB utils: [22.22, 22.22, 22.22, 22.22, 44.44, 44.44, 44.44, 44.44, 61.11, 61.11, 61.11, 61.11, 61.11, 61.11, 83.33, 83.33, 83.33, 83.33, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 0.0, 3.33, 6.67, 6.67, 23.33, 26.67, 30.0, 30.0, 33.33, 36.67, 53.33, 56.67, 56.67, 56.67, 56.67, 56.67, 73.33, 73.33, 90.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
mMTC utils: [0.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 38.0, 38.0, 67.0, 96.0, 96.0, 96.0, 96.0, 96.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             80 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | mMTC    | mMTC           | Yes            |     4 |        2.9 |         15.82 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | URLLC   | URLLC          | Yes            |     7 |        1   |          7.79 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | URLLC   | eMBB           | No             |     7 |        1   |          7.79 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | eMBB    | eMBB           | Yes            |     3 |       20   |         95.29 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | URLLC   | mMTC           | No             |     6 |        5   |         34.87 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |     6 |        1   |          6.97 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | eMBB           | No             |     4 |        1   |          5.46 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | eMBB    | eMBB           | Yes            |    15 |       15   |        227.03 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | URLLC   | URLLC          | Yes            |    15 |        1   |         15.14 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |     3 |        1   |          4.76 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     7 |        5   |         38.95 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | URLLC          | Yes            |     7 |        1   |          7.79 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | mMTC    | mMTC           | Yes            |    14 |        0.9 |         12.75 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |    15 |       20   |        302.7  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | mMTC    | mMTC           | Yes            |     9 |        2.9 |         27.59 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | mMTC    | mMTC           | Yes            |     4 |        2.9 |         15.82 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     6 |        5   |         34.87 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | eMBB    | eMBB           | Yes            |     8 |       20   |        172.78 |             80 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | URLLC   | URLLC          | Yes            |     9 |        5   |         47.57 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | URLLC   | URLLC          | Yes            |     6 |        3   |         20.92 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          5.46 |              3 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          5.46 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | mMTC    | mMTC           | Yes            |     5 |        0.9 |          5.57 |           1000 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | mMTC    | mMTC           | Yes            |     4 |        0.9 |          4.91 |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | mMTC    | mMTC           | Yes            |     9 |        0.9 |          8.56 |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Failed   | eMBB    | eMBB           | Yes            |     5 |       20   |        123.87 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Success  | mMTC    | mMTC           | Yes            |    11 |        0.9 |         10.2  |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 29/30 (96.7%)

Intent Understanding Evaluation:
Correctly identified intents: 27/30
Intent understanding rate: 90.0%

Workload Balancing Statistics:
Users with workload balancing: 0/30
Workload balancing rate: 0.0%

Slice Utilization Statistics:
Average eMBB utilization: 71.26%
Average URLLC utilization: 57.59%
Average mMTC utilization: 61.07%

Weighted Average Utilization: 67.32%

Transmission Rate Statistics:
Final eMBB total rate: 710.32 Mbps
Final URLLC total rate: 226.50 Mbps
Final mMTC total rate: 78.15 Mbps

Resource Utilization:
Average resource utilization: 100.00%

Results exported to F:\code\wirelessagent\run_results\batch_run\nkb\minimax-m2.1\network_slicing_results_TJU_west_minimax-m2.1.csv