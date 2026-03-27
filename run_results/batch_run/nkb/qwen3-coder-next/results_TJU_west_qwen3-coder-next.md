============================================================
场景 1/5: TJU_west
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_west.csv
输出: F:\code\wirelessagent\run_results\batch_run\nkb\qwen3-coder-next\network_slicing_results_TJU_west_qwen3-coder-next.csv
------------------------------------------------------------
Starting network slice management system with CSV-based user testing...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I need to download large files"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1366, completion~677, total~2043
[Token] Slice_Type_Determination: prompt~2536, completion~132, total~2668
[Token] Bandwidth_Analysis: prompt~211, completion~3, total~214
[Token] Allocate_Resources: prompt~2807, completion~275, total~3082

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-18 15:15:56
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
[Token] Network_Evaluation: prompt~3280, completion~397, total~3677

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~796, total~2170
[Token] Slice_Type_Determination: prompt~2668, completion~137, total~2805
[Token] Bandwidth_Analysis: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~2955, completion~324, total~3279

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-18 15:16:36
Total Users: 2
Average Resource Utilization: 16.85%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 10.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          0  0/30 MHz          0%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 2 → mMTC Slice
CQI: 4, Bandwidth: 1.9 MHz, Rate: 10.37 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3502, completion~719, total~4221

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1378, completion~749, total~2127
[Token] Slice_Type_Determination: prompt~2602, completion~116, total~2718
[Token] Bandwidth_Analysis: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~2826, completion~205, total~3031

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-18 15:17:01
Total Users: 3
Average Resource Utilization: 18.38%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 15.58 Mbps, mMTC Total Rate: 10.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          1  2.0/30 MHz        6.67%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 3 → URLLC Slice
CQI: 7, Bandwidth: 2.0 MHz, Rate: 15.58 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3258, completion~312, total~3570

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to send text messages and use messaging apps"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~1180, total~2554
[Token] Slice_Type_Determination: prompt~3030, completion~196, total~3226
[Token] Bandwidth_Analysis: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~3362, completion~429, total~3791

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-18 15:17:45
Total Users: 4
Average Resource Utilization: 33.77%
eMBB Total Rate: 264.91 Mbps, URLLC Total Rate: 15.58 Mbps, mMTC Total Rate: 10.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          1  2.0/30 MHz        6.67%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 4 → eMBB Slice
CQI: 7, Bandwidth: 20.0 MHz, Rate: 155.80 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4008, completion~398, total~4406

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I need to use maps for basic navigation"
CQI: 3
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~1259, total~2629
[Token] Slice_Type_Determination: prompt~3128, completion~164, total~3292
[Token] Bandwidth_Analysis: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~3447, completion~340, total~3787

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-18 15:18:16
Total Users: 5
Average Resource Utilization: 34.46%
eMBB Total Rate: 264.91 Mbps, URLLC Total Rate: 15.58 Mbps, mMTC Total Rate: 14.66 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          1  2.0/30 MHz        6.67%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 5 → mMTC Slice
CQI: 3, Bandwidth: 0.9 MHz, Rate: 4.29 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4017, completion~478, total~4495

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~562, total~1936
[Token] Slice_Type_Determination: prompt~2455, completion~107, total~2562
[Token] Workload_Balance: prompt~2657, completion~192, total~2849
[Token] Bandwidth_Analysis: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~2954, completion~201, total~3155

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-18 15:19:02
Total Users: 6
Average Resource Utilization: 36.0%
eMBB Total Rate: 264.91 Mbps, URLLC Total Rate: 29.53 Mbps, mMTC Total Rate: 14.66 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          2  4.0/30 MHz        13.33%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 6 → URLLC Slice
CQI: 6, Bandwidth: 2.0 MHz, Rate: 13.95 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3460, completion~280, total~3740

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need reliable connectivity for implanted medical devices"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~967, total~2337
[Token] Slice_Type_Determination: prompt~2831, completion~125, total~2956
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3070, completion~185, total~3255

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-18 15:19:22
Total Users: 7
Average Resource Utilization: 37.54%
eMBB Total Rate: 264.91 Mbps, URLLC Total Rate: 43.48 Mbps, mMTC Total Rate: 14.66 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          3  6.0/30 MHz        20.00%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 6, Bandwidth: 2.0 MHz, Rate: 13.95 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3481, completion~217, total~3698

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I want to remotely access my work computer"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~1312, total~2682
[Token] Slice_Type_Determination: prompt~3145, completion~135, total~3280
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3388, completion~174, total~3562

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-18 15:19:48
Total Users: 8
Average Resource Utilization: 39.08%
eMBB Total Rate: 264.91 Mbps, URLLC Total Rate: 54.39 Mbps, mMTC Total Rate: 14.66 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          4  8.0/30 MHz        26.67%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 4, Bandwidth: 2.0 MHz, Rate: 10.91 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3789, completion~326, total~4115

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to use holographic communication"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1368, completion~1355, total~2723
[Token] Slice_Type_Determination: prompt~3168, completion~156, total~3324
[Token] Bandwidth_Analysis: prompt~214, completion~3, total~217
[Token] Allocate_Resources: prompt~3428, completion~180, total~3608

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-18 15:20:09
Total Users: 9
Average Resource Utilization: 48.31%
eMBB Total Rate: 446.53 Mbps, URLLC Total Rate: 54.39 Mbps, mMTC Total Rate: 14.66 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  52.0/90 MHz       57.78%
URLLC          4  8.0/30 MHz        26.67%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 9 → eMBB Slice
CQI: 15, Bandwidth: 12.0 MHz, Rate: 181.62 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3827, completion~325, total~4152

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1378, completion~664, total~2042
[Token] Slice_Type_Determination: prompt~2537, completion~130, total~2667
[Token] Bandwidth_Analysis: prompt~219, completion~2, total~221
[Token] Allocate_Resources: prompt~2774, completion~181, total~2955

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-18 15:20:23
Total Users: 10
Average Resource Utilization: 49.85%
eMBB Total Rate: 446.53 Mbps, URLLC Total Rate: 84.66 Mbps, mMTC Total Rate: 14.66 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  52.0/90 MHz       57.78%
URLLC          5  10.0/30 MHz       33.33%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 10 → URLLC Slice
CQI: 15, Bandwidth: 2.0 MHz, Rate: 30.27 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3181, completion~318, total~3499

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need emergency response coordination during a disaste"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~997, total~2369
[Token] Slice_Type_Determination: prompt~2860, completion~125, total~2985
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3101, completion~179, total~3280

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-18 15:20:41
Total Users: 11
Average Resource Utilization: 53.69%
eMBB Total Rate: 446.53 Mbps, URLLC Total Rate: 108.48 Mbps, mMTC Total Rate: 14.66 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  52.0/90 MHz       57.78%
URLLC          6  15.0/30 MHz       50.00%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 3, Bandwidth: 5.0 MHz, Rate: 23.82 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3505, completion~264, total~3769

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I want to use remote surgery equipment"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1368, completion~715, total~2083
[Token] Slice_Type_Determination: prompt~2575, completion~142, total~2717
[Token] Bandwidth_Analysis: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~2828, completion~194, total~3022

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-18 15:21:02
Total Users: 12
Average Resource Utilization: 55.23%
eMBB Total Rate: 446.53 Mbps, URLLC Total Rate: 124.06 Mbps, mMTC Total Rate: 14.66 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  52.0/90 MHz       57.78%
URLLC          7  17.0/30 MHz       56.67%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 7, Bandwidth: 2.0 MHz, Rate: 15.58 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3249, completion~347, total~3596

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need to participate in an online multiplayer game"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1327, total~2699
[Token] Slice_Type_Determination: prompt~3161, completion~124, total~3285
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3395, completion~163, total~3558

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-18 15:21:18
Total Users: 13
Average Resource Utilization: 56.77%
eMBB Total Rate: 446.53 Mbps, URLLC Total Rate: 139.64 Mbps, mMTC Total Rate: 14.66 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  52.0/90 MHz       57.78%
URLLC          8  19.0/30 MHz       63.33%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 7, Bandwidth: 2.0 MHz, Rate: 15.58 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3787, completion~350, total~4137

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need to track the location of a shipping container"
CQI: 14
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~1042, total~2416
[Token] Slice_Type_Determination: prompt~2915, completion~139, total~3054
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3205, completion~299, total~3504

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-18 15:21:36
Total Users: 14
Average Resource Utilization: 58.23%
eMBB Total Rate: 446.53 Mbps, URLLC Total Rate: 139.64 Mbps, mMTC Total Rate: 41.58 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           3  52.0/90 MHz               57.78%
URLLC          8  19.0/30 MHz               63.33%
mMTC           3  4.699999999999999/10 MHz  47.00%

New User Allocation:
User 14 → mMTC Slice
CQI: 14, Bandwidth: 1.9 MHz, Rate: 26.92 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.9 |         26.92 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3732, completion~756, total~4488

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1376, completion~1174, total~2550
[Token] Slice_Type_Determination: prompt~3014, completion~131, total~3145
[Token] Bandwidth_Analysis: prompt~218, completion~3, total~221
[Token] Allocate_Resources: prompt~3255, completion~195, total~3450

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-18 15:21:57
Total Users: 15
Average Resource Utilization: 67.46%
eMBB Total Rate: 628.15 Mbps, URLLC Total Rate: 139.64 Mbps, mMTC Total Rate: 41.58 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           4  64.0/90 MHz               71.11%
URLLC          8  19.0/30 MHz               63.33%
mMTC           3  4.699999999999999/10 MHz  47.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 15, Bandwidth: 12.0 MHz, Rate: 181.62 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       12   |        181.62 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.9 |         26.92 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3673, completion~376, total~4049

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need to monitor water level in a reservoir"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~876, total~2248
[Token] Slice_Type_Determination: prompt~2757, completion~114, total~2871
[Token] Bandwidth_Analysis: prompt~219, completion~2, total~221
[Token] Allocate_Resources: prompt~3022, completion~453, total~3475

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-18 15:22:20
Total Users: 16
Average Resource Utilization: 68.15%
eMBB Total Rate: 628.15 Mbps, URLLC Total Rate: 139.64 Mbps, mMTC Total Rate: 50.14 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  64.0/90 MHz       71.11%
URLLC          8  19.0/30 MHz       63.33%
mMTC           4  5.6/10 MHz        56.00%

New User Allocation:
User 16 → mMTC Slice
CQI: 9, Bandwidth: 0.9 MHz, Rate: 8.56 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.9 |         26.92 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3700, completion~322, total~4022

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to monitor soil moisture levels in a large farm"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1376, completion~687, total~2063
[Token] Slice_Type_Determination: prompt~2570, completion~131, total~2701
[Token] Bandwidth_Analysis: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~2852, completion~362, total~3214

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-18 15:22:34
Total Users: 17
Average Resource Utilization: 69.62%
eMBB Total Rate: 628.15 Mbps, URLLC Total Rate: 139.64 Mbps, mMTC Total Rate: 60.51 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  64.0/90 MHz       71.11%
URLLC          8  19.0/30 MHz       63.33%
mMTC           5  7.5/10 MHz        75.00%

New User Allocation:
User 17 → mMTC Slice
CQI: 4, Bandwidth: 1.9 MHz, Rate: 10.37 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.9 |         26.92 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1.9 |         10.37 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3447, completion~514, total~3961

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need instant alerts for life-threatening patient conditions"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~910, total~2282
[Token] Slice_Type_Determination: prompt~2763, completion~131, total~2894
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3010, completion~185, total~3195

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-18 15:23:39
Total Users: 18
Average Resource Utilization: 71.15%
eMBB Total Rate: 628.15 Mbps, URLLC Total Rate: 153.59 Mbps, mMTC Total Rate: 60.51 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  64.0/90 MHz       71.11%
URLLC          9  21.0/30 MHz       70.00%
mMTC           5  7.5/10 MHz        75.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 6, Bandwidth: 2.0 MHz, Rate: 13.95 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |         13.95 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.9 |         26.92 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1.9 |         10.37 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3424, completion~358, total~3782

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to download large files"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1366, completion~665, total~2031
[Token] Slice_Type_Determination: prompt~2518, completion~109, total~2627
[Token] Bandwidth_Analysis: prompt~213, completion~3, total~216
[Token] Allocate_Resources: prompt~2740, completion~199, total~2939

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-18 15:24:03
Total Users: 19
Average Resource Utilization: 83.46%
eMBB Total Rate: 766.37 Mbps, URLLC Total Rate: 153.59 Mbps, mMTC Total Rate: 60.51 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  80.0/90 MHz       88.89%
URLLC          9  21.0/30 MHz       70.00%
mMTC           5  7.5/10 MHz        75.00%

New User Allocation:
User 19 → eMBB Slice
CQI: 8, Bandwidth: 16.0 MHz, Rate: 138.22 Mbps, Latency: 80.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.9 |         26.92 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1.9 |         10.37 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3159, completion~405, total~3564

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I need to synchronize distributed financial ledgers instantly"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1096, total~2468
[Token] Slice_Type_Determination: prompt~2947, completion~95, total~3042
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3158, completion~181, total~3339

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-18 15:24:19
Total Users: 20
Average Resource Utilization: 85.0%
eMBB Total Rate: 766.37 Mbps, URLLC Total Rate: 172.62 Mbps, mMTC Total Rate: 60.51 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  80.0/90 MHz       88.89%
URLLC         10  23.0/30 MHz       76.67%
mMTC           5  7.5/10 MHz        75.00%

New User Allocation:
User 20 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 19.03 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.9 |         26.92 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1.9 |         10.37 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3566, completion~322, total~3888

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~1235, total~2605
[Token] Slice_Type_Determination: prompt~3077, completion~120, total~3197
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3308, completion~209, total~3517

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-18 15:24:35
Total Users: 21
Average Resource Utilization: 86.54%
eMBB Total Rate: 766.37 Mbps, URLLC Total Rate: 186.57 Mbps, mMTC Total Rate: 60.51 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  80.0/90 MHz       88.89%
URLLC         11  25.0/30 MHz       83.33%
mMTC           5  7.5/10 MHz        75.00%

New User Allocation:
User 21 → URLLC Slice
CQI: 6, Bandwidth: 2.0 MHz, Rate: 13.95 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |         13.95 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.9 |         26.92 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1.9 |         10.37 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3744, completion~343, total~4087

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "Smart city parking sensor reporting availability"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1366, completion~679, total~2045
[Token] Slice_Type_Determination: prompt~2555, completion~105, total~2660
[Token] Bandwidth_Analysis: prompt~212, completion~2, total~214
[Token] Allocate_Resources: prompt~2814, completion~518, total~3332

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-18 15:25:12
Total Users: 22
Average Resource Utilization: 87.23%
eMBB Total Rate: 766.37 Mbps, URLLC Total Rate: 186.57 Mbps, mMTC Total Rate: 67.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  80.0/90 MHz       88.89%
URLLC         11  25.0/30 MHz       83.33%
mMTC           6  8.4/10 MHz        84.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.9 |         26.92 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1.9 |         10.37 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3536, completion~320, total~3856

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1255, total~2627
[Token] Slice_Type_Determination: prompt~3114, completion~104, total~3218
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3331, completion~212, total~3543

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-18 15:25:40
Total Users: 23
Average Resource Utilization: 88.77%
eMBB Total Rate: 766.37 Mbps, URLLC Total Rate: 197.48 Mbps, mMTC Total Rate: 67.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  80.0/90 MHz       88.89%
URLLC         12  27.0/30 MHz       90.00%
mMTC           6  8.4/10 MHz        84.00%

New User Allocation:
User 23 → URLLC Slice
CQI: 4, Bandwidth: 2.0 MHz, Rate: 10.91 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.9 |         26.92 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1.9 |         10.37 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3771, completion~460, total~4231

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to monitor IoT sensors in real-time"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1265, total~2637
[Token] Slice_Type_Determination: prompt~3116, completion~94, total~3210
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3326, completion~165, total~3491

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-18 15:26:00
Total Users: 24
Average Resource Utilization: 90.31%
eMBB Total Rate: 766.37 Mbps, URLLC Total Rate: 208.39 Mbps, mMTC Total Rate: 67.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  80.0/90 MHz       88.89%
URLLC         13  29.0/30 MHz       96.67%
mMTC           6  8.4/10 MHz        84.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 4, Bandwidth: 2.0 MHz, Rate: 10.91 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        2   |         10.91 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.9 |         26.92 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1.9 |         10.37 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3719, completion~403, total~4122

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "My smart trash can needs to signal that it's full"
CQI: 5
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1376, completion~950, total~2326
[Token] Slice_Type_Determination: prompt~2827, completion~144, total~2971
[Token] Bandwidth_Analysis: prompt~237, completion~6, total~243
[Token] Allocate_Resources: prompt~3122, completion~583, total~3705

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-18 15:26:17
Total Users: 25
Average Resource Utilization: 91.0%
eMBB Total Rate: 766.37 Mbps, URLLC Total Rate: 208.39 Mbps, mMTC Total Rate: 73.09 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  80.0/90 MHz       88.89%
URLLC         13  29.0/30 MHz       96.67%
mMTC           7  9.3/10 MHz        93.00%

New User Allocation:
User 25 → mMTC Slice
CQI: 5, Bandwidth: 0.9 MHz, Rate: 5.57 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.9 |         26.92 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1.9 |         10.37 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3930, completion~457, total~4387

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1518, total~2890
[Token] Slice_Type_Determination: prompt~3356, completion~145, total~3501
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3713, completion~204, total~3917

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-18 15:26:48
Total Users: 26
Average Resource Utilization: 98.69%
eMBB Total Rate: 770.83 Mbps, URLLC Total Rate: 208.39 Mbps, mMTC Total Rate: 73.09 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         13  29.0/30 MHz       96.67%
mMTC           7  9.3/10 MHz        93.00%

New User Allocation:
User 26 → eMBB Slice
CQI: 7, Bandwidth: 20.0 MHz, Rate: 155.80 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 10.0 MHz
  User 9: 12.0 → 7.0 MHz, Rate: 181.62 → 105.95 Mbps, User 15: 12.0 → 7.0 MHz, Rate: 181.62 → 105.95 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |        7   |        105.95 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     7 |       20   |        155.8  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        7   |        105.95 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.9 |         26.92 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1.9 |         10.37 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4246, completion~259, total~4505

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1364, completion~1496, total~2860
[Token] Slice_Type_Determination: prompt~3303, completion~131, total~3434
[Token] Bandwidth_Analysis: prompt~216, completion~4, total~220
[Token] Allocate_Resources: prompt~3642, completion~504, total~4146

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-18 15:27:51
Total Users: 27
Average Resource Utilization: 99.23%
eMBB Total Rate: 770.83 Mbps, URLLC Total Rate: 208.39 Mbps, mMTC Total Rate: 75.17 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         13  29.0/30 MHz       96.67%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 27 → mMTC Slice
CQI: 4, Bandwidth: 0.9 MHz, Rate: 4.91 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.20000000000000073 MHz
  User 14: 1.9 → 1.6999999999999993 MHz, Rate: 26.92 → 24.09 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1.7 |         24.09 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1.9 |         10.37 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     4 |        0.9 |          4.91 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4450, completion~468, total~4918

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1378, completion~964, total~2342
[Token] Slice_Type_Determination: prompt~2828, completion~141, total~2969
[Token] Bandwidth_Analysis: prompt~218, completion~4, total~222
[Token] Allocate_Resources: prompt~3206, completion~663, total~3869

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-18 15:28:09
Total Users: 28
Average Resource Utilization: 99.23%
eMBB Total Rate: 770.83 Mbps, URLLC Total Rate: 208.39 Mbps, mMTC Total Rate: 72.71 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         13  29.0/30 MHz       96.67%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 9, Bandwidth: 0.9 MHz, Rate: 8.56 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 0.9 MHz
  User 14: 1.6999999999999993 → 1.0 MHz, Rate: 24.09 → 14.17 Mbps, User 2: 1.9 → 1.6999999999999993 MHz, Rate: 10.37 → 9.27 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1   |         14.17 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1.9 |         10.37 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.7 |          9.27 |            500 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     9 |        0.9 |          8.56 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4200, completion~426, total~4626

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I want to stream music while browsing social media"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~961, total~2333
[Token] Slice_Type_Determination: prompt~2820, completion~137, total~2957
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3142, completion~356, total~3498
[Token] Failure_Evaluation: prompt~3565, completion~400, total~3965

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
[Token] Intent_Analysis: prompt~1364, completion~1201, total~2565
[Token] Slice_Type_Determination: prompt~3049, completion~119, total~3168
[Token] Bandwidth_Analysis: prompt~211, completion~4, total~215
[Token] Allocate_Resources: prompt~3406, completion~525, total~3931

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-18 15:29:06
Total Users: 29
Average Resource Utilization: 99.23%
eMBB Total Rate: 770.83 Mbps, URLLC Total Rate: 208.39 Mbps, mMTC Total Rate: 78.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         13  29.0/30 MHz       96.67%
mMTC          10  10.0/10 MHz       100.00%

New User Allocation:
User 30 → mMTC Slice
CQI: 11, Bandwidth: 0.9 MHz, Rate: 10.20 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 0.9 MHz
  User 17: 1.9 → 1.0 MHz, Rate: 10.37 → 5.46 Mbps, User 2: 1.6999999999999993 → 1.6999999999999993 MHz, Rate: 9.27 → 9.27 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        5   |         23.82 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |       16   |        138.22 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        1   |         14.17 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        1   |          5.46 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.7 |          9.27 |            500 | ADJUSTED |
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
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4279, completion~333, total~4612

Detailed Slice Utilization Values:
eMBB utils: [22.22, 22.22, 22.22, 44.44, 44.44, 44.44, 44.44, 44.44, 57.78, 57.78, 57.78, 57.78, 57.78, 57.78, 71.11, 71.11, 71.11, 71.11, 88.89, 88.89, 88.89, 88.89, 88.89, 88.89, 88.89, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 0.0, 6.67, 6.67, 6.67, 13.33, 20.0, 26.67, 26.67, 33.33, 50.0, 56.67, 63.33, 63.33, 63.33, 63.33, 63.33, 70.0, 70.0, 76.67, 83.33, 83.33, 90.0, 96.67, 96.67, 96.67, 96.67, 96.67, 96.67]
mMTC utils: [0.0, 19.0, 19.0, 19.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 47.0, 47.0, 56.0, 75.0, 75.0, 75.0, 75.0, 75.0, 84.0, 84.0, 84.0, 93.0, 93.0, 100.0, 100.0, 100.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             80 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | mMTC    | mMTC           | Yes            |     4 |        1.9 |         10.37 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | URLLC   | URLLC          | Yes            |     7 |        2   |         15.58 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | mMTC    | eMBB           | No             |     3 |        0.9 |          4.29 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | URLLC   | mMTC           | No             |     6 |        2   |         13.95 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |     6 |        2   |         13.95 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | eMBB           | No             |     4 |        2   |         10.91 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | eMBB    | eMBB           | Yes            |    15 |       12   |        181.62 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | URLLC   | URLLC          | Yes            |    15 |        2   |         30.27 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |     3 |        5   |         23.82 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     7 |        2   |         15.58 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | URLLC          | Yes            |     7 |        2   |         15.58 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | mMTC    | mMTC           | Yes            |    14 |        1.9 |         26.92 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |    15 |       12   |        181.62 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | mMTC    | mMTC           | Yes            |     9 |        0.9 |          8.56 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | mMTC    | mMTC           | Yes            |     4 |        1.9 |         10.37 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     6 |        2   |         13.95 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | eMBB    | eMBB           | Yes            |     8 |       16   |        138.22 |             80 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | URLLC   | URLLC          | Yes            |     9 |        2   |         19.03 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | URLLC   | URLLC          | Yes            |     6 |        2   |         13.95 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | URLLC   | URLLC          | Yes            |     4 |        2   |         10.91 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     4 |        2   |         10.91 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | mMTC    | mMTC           | Yes            |     5 |        0.9 |          5.57 |           1000 | No         |
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
Average eMBB utilization: 66.97%
Average URLLC utilization: 55.75%
Average mMTC utilization: 54.21%

Weighted Average Utilization: 63.40%

Transmission Rate Statistics:
Final eMBB total rate: 770.83 Mbps
Final URLLC total rate: 208.39 Mbps
Final mMTC total rate: 78.00 Mbps

Resource Utilization:
Average resource utilization: 99.23%

Results exported to F:\code\wirelessagent\run_results\batch_run\nkb\qwen3-coder-next\network_slicing_results_TJU_west_qwen3-coder-next.csv

✓ TJU_west 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\nkb\qwen3-coder-next\network_slicing_results_TJU_west_qwen3-coder-next.csv