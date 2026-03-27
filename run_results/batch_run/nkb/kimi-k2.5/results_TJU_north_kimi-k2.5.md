============================================================
场景 1/5: TJU_north
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_north.csv
输出: F:\code\wirelessagent\run_results\batch_run\nkb\kimi-k2.5\network_slicing_results_TJU_north_kimi-k2.5.csv
------------------------------------------------------------
Starting network slice management system with CSV-based user testing...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I want to use augmented reality navigation"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1368, completion~985, total~2353
[Token] Slice_Type_Determination: prompt~2898, completion~273, total~3171
[Token] Bandwidth_Analysis: prompt~212, completion~449, total~661
[Token] Allocate_Resources: prompt~3319, completion~1138, total~4457

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-22 16:09:23
Total Users: 1
Average Resource Utilization: 15.38%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → eMBB Slice
CQI: 8, Bandwidth: 20.0 MHz, Rate: 172.78 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |     8 |         20 |        172.78 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4713, completion~899, total~5612

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1193, total~2565
[Token] Slice_Type_Determination: prompt~3140, completion~496, total~3636
[Token] Bandwidth_Analysis: prompt~214, completion~381, total~595
[Token] Allocate_Resources: prompt~3764, completion~730, total~4494

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-22 16:10:20
Total Users: 2
Average Resource Utilization: 16.15%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 6.19 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          1  1.0/30 MHz        3.33%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → URLLC Slice
CQI: 5, Bandwidth: 1.0 MHz, Rate: 6.19 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          1 |          6.19 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         20 |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4754, completion~853, total~5607

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 11
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1380, completion~1122, total~2502
[Token] Slice_Type_Determination: prompt~3051, completion~288, total~3339
[Token] Bandwidth_Analysis: prompt~220, completion~540, total~760
[Token] Allocate_Resources: prompt~3463, completion~830, total~4293

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-22 16:11:22
Total Users: 3
Average Resource Utilization: 16.92%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 17.52 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          2  2.0/30 MHz        6.67%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 3 → URLLC Slice
CQI: 11, Bandwidth: 1.0 MHz, Rate: 11.33 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          1 |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          1 |         11.33 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         20 |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4559, completion~936, total~5495

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1378, completion~891, total~2269
[Token] Slice_Type_Determination: prompt~2797, completion~470, total~3267
[Token] Bandwidth_Analysis: prompt~216, completion~535, total~751
[Token] Allocate_Resources: prompt~3424, completion~1250, total~4674

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-22 16:12:35
Total Users: 4
Average Resource Utilization: 17.62%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 17.52 Mbps, mMTC Total Rate: 7.78 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          2  2.0/30 MHz        6.67%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 4 → mMTC Slice
CQI: 8, Bandwidth: 0.9 MHz, Rate: 7.78 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4933, completion~710, total~5643

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I want to use remote surgery equipment"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1368, completion~1223, total~2591
[Token] Slice_Type_Determination: prompt~3157, completion~296, total~3453
[Token] Bandwidth_Analysis: prompt~214, completion~537, total~751
[Token] Allocate_Resources: prompt~3581, completion~899, total~4480

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-22 16:13:36
Total Users: 5
Average Resource Utilization: 18.38%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 31.69 Mbps, mMTC Total Rate: 7.78 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          3  3.0/30 MHz        10.00%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 5 → URLLC Slice
CQI: 14, Bandwidth: 1.0 MHz, Rate: 14.17 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4743, completion~1005, total~5748

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1378, completion~1038, total~2416
[Token] Slice_Type_Determination: prompt~2947, completion~314, total~3261
[Token] Bandwidth_Analysis: prompt~218, completion~758, total~976
[Token] Allocate_Resources: prompt~3419, completion~982, total~4401

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-22 16:14:40
Total Users: 6
Average Resource Utilization: 19.08%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 31.69 Mbps, mMTC Total Rate: 14.79 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          3  3.0/30 MHz        10.00%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 6 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4647, completion~778, total~5425

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1359, total~2731
[Token] Slice_Type_Determination: prompt~3295, completion~411, total~3706
[Token] Bandwidth_Analysis: prompt~216, completion~801, total~1017
[Token] Allocate_Resources: prompt~3840, completion~807, total~4647

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-22 16:15:48
Total Users: 7
Average Resource Utilization: 19.85%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 38.66 Mbps, mMTC Total Rate: 14.79 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          4  4.0/30 MHz        13.33%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 6.97 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4920, completion~1001, total~5921

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I need to transmit real-time patient vital signs during critical care"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1378, completion~961, total~2339
[Token] Slice_Type_Determination: prompt~2885, completion~264, total~3149
[Token] Bandwidth_Analysis: prompt~219, completion~696, total~915
[Token] Allocate_Resources: prompt~3273, completion~851, total~4124

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-22 16:16:53
Total Users: 8
Average Resource Utilization: 20.62%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 47.30 Mbps, mMTC Total Rate: 14.79 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          5  5.0/30 MHz        16.67%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 8.64 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4376, completion~780, total~5156

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1378, completion~1070, total~2448
[Token] Slice_Type_Determination: prompt~2978, completion~336, total~3314
[Token] Bandwidth_Analysis: prompt~218, completion~635, total~853
[Token] Allocate_Resources: prompt~3470, completion~1002, total~4472

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-22 16:17:51
Total Users: 9
Average Resource Utilization: 21.31%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 47.30 Mbps, mMTC Total Rate: 21.80 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          5  5.0/30 MHz        16.67%
mMTC           3  2.7/10 MHz        27.00%

New User Allocation:
User 9 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4730, completion~940, total~5670

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~1106, total~2480
[Token] Slice_Type_Determination: prompt~3008, completion~404, total~3412
[Token] Bandwidth_Analysis: prompt~216, completion~743, total~959
[Token] Allocate_Resources: prompt~3572, completion~2172, total~5744

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-22 16:19:24
Total Users: 10
Average Resource Utilization: 22.0%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 47.30 Mbps, mMTC Total Rate: 30.36 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          5  5.0/30 MHz        16.67%
mMTC           4  3.6/10 MHz        36.00%

New User Allocation:
User 10 → mMTC Slice
CQI: 9, Bandwidth: 0.9 MHz, Rate: 8.56 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~6016, completion~947, total~6963

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~893, total~2267
[Token] Slice_Type_Determination: prompt~2817, completion~338, total~3155
[Token] Bandwidth_Analysis: prompt~217, completion~390, total~607
[Token] Allocate_Resources: prompt~3283, completion~847, total~4130

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-22 16:20:15
Total Users: 11
Average Resource Utilization: 22.77%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 62.44 Mbps, mMTC Total Rate: 30.36 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          6  6.0/30 MHz        20.00%
mMTC           4  3.6/10 MHz        36.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 15.14 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4402, completion~1075, total~5477

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need to check the status of my smart home sensors"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1376, completion~887, total~2263
[Token] Slice_Type_Determination: prompt~2780, completion~436, total~3216
[Token] Bandwidth_Analysis: prompt~217, completion~592, total~809
[Token] Allocate_Resources: prompt~3364, completion~1087, total~4451

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-22 16:21:20
Total Users: 12
Average Resource Utilization: 23.46%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 62.44 Mbps, mMTC Total Rate: 37.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          6  6.0/30 MHz        20.00%
mMTC           5  4.5/10 MHz        45.00%

New User Allocation:
User 12 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4680, completion~1008, total~5688

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need to check weather forecasts"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1366, completion~1039, total~2405
[Token] Slice_Type_Determination: prompt~2930, completion~716, total~3646
[Token] Workload_Balance: prompt~3744, completion~457, total~4201
[Token] Bandwidth_Analysis: prompt~213, completion~678, total~891
[Token] Allocate_Resources: prompt~4341, completion~812, total~5153

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-22 16:22:41
Total Users: 13
Average Resource Utilization: 24.23%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 71.08 Mbps, mMTC Total Rate: 37.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          7  7.0/30 MHz        23.33%
mMTC           5  4.5/10 MHz        45.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 8.64 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5497, completion~1045, total~6542

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~1065, total~2435
[Token] Slice_Type_Determination: prompt~3001, completion~465, total~3466
[Token] Bandwidth_Analysis: prompt~215, completion~620, total~835
[Token] Allocate_Resources: prompt~3595, completion~812, total~4407

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-22 16:23:47
Total Users: 14
Average Resource Utilization: 25.0%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 80.59 Mbps, mMTC Total Rate: 37.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          8  8.0/30 MHz        26.67%
mMTC           5  4.5/10 MHz        45.00%

New User Allocation:
User 14 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4674, completion~1050, total~5724

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use maps for basic navigation"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~1208, total~2578
[Token] Slice_Type_Determination: prompt~3122, completion~957, total~4079
WARNING: LLM didn't provide explicit slice recommendation. Using extracted recommendation: eMBB
[Token] Bandwidth_Analysis: prompt~215, completion~382, total~597
[Token] Allocate_Resources: prompt~4225, completion~1081, total~5306

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-22 16:25:05
Total Users: 15
Average Resource Utilization: 40.38%
eMBB Total Rate: 363.08 Mbps, URLLC Total Rate: 80.59 Mbps, mMTC Total Rate: 37.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          8  8.0/30 MHz        26.67%
mMTC           5  4.5/10 MHz        45.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 9, Bandwidth: 20.0 MHz, Rate: 190.30 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5555, completion~1120, total~6675

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need to stream 8K video content"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1265, total~2637
[Token] Slice_Type_Determination: prompt~3198, completion~386, total~3584
[Token] Bandwidth_Analysis: prompt~216, completion~341, total~557
[Token] Allocate_Resources: prompt~3733, completion~861, total~4594

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-22 16:26:08
Total Users: 16
Average Resource Utilization: 55.77%
eMBB Total Rate: 608.39 Mbps, URLLC Total Rate: 80.59 Mbps, mMTC Total Rate: 37.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          8  8.0/30 MHz        26.67%
mMTC           5  4.5/10 MHz        45.00%

New User Allocation:
User 16 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 245.31 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       20   |        245.31 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4846, completion~1374, total~6220

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need instant alerts for life-threatening patient conditions"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1200, total~2572
[Token] Slice_Type_Determination: prompt~3128, completion~352, total~3480
[Token] Bandwidth_Analysis: prompt~216, completion~573, total~789
[Token] Allocate_Resources: prompt~3610, completion~894, total~4504

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-22 16:27:17
Total Users: 17
Average Resource Utilization: 59.62%
eMBB Total Rate: 608.39 Mbps, URLLC Total Rate: 119.54 Mbps, mMTC Total Rate: 37.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          9  13.0/30 MHz       43.33%
mMTC           5  4.5/10 MHz        45.00%

New User Allocation:
User 17 → URLLC Slice
CQI: 7, Bandwidth: 5.0 MHz, Rate: 38.95 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4778, completion~1365, total~6143

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "My smart meter needs to report its reading"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~1427, total~2797
[Token] Slice_Type_Determination: prompt~3324, completion~330, total~3654
WARNING: LLM didn't provide explicit slice recommendation. Using extracted recommendation: mMTC
[Token] Bandwidth_Analysis: prompt~214, completion~613, total~827
[Token] Allocate_Resources: prompt~3813, completion~1507, total~5320

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-22 16:28:37
Total Users: 18
Average Resource Utilization: 60.31%
eMBB Total Rate: 608.39 Mbps, URLLC Total Rate: 119.54 Mbps, mMTC Total Rate: 45.15 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          9  13.0/30 MHz       43.33%
mMTC           6  5.4/10 MHz        54.00%

New User Allocation:
User 18 → mMTC Slice
CQI: 8, Bandwidth: 0.9 MHz, Rate: 7.78 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5566, completion~1135, total~6701

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to control a robotic arm in real time"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~1202, total~2576
[Token] Slice_Type_Determination: prompt~3133, completion~249, total~3382
[Token] Bandwidth_Analysis: prompt~217, completion~501, total~718
[Token] Allocate_Resources: prompt~3509, completion~724, total~4233

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-22 16:29:41
Total Users: 19
Average Resource Utilization: 61.08%
eMBB Total Rate: 608.39 Mbps, URLLC Total Rate: 127.33 Mbps, mMTC Total Rate: 45.15 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC         10  14.0/30 MHz       46.67%
mMTC           6  5.4/10 MHz        54.00%

New User Allocation:
User 19 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4488, completion~831, total~5319

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I want to watch 4K video"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~1017, total~2387
[Token] Slice_Type_Determination: prompt~2926, completion~300, total~3226
[Token] Bandwidth_Analysis: prompt~215, completion~704, total~919
[Token] Allocate_Resources: prompt~3373, completion~949, total~4322

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-22 16:30:42
Total Users: 20
Average Resource Utilization: 76.46%
eMBB Total Rate: 835.03 Mbps, URLLC Total Rate: 127.33 Mbps, mMTC Total Rate: 45.15 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC         10  14.0/30 MHz       46.67%
mMTC           6  5.4/10 MHz        54.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 11, Bandwidth: 20.0 MHz, Rate: 226.64 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       20   |        226.64 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4584, completion~876, total~5460

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to stream music while browsing social media"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1240, total~2612
[Token] Slice_Type_Determination: prompt~3153, completion~278, total~3431
[Token] Bandwidth_Analysis: prompt~216, completion~460, total~676
[Token] Allocate_Resources: prompt~3627, completion~872, total~4499

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-22 16:31:47
Total Users: 21
Average Resource Utilization: 84.15%
eMBB Total Rate: 957.69 Mbps, URLLC Total Rate: 127.33 Mbps, mMTC Total Rate: 45.15 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         10  14.0/30 MHz       46.67%
mMTC           6  5.4/10 MHz        54.00%

New User Allocation:
User 21 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 245.31 Mbps, Latency: 50.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 10.0 MHz
  User 16: 20.0 → 10.0 MHz, Rate: 245.31 → 122.66 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       20   |        226.64 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4834, completion~971, total~5805

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "My asset tracking device needs to send location update"
CQI: 12
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1057, total~2429
[Token] Slice_Type_Determination: prompt~2953, completion~318, total~3271
[Token] Bandwidth_Analysis: prompt~215, completion~469, total~684
[Token] Allocate_Resources: prompt~3428, completion~2475, total~5903

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-22 16:33:18
Total Users: 22
Average Resource Utilization: 84.85%
eMBB Total Rate: 957.69 Mbps, URLLC Total Rate: 127.33 Mbps, mMTC Total Rate: 56.19 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         10  14.0/30 MHz               46.67%
mMTC           7  6.300000000000001/10 MHz  63.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 12, Bandwidth: 0.9 MHz, Rate: 11.04 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       20   |        226.64 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~6164, completion~1385, total~7549

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~1295, total~2669
[Token] Slice_Type_Determination: prompt~3222, completion~248, total~3470
[Token] Bandwidth_Analysis: prompt~221, completion~802, total~1023
[Token] Allocate_Resources: prompt~3624, completion~968, total~4592

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-22 16:34:35
Total Users: 23
Average Resource Utilization: 85.54%
eMBB Total Rate: 957.69 Mbps, URLLC Total Rate: 127.33 Mbps, mMTC Total Rate: 69.81 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         10  14.0/30 MHz               46.67%
mMTC           8  7.200000000000001/10 MHz  72.00%

New User Allocation:
User 23 → mMTC Slice
CQI: 15, Bandwidth: 0.9 MHz, Rate: 13.62 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       20   |        226.64 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4845, completion~1322, total~6167

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to deploy early warning systems for natural disasters"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~1220, total~2594
[Token] Slice_Type_Determination: prompt~3145, completion~247, total~3392
[Token] Bandwidth_Analysis: prompt~217, completion~440, total~657
[Token] Allocate_Resources: prompt~3517, completion~785, total~4302

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-22 16:35:40
Total Users: 24
Average Resource Utilization: 86.31%
eMBB Total Rate: 957.69 Mbps, URLLC Total Rate: 135.97 Mbps, mMTC Total Rate: 69.81 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         11  15.0/30 MHz               50.00%
mMTC           8  7.200000000000001/10 MHz  72.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 8.64 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       20   |        226.64 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4565, completion~970, total~5535

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I want to update my social media status"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~1009, total~2379
[Token] Slice_Type_Determination: prompt~2916, completion~666, total~3582
WARNING: LLM didn't provide explicit slice recommendation. Using extracted recommendation: eMBB
[Token] Bandwidth_Analysis: prompt~215, completion~639, total~854
[Token] Allocate_Resources: prompt~3810, completion~1109, total~4919

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-22 16:36:54
Total Users: 25
Average Resource Utilization: 86.31%
eMBB Total Rate: 876.58 Mbps, URLLC Total Rate: 135.97 Mbps, mMTC Total Rate: 69.81 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC         11  15.0/30 MHz               50.00%
mMTC           8  7.200000000000001/10 MHz  72.00%

New User Allocation:
User 25 → eMBB Slice
CQI: 7, Bandwidth: 20.0 MHz, Rate: 155.80 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 20.0 MHz
  User 21: 20.0 → 9.0 MHz, Rate: 245.31 → 110.39 Mbps, User 20: 20.0 → 11.0 MHz, Rate: 226.64 → 124.65 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       11   |        124.65 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5297, completion~1044, total~6341

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1378, completion~933, total~2311
[Token] Slice_Type_Determination: prompt~2863, completion~258, total~3121
[Token] Bandwidth_Analysis: prompt~219, completion~362, total~581
[Token] Allocate_Resources: prompt~3250, completion~725, total~3975

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-22 16:37:49
Total Users: 26
Average Resource Utilization: 87.08%
eMBB Total Rate: 876.58 Mbps, URLLC Total Rate: 145.48 Mbps, mMTC Total Rate: 69.81 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC         12  16.0/30 MHz               53.33%
mMTC           8  7.200000000000001/10 MHz  72.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       11   |        124.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4242, completion~1181, total~5423

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need to deploy early warning systems for natural disasters"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~1349, total~2723
[Token] Slice_Type_Determination: prompt~3293, completion~246, total~3539
[Token] Bandwidth_Analysis: prompt~217, completion~812, total~1029
[Token] Allocate_Resources: prompt~3667, completion~873, total~4540

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-22 16:39:09
Total Users: 27
Average Resource Utilization: 90.92%
eMBB Total Rate: 876.58 Mbps, URLLC Total Rate: 193.05 Mbps, mMTC Total Rate: 69.81 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC         13  21.0/30 MHz               70.00%
mMTC           8  7.200000000000001/10 MHz  72.00%

New User Allocation:
User 27 → URLLC Slice
CQI: 9, Bandwidth: 5.0 MHz, Rate: 47.57 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        5   |         47.57 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       11   |        124.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4823, completion~815, total~5638

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1378, completion~1110, total~2488
[Token] Slice_Type_Determination: prompt~3009, completion~221, total~3230
[Token] Bandwidth_Analysis: prompt~234, completion~452, total~686
[Token] Allocate_Resources: prompt~3385, completion~1119, total~4504

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-22 16:40:21
Total Users: 28
Average Resource Utilization: 91.62%
eMBB Total Rate: 876.58 Mbps, URLLC Total Rate: 193.05 Mbps, mMTC Total Rate: 77.59 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC         13  21.0/30 MHz               70.00%
mMTC           9  8.100000000000001/10 MHz  81.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 8, Bandwidth: 0.9 MHz, Rate: 7.78 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       11   |        124.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     8 |        0.9 |          7.78 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4755, completion~1120, total~5875

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need to check the status of my smart home sensors"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1376, completion~1105, total~2481
[Token] Slice_Type_Determination: prompt~3001, completion~232, total~3233
[Token] Bandwidth_Analysis: prompt~237, completion~395, total~632
[Token] Allocate_Resources: prompt~3384, completion~1013, total~4397

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-22 16:41:25
Total Users: 29
Average Resource Utilization: 92.31%
eMBB Total Rate: 876.58 Mbps, URLLC Total Rate: 193.05 Mbps, mMTC Total Rate: 85.37 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC         13  21.0/30 MHz               70.00%
mMTC          10  9.000000000000002/10 MHz  90.00%

New User Allocation:
User 29 → mMTC Slice
CQI: 8, Bandwidth: 0.9 MHz, Rate: 7.78 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       11   |        124.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | mMTC    |     8 |        0.9 |          7.78 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4649, completion~1059, total~5708

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1380, completion~1173, total~2553
[Token] Slice_Type_Determination: prompt~3116, completion~340, total~3456
[Token] Bandwidth_Analysis: prompt~220, completion~528, total~748
[Token] Allocate_Resources: prompt~3581, completion~879, total~4460

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-22 16:42:30
Total Users: 30
Average Resource Utilization: 93.08%
eMBB Total Rate: 876.58 Mbps, URLLC Total Rate: 200.84 Mbps, mMTC Total Rate: 85.37 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC         14  22.0/30 MHz               73.33%
mMTC          10  9.000000000000002/10 MHz  90.00%

New User Allocation:
User 30 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | URLLC   |     7 |        1   |          7.79 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       11   |        124.65 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        0.9 |          7.78 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4727, completion~1053, total~5780

Detailed Slice Utilization Values:
eMBB utils: [22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 44.44, 66.67, 66.67, 66.67, 66.67, 88.89, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 3.33, 6.67, 6.67, 10.0, 10.0, 13.33, 16.67, 16.67, 16.67, 20.0, 20.0, 23.33, 26.67, 26.67, 26.67, 43.33, 43.33, 46.67, 46.67, 46.67, 46.67, 46.67, 50.0, 50.0, 53.33, 70.0, 70.0, 70.0, 73.33]
mMTC utils: [0.0, 0.0, 0.0, 9.0, 9.0, 18.0, 18.0, 18.0, 27.0, 36.0, 36.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 54.0, 54.0, 54.0, 54.0, 63.0, 72.0, 72.0, 72.0, 72.0, 72.0, 81.0, 90.0, 90.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |     8 |       20   |        172.78 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | URLLC   | URLLC          | Yes            |     5 |        1   |          6.19 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | URLLC   | URLLC          | Yes            |    11 |        1   |         11.33 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | mMTC    | mMTC           | Yes            |     8 |        0.9 |          7.78 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | URLLC   | URLLC          | Yes            |    14 |        1   |         14.17 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |     6 |        1   |          6.97 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | URLLC          | Yes            |     8 |        1   |          8.64 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | mMTC    | mMTC           | Yes            |     9 |        0.9 |          8.56 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |    15 |        1   |         15.14 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | eMBB           | No             |     8 |        1   |          8.64 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | URLLC   | URLLC          | Yes            |     9 |        1   |          9.51 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |     9 |       20   |        190.3  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | URLLC   | URLLC          | Yes            |     7 |        5   |         38.95 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | mMTC    | mMTC           | Yes            |     8 |        0.9 |          7.78 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | URLLC   | URLLC          | Yes            |     7 |        1   |          7.79 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    11 |       20   |        226.64 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             50 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | mMTC    | mMTC           | Yes            |    12 |        0.9 |         11.04 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | mMTC    | mMTC           | Yes            |    15 |        0.9 |         13.62 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     8 |        1   |          8.64 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | URLLC          | Yes            |     9 |        1   |          9.51 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | URLLC   | URLLC          | Yes            |     9 |        5   |         47.57 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | mMTC    | mMTC           | Yes            |     8 |        0.9 |          7.78 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | mMTC    | mMTC           | Yes            |     8 |        0.9 |          7.78 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Success  | URLLC   | URLLC          | Yes            |     7 |        1   |          7.79 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 30/30 (100.0%)

Intent Understanding Evaluation:
Correctly identified intents: 29/30
Intent understanding rate: 96.7%

Workload Balancing Statistics:
Users with workload balancing: 0/30
Workload balancing rate: 0.0%

Slice Utilization Statistics:
Average eMBB utilization: 57.04%
Average URLLC utilization: 33.33%
Average mMTC utilization: 44.70%

Weighted Average Utilization: 50.62%

Transmission Rate Statistics:
Final eMBB total rate: 876.58 Mbps
Final URLLC total rate: 200.84 Mbps
Final mMTC total rate: 85.37 Mbps

Resource Utilization:
Average resource utilization: 93.08%

Results exported to F:\code\wirelessagent\run_results\batch_run\nkb\kimi-k2.5\network_slicing_results_TJU_north_kimi-k2.5.csv

✓ TJU_north 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\nkb\kimi-k2.5\network_slicing_results_TJU_north_kimi-k2.5.csv