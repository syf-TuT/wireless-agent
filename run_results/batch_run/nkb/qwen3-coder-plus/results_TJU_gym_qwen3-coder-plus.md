============================================================
场景 4/5: TJU_gym
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_gym.csv
输出: F:\code\wirelessagent\run_results\batch_run\nkb\qwen3-coder-plus\network_slicing_results_TJU_gym_qwen3-coder-plus.csv
------------------------------------------------------------
Starting network slice management system with CSV-based user testing...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~1077, total~2451
[Token] Slice_Type_Determination: prompt~2994, completion~254, total~3248
[Token] Bandwidth_Analysis: prompt~214, completion~731, total~945
[Token] Allocate_Resources: prompt~3401, completion~1366, total~4767

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-21 21:22:26
Total Users: 1
Average Resource Utilization: 0.69%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 4.29 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          0  0/30 MHz          0%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 1 → mMTC Slice
CQI: 3, Bandwidth: 0.9 MHz, Rate: 4.29 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5017, completion~1070, total~6087

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I want to browse websites and check email"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~1194, total~2564
[Token] Slice_Type_Determination: prompt~3109, completion~489, total~3598
[Token] Bandwidth_Analysis: prompt~213, completion~425, total~638
[Token] Allocate_Resources: prompt~3744, completion~863, total~4607

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-21 21:23:28
Total Users: 2
Average Resource Utilization: 16.08%
eMBB Total Rate: 226.64 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 4.29 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          0  0/30 MHz          0%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 2 → eMBB Slice
CQI: 11, Bandwidth: 20.0 MHz, Rate: 226.64 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4861, completion~956, total~5817

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to monitor water level in a reservoir"
CQI: 13
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~976, total~2348
[Token] Slice_Type_Determination: prompt~2876, completion~202, total~3078
[Token] Bandwidth_Analysis: prompt~215, completion~502, total~717
[Token] Allocate_Resources: prompt~3235, completion~1053, total~4288

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-21 21:24:24
Total Users: 3
Average Resource Utilization: 16.77%
eMBB Total Rate: 226.64 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 16.18 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          0  0/30 MHz          0%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 3 → mMTC Slice
CQI: 13, Bandwidth: 0.9 MHz, Rate: 11.89 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4525, completion~878, total~5403

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1436, total~2808
[Token] Slice_Type_Determination: prompt~3341, completion~350, total~3691
[Token] Bandwidth_Analysis: prompt~215, completion~696, total~911
[Token] Allocate_Resources: prompt~3843, completion~998, total~4841

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-21 21:25:31
Total Users: 4
Average Resource Utilization: 17.46%
eMBB Total Rate: 226.64 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 21.09 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          0  0/30 MHz          0%
mMTC           3  2.7/10 MHz        27.00%

New User Allocation:
User 4 → mMTC Slice
CQI: 4, Bandwidth: 0.9 MHz, Rate: 4.91 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5090, completion~1144, total~6234

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1376, completion~1054, total~2430
[Token] Slice_Type_Determination: prompt~2970, completion~652, total~3622
[Token] Bandwidth_Analysis: prompt~218, completion~443, total~661
[Token] Allocate_Resources: prompt~3778, completion~1047, total~4825

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-21 21:26:35
Total Users: 5
Average Resource Utilization: 32.85%
eMBB Total Rate: 399.42 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 21.09 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          0  0/30 MHz          0%
mMTC           3  2.7/10 MHz        27.00%

New User Allocation:
User 5 → eMBB Slice
CQI: 8, Bandwidth: 20.0 MHz, Rate: 172.78 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5081, completion~1053, total~6134

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I need to stream 8K video content"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~833, total~2205
[Token] Slice_Type_Determination: prompt~2747, completion~519, total~3266
WARNING: LLM didn't provide explicit slice recommendation. Using extracted recommendation: eMBB
[Token] Bandwidth_Analysis: prompt~216, completion~393, total~609
[Token] Allocate_Resources: prompt~3419, completion~956, total~4375

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-21 21:27:29
Total Users: 6
Average Resource Utilization: 48.23%
eMBB Total Rate: 523.29 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 21.09 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          0  0/30 MHz          0%
mMTC           3  2.7/10 MHz        27.00%

New User Allocation:
User 6 → eMBB Slice
CQI: 5, Bandwidth: 20.0 MHz, Rate: 123.87 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4621, completion~850, total~5471

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to track the location of a shipping container"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~1110, total~2484
[Token] Slice_Type_Determination: prompt~3029, completion~316, total~3345
[Token] Workload_Balance: prompt~3445, completion~691, total~4136
[Token] Bandwidth_Analysis: prompt~215, completion~664, total~879
[Token] Allocate_Resources: prompt~4277, completion~784, total~5061

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-21 21:28:43
Total Users: 7
Average Resource Utilization: 49.0%
eMBB Total Rate: 523.29 Mbps, URLLC Total Rate: 6.97 Mbps, mMTC Total Rate: 21.09 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          1  1.0/30 MHz        3.33%
mMTC           3  2.7/10 MHz        27.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 6.97 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5412, completion~956, total~6368

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1137, total~2509
[Token] Slice_Type_Determination: prompt~3042, completion~309, total~3351
[Token] Workload_Balance: prompt~3448, completion~744, total~4192
[Token] Bandwidth_Analysis: prompt~216, completion~487, total~703
[Token] Allocate_Resources: prompt~4340, completion~794, total~5134

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-21 21:29:57
Total Users: 8
Average Resource Utilization: 49.77%
eMBB Total Rate: 523.29 Mbps, URLLC Total Rate: 22.11 Mbps, mMTC Total Rate: 21.09 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          2  2.0/30 MHz        6.67%
mMTC           3  2.7/10 MHz        27.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 15.14 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5480, completion~1013, total~6493

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I need reliable connectivity for implanted medical devices"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~993, total~2363
[Token] Slice_Type_Determination: prompt~2910, completion~252, total~3162
[Token] Bandwidth_Analysis: prompt~215, completion~666, total~881
[Token] Allocate_Resources: prompt~3284, completion~929, total~4213

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-21 21:30:58
Total Users: 9
Average Resource Utilization: 50.54%
eMBB Total Rate: 523.29 Mbps, URLLC Total Rate: 29.90 Mbps, mMTC Total Rate: 21.09 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          3  3.0/30 MHz        10.00%
mMTC           3  2.7/10 MHz        27.00%

New User Allocation:
User 9 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4488, completion~960, total~5448

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to check weather forecasts"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1366, completion~1088, total~2454
[Token] Slice_Type_Determination: prompt~2985, completion~481, total~3466
[Token] Bandwidth_Analysis: prompt~213, completion~362, total~575
[Token] Allocate_Resources: prompt~3616, completion~1051, total~4667

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-21 21:32:00
Total Users: 10
Average Resource Utilization: 65.92%
eMBB Total Rate: 647.16 Mbps, URLLC Total Rate: 29.90 Mbps, mMTC Total Rate: 21.09 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          3  3.0/30 MHz        10.00%
mMTC           3  2.7/10 MHz        27.00%

New User Allocation:
User 10 → eMBB Slice
CQI: 5, Bandwidth: 20.0 MHz, Rate: 123.87 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4927, completion~1079, total~6006

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1378, completion~929, total~2307
[Token] Slice_Type_Determination: prompt~2834, completion~220, total~3054
[Token] Bandwidth_Analysis: prompt~218, completion~643, total~861
[Token] Allocate_Resources: prompt~3207, completion~1119, total~4326

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-21 21:33:04
Total Users: 11
Average Resource Utilization: 66.62%
eMBB Total Rate: 647.16 Mbps, URLLC Total Rate: 29.90 Mbps, mMTC Total Rate: 27.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          3  3.0/30 MHz        10.00%
mMTC           4  3.6/10 MHz        36.00%

New User Allocation:
User 11 → mMTC Slice
CQI: 6, Bandwidth: 0.9 MHz, Rate: 6.28 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4570, completion~704, total~5274

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need microsecond-level latency for high-frequency tradin"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1376, completion~950, total~2326
[Token] Slice_Type_Determination: prompt~2885, completion~481, total~3366
[Token] Bandwidth_Analysis: prompt~218, completion~473, total~691
[Token] Allocate_Resources: prompt~3501, completion~782, total~4283

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-21 21:33:58
Total Users: 12
Average Resource Utilization: 67.38%
eMBB Total Rate: 647.16 Mbps, URLLC Total Rate: 34.66 Mbps, mMTC Total Rate: 27.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          4  4.0/30 MHz        13.33%
mMTC           4  3.6/10 MHz        36.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 4.76 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4547, completion~774, total~5321

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I want to use holographic communication"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1368, completion~1170, total~2538
[Token] Slice_Type_Determination: prompt~3095, completion~297, total~3392
WARNING: LLM didn't provide explicit slice recommendation. Using extracted recommendation: eMBB
[Token] Bandwidth_Analysis: prompt~214, completion~531, total~745
[Token] Allocate_Resources: prompt~3590, completion~997, total~4587

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-21 21:34:57
Total Users: 13
Average Resource Utilization: 75.08%
eMBB Total Rate: 724.14 Mbps, URLLC Total Rate: 34.66 Mbps, mMTC Total Rate: 27.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          4  4.0/30 MHz        13.33%
mMTC           4  3.6/10 MHz        36.00%

New User Allocation:
User 13 → eMBB Slice
CQI: 9, Bandwidth: 20.0 MHz, Rate: 190.30 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 10.0 MHz
  User 2: 20.0 → 10.0 MHz, Rate: 226.64 → 113.32 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4917, completion~987, total~5904

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 12
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~947, total~2319
[Token] Slice_Type_Determination: prompt~2869, completion~423, total~3292
[Token] Bandwidth_Analysis: prompt~216, completion~563, total~779
[Token] Allocate_Resources: prompt~3423, completion~784, total~4207

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-21 21:35:58
Total Users: 14
Average Resource Utilization: 75.85%
eMBB Total Rate: 724.14 Mbps, URLLC Total Rate: 46.93 Mbps, mMTC Total Rate: 27.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          5  5.0/30 MHz        16.67%
mMTC           4  3.6/10 MHz        36.00%

New User Allocation:
User 14 → URLLC Slice
CQI: 12, Bandwidth: 1.0 MHz, Rate: 12.27 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        1   |         12.27 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4471, completion~987, total~5458

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need real-time fraud detection for financial transactions"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1129, total~2501
[Token] Slice_Type_Determination: prompt~3063, completion~283, total~3346
[Token] Bandwidth_Analysis: prompt~216, completion~751, total~967
[Token] Allocate_Resources: prompt~3471, completion~891, total~4362

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-21 21:36:58
Total Users: 15
Average Resource Utilization: 76.62%
eMBB Total Rate: 724.14 Mbps, URLLC Total Rate: 54.72 Mbps, mMTC Total Rate: 27.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          6  6.0/30 MHz        20.00%
mMTC           4  3.6/10 MHz        36.00%

New User Allocation:
User 15 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        1   |         12.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        1   |          7.79 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4632, completion~937, total~5569

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need instant facial recognition for public security threats"
CQI: 10
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1218, total~2590
[Token] Slice_Type_Determination: prompt~3139, completion~430, total~3569
[Token] Bandwidth_Analysis: prompt~216, completion~385, total~601
[Token] Allocate_Resources: prompt~3701, completion~943, total~4644

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-21 21:38:00
Total Users: 16
Average Resource Utilization: 77.38%
eMBB Total Rate: 724.14 Mbps, URLLC Total Rate: 65.13 Mbps, mMTC Total Rate: 27.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          7  7.0/30 MHz        23.33%
mMTC           4  3.6/10 MHz        36.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 10, Bandwidth: 1.0 MHz, Rate: 10.41 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        1   |         12.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4901, completion~1112, total~6013

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "My smart trash can needs to signal that it's full"
CQI: 13
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1376, completion~1038, total~2414
[Token] Slice_Type_Determination: prompt~2949, completion~376, total~3325
WARNING: LLM didn't provide explicit slice recommendation. Using extracted recommendation: mMTC
[Token] Bandwidth_Analysis: prompt~217, completion~401, total~618
[Token] Allocate_Resources: prompt~3479, completion~1217, total~4696

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-21 21:39:06
Total Users: 17
Average Resource Utilization: 78.08%
eMBB Total Rate: 724.14 Mbps, URLLC Total Rate: 65.13 Mbps, mMTC Total Rate: 39.26 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          7  7.0/30 MHz        23.33%
mMTC           5  4.5/10 MHz        45.00%

New User Allocation:
User 17 → mMTC Slice
CQI: 13, Bandwidth: 0.9 MHz, Rate: 11.89 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        1   |         12.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4932, completion~886, total~5818

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~1255, total~2625
[Token] Slice_Type_Determination: prompt~3197, completion~276, total~3473
[Token] Bandwidth_Analysis: prompt~215, completion~622, total~837
[Token] Allocate_Resources: prompt~3594, completion~800, total~4394

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-21 21:40:06
Total Users: 18
Average Resource Utilization: 78.85%
eMBB Total Rate: 724.14 Mbps, URLLC Total Rate: 69.89 Mbps, mMTC Total Rate: 39.26 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          8  8.0/30 MHz        26.67%
mMTC           5  4.5/10 MHz        45.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 4.76 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        1   |         12.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        1   |          4.76 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4657, completion~1036, total~5693

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to monitor water level in a reservoir"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1239, total~2611
[Token] Slice_Type_Determination: prompt~3148, completion~289, total~3437
WARNING: LLM didn't provide explicit slice recommendation. Using extracted recommendation: mMTC
[Token] Bandwidth_Analysis: prompt~215, completion~661, total~876
[Token] Allocate_Resources: prompt~3590, completion~1014, total~4604

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-21 21:41:16
Total Users: 19
Average Resource Utilization: 79.54%
eMBB Total Rate: 724.14 Mbps, URLLC Total Rate: 69.89 Mbps, mMTC Total Rate: 43.55 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          8  8.0/30 MHz        26.67%
mMTC           6  5.4/10 MHz        54.00%

New User Allocation:
User 19 → mMTC Slice
CQI: 3, Bandwidth: 0.9 MHz, Rate: 4.29 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        1   |         12.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4850, completion~1035, total~5885

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I need to make a high-quality voice call"
CQI: 10
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1406, total~2778
[Token] Slice_Type_Determination: prompt~3336, completion~458, total~3794
[Token] Bandwidth_Analysis: prompt~216, completion~809, total~1025
[Token] Allocate_Resources: prompt~3925, completion~858, total~4783

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-21 21:42:30
Total Users: 20
Average Resource Utilization: 80.31%
eMBB Total Rate: 724.14 Mbps, URLLC Total Rate: 80.30 Mbps, mMTC Total Rate: 43.55 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          9  9.0/30 MHz        30.00%
mMTC           6  5.4/10 MHz        54.00%

New User Allocation:
User 20 → URLLC Slice
CQI: 10, Bandwidth: 1.0 MHz, Rate: 10.41 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        1   |         12.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |         10.41 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5051, completion~966, total~6017

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to use holographic communication"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1368, completion~1159, total~2527
[Token] Slice_Type_Determination: prompt~3081, completion~356, total~3437
[Token] Bandwidth_Analysis: prompt~214, completion~587, total~801
[Token] Allocate_Resources: prompt~3701, completion~774, total~4475

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-21 21:43:32
Total Users: 21
Average Resource Utilization: 80.31%
eMBB Total Rate: 796.12 Mbps, URLLC Total Rate: 80.30 Mbps, mMTC Total Rate: 43.55 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          9  9.0/30 MHz        30.00%
mMTC           6  5.4/10 MHz        54.00%

New User Allocation:
User 21 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 245.31 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 3, Bandwidth freed: 20.0 MHz
  User 13: 20.0 → 11.0 MHz, Rate: 190.30 → 104.66 Mbps, User 5: 20.0 → 12.0 MHz, Rate: 172.78 → 103.67 Mbps, User 6: 20.0 → 17.0 MHz, Rate: 123.87 → 105.29 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        1   |         12.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4882, completion~1277, total~6159

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "I need to check the status of my smart home sensors"
CQI: 11
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1376, completion~1072, total~2448
[Token] Slice_Type_Determination: prompt~2965, completion~449, total~3414
WARNING: LLM didn't provide explicit slice recommendation. Using extracted recommendation: eMBB
[Token] Bandwidth_Analysis: prompt~218, completion~620, total~838
[Token] Allocate_Resources: prompt~3610, completion~871, total~4481
[Token] Failure_Evaluation: prompt~4567, completion~559, total~5126

----------------------------------------
ALLOCATION FAILED FOR USER 22
----------------------------------------
Request: I need to check the status of my smart home sensors
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 1
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1374, completion~1254, total~2628
[Token] Slice_Type_Determination: prompt~3191, completion~218, total~3409
[Token] Bandwidth_Analysis: prompt~217, completion~698, total~915
[Token] Allocate_Resources: prompt~3535, completion~918, total~4453

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-21 21:45:42
Total Users: 22
Average Resource Utilization: 81.08%
eMBB Total Rate: 796.12 Mbps, URLLC Total Rate: 83.84 Mbps, mMTC Total Rate: 43.55 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         10  10.0/30 MHz       33.33%
mMTC           6  5.4/10 MHz        54.00%

New User Allocation:
User 23 → URLLC Slice
CQI: 1, Bandwidth: 1.0 MHz, Rate: 3.54 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        1   |         12.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        1   |          3.54 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4727, completion~1125, total~5852

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I want to use remote surgery equipment"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1368, completion~987, total~2355
[Token] Slice_Type_Determination: prompt~2905, completion~325, total~3230
[Token] Bandwidth_Analysis: prompt~214, completion~382, total~596
[Token] Allocate_Resources: prompt~3363, completion~1036, total~4399

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-21 21:46:45
Total Users: 23
Average Resource Utilization: 81.85%
eMBB Total Rate: 796.12 Mbps, URLLC Total Rate: 90.03 Mbps, mMTC Total Rate: 43.55 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         11  11.0/30 MHz       36.67%
mMTC           6  5.4/10 MHz        54.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 5, Bandwidth: 1.0 MHz, Rate: 6.19 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        1   |         12.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        1   |          3.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4665, completion~901, total~5566

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I want to stream music while browsing social media"
CQI: 2
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1134, total~2506
[Token] Slice_Type_Determination: prompt~3060, completion~369, total~3429
[Token] Bandwidth_Analysis: prompt~216, completion~501, total~717
[Token] Allocate_Resources: prompt~3623, completion~1044, total~4667
[Token] Failure_Evaluation: prompt~4779, completion~707, total~5486

----------------------------------------
ALLOCATION FAILED FOR USER 25
----------------------------------------
Request: I want to stream music while browsing social media
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1380, completion~1071, total~2451
[Token] Slice_Type_Determination: prompt~2989, completion~278, total~3267
[Token] Bandwidth_Analysis: prompt~220, completion~471, total~691
[Token] Allocate_Resources: prompt~3397, completion~952, total~4349

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-21 21:48:54
Total Users: 24
Average Resource Utilization: 82.62%
eMBB Total Rate: 796.12 Mbps, URLLC Total Rate: 97.00 Mbps, mMTC Total Rate: 43.55 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         12  12.0/30 MHz       40.00%
mMTC           6  5.4/10 MHz        54.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 6.97 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        1   |         12.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        1   |          3.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        1   |          6.97 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4614, completion~832, total~5446

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need to transmit real-time patient vital signs during critical care"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1378, completion~1263, total~2641
[Token] Slice_Type_Determination: prompt~3201, completion~399, total~3600
[Token] Bandwidth_Analysis: prompt~219, completion~699, total~918
[Token] Allocate_Resources: prompt~3728, completion~774, total~4502

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-21 21:49:59
Total Users: 25
Average Resource Utilization: 84.92%
eMBB Total Rate: 796.12 Mbps, URLLC Total Rate: 111.29 Mbps, mMTC Total Rate: 43.55 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         13  15.0/30 MHz       50.00%
mMTC           6  5.4/10 MHz        54.00%

New User Allocation:
User 27 → URLLC Slice
CQI: 3, Bandwidth: 3.0 MHz, Rate: 14.29 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        1   |         12.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        1   |          3.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        3   |         14.29 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4771, completion~1149, total~5920

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~1099, total~2469
[Token] Slice_Type_Determination: prompt~3031, completion~250, total~3281
[Token] Bandwidth_Analysis: prompt~215, completion~494, total~709
[Token] Allocate_Resources: prompt~3410, completion~941, total~4351

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-21 21:51:06
Total Users: 26
Average Resource Utilization: 85.69%
eMBB Total Rate: 796.12 Mbps, URLLC Total Rate: 116.75 Mbps, mMTC Total Rate: 43.55 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         14  16.0/30 MHz       53.33%
mMTC           6  5.4/10 MHz        54.00%

New User Allocation:
User 28 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        1   |         12.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        1   |          3.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | URLLC   |     4 |        1   |          5.46 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4616, completion~884, total~5500

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need to participate in an online multiplayer game"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1372, completion~1284, total~2656
[Token] Slice_Type_Determination: prompt~3194, completion~457, total~3651
[Token] Bandwidth_Analysis: prompt~216, completion~440, total~656
[Token] Allocate_Resources: prompt~3775, completion~885, total~4660

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-21 21:52:13
Total Users: 27
Average Resource Utilization: 86.46%
eMBB Total Rate: 796.12 Mbps, URLLC Total Rate: 131.89 Mbps, mMTC Total Rate: 43.55 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         15  17.0/30 MHz       56.67%
mMTC           6  5.4/10 MHz        54.00%

New User Allocation:
User 29 → URLLC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 15.14 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        1   |         12.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        1   |          3.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | URLLC   |    15 |        1   |         15.14 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       10   |        113.32 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4918, completion~1029, total~5947

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I want to browse websites and check email"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Intent_Analysis: prompt~1370, completion~1353, total~2723
[Token] Slice_Type_Determination: prompt~3274, completion~322, total~3596
WARNING: LLM didn't provide explicit slice recommendation. Using extracted recommendation: eMBB
[Token] Bandwidth_Analysis: prompt~215, completion~482, total~697
[Token] Allocate_Resources: prompt~3794, completion~999, total~4793
[Token] Failure_Evaluation: prompt~4897, completion~633, total~5530

----------------------------------------
ALLOCATION FAILED FOR USER 30
----------------------------------------
Request: I want to browse websites and check email
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

Detailed Slice Utilization Values:
eMBB utils: [0.0, 22.22, 22.22, 22.22, 44.44, 66.67, 66.67, 66.67, 66.67, 88.89, 88.89, 88.89, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.33, 6.67, 10.0, 10.0, 10.0, 13.33, 13.33, 16.67, 20.0, 23.33, 23.33, 26.67, 26.67, 30.0, 30.0, 33.33, 36.67, 40.0, 50.0, 53.33, 56.67]
mMTC utils: [9.0, 9.0, 18.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 27.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 45.0, 45.0, 54.0, 54.0, 54.0, 54.0, 54.0, 54.0, 54.0, 54.0, 54.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | mMTC    | mMTC           | Yes            |     3 |        0.9 |          4.29 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | eMBB    | eMBB           | Yes            |    11 |       20   |        226.64 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | mMTC    | mMTC           | Yes            |    13 |        0.9 |         11.89 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | mMTC    | mMTC           | Yes            |     4 |        0.9 |          4.91 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | eMBB    | eMBB           | Yes            |     8 |       20   |        172.78 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | eMBB    | eMBB           | Yes            |     5 |       20   |        123.87 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | mMTC           | No             |     6 |        1   |          6.97 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | mMTC           | No             |    15 |        1   |         15.14 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | URLLC   | URLLC          | Yes            |     7 |        1   |          7.79 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | eMBB    | eMBB           | Yes            |     5 |       20   |        123.87 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | mMTC    | mMTC           | Yes            |     6 |        0.9 |          6.28 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     3 |        1   |          4.76 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | eMBB    | eMBB           | Yes            |     9 |       20   |        190.3  |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | URLLC   | URLLC          | Yes            |    12 |        1   |         12.27 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | URLLC   | URLLC          | Yes            |     7 |        1   |          7.79 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |    10 |        1   |         10.41 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | mMTC    | mMTC           | Yes            |    13 |        0.9 |         11.89 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     3 |        1   |          4.76 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | mMTC    | mMTC           | Yes            |     3 |        0.9 |          4.29 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | URLLC   | eMBB           | No             |    10 |        1   |         10.41 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Failed   | eMBB    | mMTC           | No             |    11 |       20   |        226.64 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | URLLC   | URLLC          | Yes            |     1 |        1   |          3.54 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     5 |        1   |          6.19 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Failed   | eMBB    | eMBB           | Yes            |     2 |       20   |         82.49 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | URLLC          | Yes            |     6 |        1   |          6.97 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | URLLC   | URLLC          | Yes            |     3 |        3   |         14.29 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          5.46 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | URLLC   | URLLC          | Yes            |    15 |        1   |         15.14 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Failed   | eMBB    | eMBB           | Yes            |     8 |       20   |        172.78 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 27/30 (90.0%)

Intent Understanding Evaluation:
Correctly identified intents: 26/30
Intent understanding rate: 86.7%

Workload Balancing Statistics:
Users with workload balancing: 0/30
Workload balancing rate: 0.0%

Slice Utilization Statistics:
Average eMBB utilization: 79.42%
Average URLLC utilization: 19.75%
Average mMTC utilization: 37.67%

Weighted Average Utilization: 62.44%

Transmission Rate Statistics:
Final eMBB total rate: 796.12 Mbps
Final URLLC total rate: 131.89 Mbps
Final mMTC total rate: 43.55 Mbps

Resource Utilization:
Average resource utilization: 86.46%

Results exported to F:\code\wirelessagent\run_results\batch_run\nkb\qwen3-coder-plus\network_slicing_results_TJU_gym_qwen3-coder-plus.csv

✓ TJU_gym 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\nkb\qwen3-coder-plus\network_slicing_results_TJU_gym_qwen3-coder-plus.csv