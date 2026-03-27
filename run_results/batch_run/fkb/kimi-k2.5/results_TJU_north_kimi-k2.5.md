F:\code\wirelessagent\.venv\Scripts\python.exe F:\code\wirelessagent\run_all_scenarios_fkb.py 
============================================================
开始运行所有场景... (模型: kimi-k2.5)
============================================================

============================================================
场景 1/5: TJU_north
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_north.csv
输出: F:\code\wirelessagent\run_results\batch_run\fkb\kimi-k2.5\network_slicing_results_TJU_north_kimi-k2.5.csv
------------------------------------------------------------
Starting network slice management system with CSV-based user testing...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I want to use augmented reality navigation"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use augmented reality navigation", "embb"),)
[Token] Intent_Analysis: prompt~1368, completion~701, total~2069
[Token] Slice_Type_Determination: prompt~2530, completion~110, total~2640
[Token] Beamforming_Bandwidth: prompt~212, completion~253, total~465
[Token] Allocate_Resources: prompt~2787, completion~352, total~3139

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-16 11:03:33
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
[Token] Network_Evaluation: prompt~3369, completion~584, total~3953

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~660, total~2032
[Token] Slice_Type_Determination: prompt~2499, completion~121, total~2620
[Token] Beamforming_Bandwidth: prompt~214, completion~305, total~519
[Token] Allocate_Resources: prompt~2738, completion~305, total~3043

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-16 11:04:33
Total Users: 2
Average Resource Utilization: 19.23%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 30.97 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          1  5.0/30 MHz        16.67%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → URLLC Slice
CQI: 5, Bandwidth: 5.0 MHz, Rate: 30.97 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          5 |         30.97 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         20 |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3278, completion~609, total~3887

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 11
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to balance electrical load in real-time across microgrids", "urllc"),)
[Token] Intent_Analysis: prompt~1380, completion~784, total~2164
[Token] Slice_Type_Determination: prompt~2628, completion~122, total~2750
[Token] Beamforming_Bandwidth: prompt~220, completion~283, total~503
[Token] Allocate_Resources: prompt~2868, completion~280, total~3148

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-16 11:05:35
Total Users: 3
Average Resource Utilization: 22.31%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 76.30 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          2  9.0/30 MHz        30.00%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 3 → URLLC Slice
CQI: 11, Bandwidth: 4.0 MHz, Rate: 45.33 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          5 |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          4 |         45.33 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         20 |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3380, completion~514, total~3894

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of city-wide smart streetlights", "mmtc"),, ("i need to check the status of city-wide smart streetlights", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~710, total~2088
[Token] Slice_Type_Determination: prompt~2536, completion~86, total~2622
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2776, completion~538, total~3314

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-16 11:06:23
Total Users: 4
Average Resource Utilization: 23.77%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 76.30 Mbps, mMTC Total Rate: 16.41 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          2  9.0/30 MHz        30.00%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 4 → mMTC Slice
CQI: 8, Bandwidth: 1.9 MHz, Rate: 16.41 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3546, completion~704, total~4250

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I want to use remote surgery equipment"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to use remote surgery equipment", "urllc"),)
[Token] Intent_Analysis: prompt~1368, completion~764, total~2132
[Token] Slice_Type_Determination: prompt~2593, completion~77, total~2670
[Token] Beamforming_Bandwidth: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~2790, completion~352, total~3142

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-16 11:07:14
Total Users: 5
Average Resource Utilization: 27.62%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 147.15 Mbps, mMTC Total Rate: 16.41 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          3  14.0/30 MHz       46.67%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 5 → URLLC Slice
CQI: 14, Bandwidth: 5.0 MHz, Rate: 70.85 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |         70.85 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3379, completion~710, total~4089

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart parking sensor needs to report if the spot is free", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~660, total~2038
[Token] Slice_Type_Determination: prompt~2492, completion~73, total~2565
[Token] Beamforming_Bandwidth: prompt~218, completion~2, total~220
[Token] Allocate_Resources: prompt~2717, completion~627, total~3344

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-16 11:08:06
Total Users: 6
Average Resource Utilization: 28.31%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 147.15 Mbps, mMTC Total Rate: 23.42 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          3  14.0/30 MHz       46.67%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 6 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |         70.85 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3579, completion~845, total~4424

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~691, total~2063
[Token] Slice_Type_Determination: prompt~2528, completion~82, total~2610
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2727, completion~369, total~3096

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-16 11:08:50
Total Users: 7
Average Resource Utilization: 32.15%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 182.02 Mbps, mMTC Total Rate: 23.42 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          4  19.0/30 MHz       63.33%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 6, Bandwidth: 5.0 MHz, Rate: 34.87 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |         70.85 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3332, completion~698, total~4030

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I need to transmit real-time patient vital signs during critical care"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to transmit real-time patient vital signs during critical care", "urllc"),)
[Token] Intent_Analysis: prompt~1378, completion~886, total~2264
[Token] Slice_Type_Determination: prompt~2726, completion~98, total~2824
[Token] Beamforming_Bandwidth: prompt~219, completion~2, total~221
[Token] Allocate_Resources: prompt~2942, completion~355, total~3297

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-16 11:09:51
Total Users: 8
Average Resource Utilization: 35.23%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 216.58 Mbps, mMTC Total Rate: 23.42 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          5  23.0/30 MHz       76.67%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 8, Bandwidth: 4.0 MHz, Rate: 34.56 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |         70.85 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3536, completion~581, total~4117

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a fleet of delivery drones needs to send low-rate telemetry data", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~704, total~2082
[Token] Slice_Type_Determination: prompt~2533, completion~111, total~2644
[Token] Beamforming_Bandwidth: prompt~218, completion~286, total~504
[Token] Allocate_Resources: prompt~2798, completion~568, total~3366

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-16 11:10:53
Total Users: 9
Average Resource Utilization: 36.69%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 216.58 Mbps, mMTC Total Rate: 38.22 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           1  20.0/90 MHz               22.22%
URLLC          5  23.0/30 MHz               76.67%
mMTC           3  4.699999999999999/10 MHz  47.00%

New User Allocation:
User 9 → mMTC Slice
CQI: 7, Bandwidth: 1.9 MHz, Rate: 14.80 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |         70.85 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3605, completion~670, total~4275

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~557, total~1931
[Token] Slice_Type_Determination: prompt~2383, completion~96, total~2479
[Token] Beamforming_Bandwidth: prompt~220, completion~2, total~222
[Token] Allocate_Resources: prompt~2632, completion~567, total~3199

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-16 11:11:38
Total Users: 10
Average Resource Utilization: 38.15%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 216.58 Mbps, mMTC Total Rate: 56.30 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          5  23.0/30 MHz       76.67%
mMTC           4  6.6/10 MHz        66.00%

New User Allocation:
User 10 → mMTC Slice
CQI: 9, Bandwidth: 1.9 MHz, Rate: 18.08 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |         70.85 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3434, completion~743, total~4177

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to detect and isolate power grid faults instantly", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~532, total~1906
[Token] Slice_Type_Determination: prompt~2364, completion~86, total~2450
[Token] Beamforming_Bandwidth: prompt~217, completion~293, total~510
[Token] Allocate_Resources: prompt~2567, completion~291, total~2858

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-16 11:12:34
Total Users: 11
Average Resource Utilization: 42.0%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 292.26 Mbps, mMTC Total Rate: 56.30 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          6  28.0/30 MHz       93.33%
mMTC           4  6.6/10 MHz        66.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 15, Bandwidth: 5.0 MHz, Rate: 75.68 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |         70.85 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3097, completion~599, total~3696

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need to check the status of my smart home sensors"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of my smart home sensors", "mmtc"),, ("i need to check the status of my smart home sensors", "mmtc"),)
[Token] Intent_Analysis: prompt~1376, completion~789, total~2165
[Token] Slice_Type_Determination: prompt~2614, completion~71, total~2685
[Token] Beamforming_Bandwidth: prompt~222, completion~2, total~224
[Token] Allocate_Resources: prompt~2836, completion~586, total~3422

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-16 11:13:23
Total Users: 12
Average Resource Utilization: 43.46%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 292.26 Mbps, mMTC Total Rate: 71.10 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          6  28.0/30 MHz       93.33%
mMTC           5  8.5/10 MHz        85.00%

New User Allocation:
User 12 → mMTC Slice
CQI: 7, Bandwidth: 1.9 MHz, Rate: 14.80 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |         70.85 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3653, completion~698, total~4351

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need to check weather forecasts"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to check weather forecasts", "embb"),)
[Token] Intent_Analysis: prompt~1366, completion~758, total~2124
[Token] Slice_Type_Determination: prompt~2583, completion~86, total~2669
[Token] Beamforming_Bandwidth: prompt~213, completion~2, total~215
[Token] Allocate_Resources: prompt~2815, completion~561, total~3376

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-16 11:14:22
Total Users: 13
Average Resource Utilization: 58.85%
eMBB Total Rate: 345.56 Mbps, URLLC Total Rate: 292.26 Mbps, mMTC Total Rate: 71.10 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          6  28.0/30 MHz       93.33%
mMTC           5  8.5/10 MHz        85.00%

New User Allocation:
User 13 → eMBB Slice
CQI: 8, Bandwidth: 20.0 MHz, Rate: 172.78 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |         70.85 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3607, completion~635, total~4242

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~822, total~2192
[Token] Slice_Type_Determination: prompt~2656, completion~93, total~2749
[Token] Beamforming_Bandwidth: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~2865, completion~372, total~3237

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-16 11:15:13
Total Users: 14
Average Resource Utilization: 60.38%
eMBB Total Rate: 345.56 Mbps, URLLC Total Rate: 311.29 Mbps, mMTC Total Rate: 71.10 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          7  30.0/30 MHz       100.00%
mMTC           5  8.5/10 MHz        85.00%

New User Allocation:
User 14 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 19.03 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |         70.85 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3485, completion~811, total~4296

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use maps for basic navigation"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use maps for basic navigation", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~681, total~2051
[Token] Slice_Type_Determination: prompt~2513, completion~84, total~2597
[Token] Beamforming_Bandwidth: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~2741, completion~511, total~3252

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-16 11:16:07
Total Users: 15
Average Resource Utilization: 75.77%
eMBB Total Rate: 535.86 Mbps, URLLC Total Rate: 311.29 Mbps, mMTC Total Rate: 71.10 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          7  30.0/30 MHz       100.00%
mMTC           5  8.5/10 MHz        85.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 9, Bandwidth: 20.0 MHz, Rate: 190.30 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |         70.85 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3499, completion~597, total~4096

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need to stream 8K video content"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to stream 8k video content", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~845, total~2217
[Token] Slice_Type_Determination: prompt~2694, completion~87, total~2781
[Token] Beamforming_Bandwidth: prompt~216, completion~209, total~425
[Token] Allocate_Resources: prompt~2901, completion~384, total~3285

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-16 11:16:55
Total Users: 16
Average Resource Utilization: 91.15%
eMBB Total Rate: 781.17 Mbps, URLLC Total Rate: 311.29 Mbps, mMTC Total Rate: 71.10 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          7  30.0/30 MHz       100.00%
mMTC           5  8.5/10 MHz        85.00%

New User Allocation:
User 16 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 245.31 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |         70.85 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       20   |        245.31 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3527, completion~808, total~4335

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need instant alerts for life-threatening patient conditions"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need instant alerts for life-threatening patient conditions", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~816, total~2188
[Token] Slice_Type_Determination: prompt~2652, completion~82, total~2734
[Token] Beamforming_Bandwidth: prompt~216, completion~4, total~220
[Token] Allocate_Resources: prompt~2899, completion~436, total~3335

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-16 11:18:01
Total Users: 17
Average Resource Utilization: 91.15%
eMBB Total Rate: 781.17 Mbps, URLLC Total Rate: 303.94 Mbps, mMTC Total Rate: 71.10 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          8  30.0/30 MHz       100.00%
mMTC           5  8.5/10 MHz        85.00%

New User Allocation:
User 17 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 11: 5.0 → 4.0 MHz, Rate: 75.68 → 60.54 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        4   |         60.54 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        1   |          7.79 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |         70.85 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3642, completion~908, total~4550

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "My smart meter needs to report its reading"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart meter needs to report its reading", "mmtc"),)
[Token] Intent_Analysis: prompt~1370, completion~750, total~2120
[Token] Slice_Type_Determination: prompt~2574, completion~118, total~2692
[Token] Beamforming_Bandwidth: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~2844, completion~511, total~3355

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-16 11:18:57
Total Users: 18
Average Resource Utilization: 91.85%
eMBB Total Rate: 781.17 Mbps, URLLC Total Rate: 303.94 Mbps, mMTC Total Rate: 78.88 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          8  30.0/30 MHz       100.00%
mMTC           6  9.4/10 MHz        94.00%

New User Allocation:
User 18 → mMTC Slice
CQI: 8, Bandwidth: 0.9 MHz, Rate: 7.78 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        5   |         70.85 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3593, completion~898, total~4491

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to control a robotic arm in real time"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control a robotic arm in real time", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~715, total~2089
[Token] Slice_Type_Determination: prompt~2548, completion~94, total~2642
[Token] Beamforming_Bandwidth: prompt~217, completion~4, total~221
[Token] Allocate_Resources: prompt~2807, completion~355, total~3162

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-16 11:19:56
Total Users: 19
Average Resource Utilization: 91.85%
eMBB Total Rate: 781.17 Mbps, URLLC Total Rate: 297.56 Mbps, mMTC Total Rate: 78.88 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          9  30.0/30 MHz       100.00%
mMTC           6  9.4/10 MHz        94.00%

New User Allocation:
User 19 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 3.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 5: 5.0 → 4.0 MHz, Rate: 70.85 → 56.68 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        4   |         56.68 |              1 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3474, completion~797, total~4271

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I want to watch 4K video"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to watch 4k video", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~465, total~1835
[Token] Slice_Type_Determination: prompt~2298, completion~115, total~2413
[Token] Beamforming_Bandwidth: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~2605, completion~384, total~2989

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-16 11:25:36
Total Users: 20
Average Resource Utilization: 99.54%
eMBB Total Rate: 885.16 Mbps, URLLC Total Rate: 297.56 Mbps, mMTC Total Rate: 78.88 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           6  9.4/10 MHz        94.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 11, Bandwidth: 20.0 MHz, Rate: 226.64 Mbps, Latency: 50.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 10.0 MHz
  User 16: 20.0 → 10.0 MHz, Rate: 245.31 → 122.66 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        4   |         56.68 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       20   |        226.64 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3299, completion~645, total~3944

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to stream music while browsing social media"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to stream music while browsing social media", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~704, total~2076
[Token] Slice_Type_Determination: prompt~2539, completion~92, total~2631
[Token] Beamforming_Bandwidth: prompt~216, completion~4, total~220
[Token] Allocate_Resources: prompt~2854, completion~350, total~3204

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-16 11:26:17
Total Users: 21
Average Resource Utilization: 99.54%
eMBB Total Rate: 920.18 Mbps, URLLC Total Rate: 297.56 Mbps, mMTC Total Rate: 78.88 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           6  9.4/10 MHz        94.00%

New User Allocation:
User 21 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 245.31 Mbps, Latency: 50.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 20.0 MHz
  User 20: 20.0 → 9.0 MHz, Rate: 226.64 → 101.99 Mbps, User 15: 20.0 → 11.0 MHz, Rate: 190.30 → 104.66 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        4   |         56.68 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        9   |        101.99 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3557, completion~766, total~4323

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "My asset tracking device needs to send location update"
CQI: 12
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my asset tracking device needs to send location update", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~693, total~2065
[Token] Slice_Type_Determination: prompt~2517, completion~68, total~2585
[Token] Beamforming_Bandwidth: prompt~220, completion~2, total~222
[Token] Allocate_Resources: prompt~2795, completion~645, total~3440

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-16 11:27:16
Total Users: 22
Average Resource Utilization: 100.0%
eMBB Total Rate: 920.18 Mbps, URLLC Total Rate: 297.56 Mbps, mMTC Total Rate: 87.06 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 12, Bandwidth: 0.9 MHz, Rate: 11.04 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.3000000000000004 MHz
  User 10: 1.9 → 1.5999999999999996 MHz, Rate: 18.08 → 15.22 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        4   |         56.68 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        9   |        101.99 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.6 |         15.22 |            500 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3769, completion~640, total~4409

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~645, total~2019
[Token] Slice_Type_Determination: prompt~2470, completion~63, total~2533
[Token] Beamforming_Bandwidth: prompt~216, completion~4, total~220
[Token] Allocate_Resources: prompt~2776, completion~592, total~3368

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-16 11:28:10
Total Users: 23
Average Resource Utilization: 100.0%
eMBB Total Rate: 920.18 Mbps, URLLC Total Rate: 297.56 Mbps, mMTC Total Rate: 92.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 23 → mMTC Slice
CQI: 15, Bandwidth: 0.9 MHz, Rate: 13.62 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 0.9 MHz
  User 4: 1.9 → 1.0 MHz, Rate: 16.41 → 8.64 Mbps, User 10: 1.5999999999999996 → 1.5999999999999996 MHz, Rate: 15.22 → 15.22 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        4   |         56.68 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        9   |        101.99 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.6 |         15.22 |            500 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1   |          8.64 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3738, completion~699, total~4437

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to deploy early warning systems for natural disasters"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to deploy early warning systems for natural disasters", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~766, total~2140
[Token] Slice_Type_Determination: prompt~2603, completion~118, total~2721
[Token] Beamforming_Bandwidth: prompt~217, completion~4, total~221
[Token] Allocate_Resources: prompt~2887, completion~400, total~3287

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-16 11:29:00
Total Users: 24
Average Resource Utilization: 100.0%
eMBB Total Rate: 920.18 Mbps, URLLC Total Rate: 291.07 Mbps, mMTC Total Rate: 92.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         10  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 8.64 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 11: 4.0 → 3.0 MHz, Rate: 60.54 → 45.41 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        3   |         45.41 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        4   |         56.68 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        9   |        101.99 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.6 |         15.22 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1   |          8.64 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3603, completion~708, total~4311

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I want to update my social media status"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~796, total~2166
[Token] Slice_Type_Determination: prompt~2633, completion~83, total~2716
[Token] Beamforming_Bandwidth: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~2969, completion~398, total~3367

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-16 11:29:55
Total Users: 25
Average Resource Utilization: 100.0%
eMBB Total Rate: 863.31 Mbps, URLLC Total Rate: 291.07 Mbps, mMTC Total Rate: 92.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         10  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 25 → eMBB Slice
CQI: 7, Bandwidth: 20.0 MHz, Rate: 155.80 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 3, Bandwidth freed: 20.0 MHz
  User 21: 20.0 → 9.0 MHz, Rate: 245.31 → 110.39 Mbps, User 1: 20.0 → 12.0 MHz, Rate: 172.78 → 103.67 Mbps, User 13: 20.0 → 19.0 MHz, Rate: 172.78 → 164.14 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        4   |         56.68 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       19   |        164.14 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        9   |        101.99 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.6 |         15.22 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1   |          8.64 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3760, completion~776, total~4536

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control precision cnc machines with zero tolerance for delay", "urllc"),)
[Token] Intent_Analysis: prompt~1378, completion~677, total~2055
[Token] Slice_Type_Determination: prompt~2515, completion~89, total~2604
[Token] Beamforming_Bandwidth: prompt~219, completion~4, total~223
[Token] Allocate_Resources: prompt~2770, completion~327, total~3097

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-16 11:30:50
Total Users: 26
Average Resource Utilization: 100.0%
eMBB Total Rate: 863.31 Mbps, URLLC Total Rate: 286.41 Mbps, mMTC Total Rate: 92.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         11  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 3.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 5: 4.0 → 3.0 MHz, Rate: 56.68 → 42.51 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        3   |         42.51 |              1 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        9   |        101.99 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.6 |         15.22 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1   |          8.64 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3405, completion~692, total~4097

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need to deploy early warning systems for natural disasters"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to deploy early warning systems for natural disasters", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~706, total~2080
[Token] Slice_Type_Determination: prompt~2541, completion~123, total~2664
[Token] Beamforming_Bandwidth: prompt~217, completion~4, total~221
[Token] Allocate_Resources: prompt~2829, completion~327, total~3156

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-16 11:31:37
Total Users: 27
Average Resource Utilization: 100.0%
eMBB Total Rate: 863.31 Mbps, URLLC Total Rate: 280.78 Mbps, mMTC Total Rate: 92.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         12  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 27 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 11: 3.0 → 2.0 MHz, Rate: 45.41 → 30.27 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        1   |          9.51 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        3   |         42.51 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        9   |        101.99 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.6 |         15.22 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1   |          8.64 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.9 |         14.8  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3470, completion~864, total~4334

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of city-wide smart streetlights", "mmtc"),, ("i need to check the status of city-wide smart streetlights", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~756, total~2134
[Token] Slice_Type_Determination: prompt~2583, completion~70, total~2653
[Token] Beamforming_Bandwidth: prompt~218, completion~4, total~222
[Token] Allocate_Resources: prompt~2895, completion~706, total~3601

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-16 11:32:38
Total Users: 28
Average Resource Utilization: 100.0%
eMBB Total Rate: 863.31 Mbps, URLLC Total Rate: 280.78 Mbps, mMTC Total Rate: 92.64 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         12  30.0/30 MHz       100.00%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 8, Bandwidth: 0.9 MHz, Rate: 7.78 Mbps, Latency: 1000.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 0.9 MHz
  User 10: 1.5999999999999996 → 1.0 MHz, Rate: 15.22 → 9.51 Mbps, User 9: 1.9 → 1.5999999999999996 MHz, Rate: 14.80 → 12.46 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        3   |         42.51 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        9   |        101.99 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1   |          9.51 |            500 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     8 |        0.9 |          7.78 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1   |          8.64 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.6 |         12.46 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3980, completion~887, total~4867

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need to check the status of my smart home sensors"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of my smart home sensors", "mmtc"),, ("i need to check the status of my smart home sensors", "mmtc"),)
[Token] Intent_Analysis: prompt~1376, completion~740, total~2116
[Token] Slice_Type_Determination: prompt~2565, completion~77, total~2642
[Token] Beamforming_Bandwidth: prompt~217, completion~4, total~221
[Token] Allocate_Resources: prompt~2884, completion~658, total~3542

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-16 11:33:33
Total Users: 29
Average Resource Utilization: 100.0%
eMBB Total Rate: 863.31 Mbps, URLLC Total Rate: 280.78 Mbps, mMTC Total Rate: 93.41 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         12  30.0/30 MHz       100.00%
mMTC          10  10.0/10 MHz       100.00%

New User Allocation:
User 29 → mMTC Slice
CQI: 8, Bandwidth: 0.9 MHz, Rate: 7.78 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 0.9 MHz
  User 12: 1.9 → 1.0 MHz, Rate: 14.80 → 7.79 Mbps, User 9: 1.5999999999999996 → 1.5999999999999996 MHz, Rate: 12.46 → 12.46 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        3   |         42.51 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        9   |        101.99 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1   |          9.51 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1   |          7.79 |            500 | ADJUSTED |
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
|         4 | mMTC    |     8 |        1   |          8.64 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.6 |         12.46 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3904, completion~800, total~4704

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to balance electrical load in real-time across microgrids", "urllc"),)
[Token] Intent_Analysis: prompt~1380, completion~897, total~2277
[Token] Slice_Type_Determination: prompt~2734, completion~118, total~2852
[Token] Beamforming_Bandwidth: prompt~220, completion~4, total~224
[Token] Allocate_Resources: prompt~3019, completion~378, total~3397

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-16 11:34:17
Total Users: 30
Average Resource Utilization: 100.0%
eMBB Total Rate: 863.31 Mbps, URLLC Total Rate: 277.24 Mbps, mMTC Total Rate: 93.41 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         13  30.0/30 MHz       100.00%
mMTC          10  10.0/10 MHz       100.00%

New User Allocation:
User 30 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 3: 4.0 → 3.0 MHz, Rate: 45.33 → 34.00 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        3   |         34    |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | URLLC   |     7 |        1   |          7.79 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        3   |         42.51 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        4   |         34.56 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       10   |        122.66 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |        9   |        101.99 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1   |          9.51 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        1   |          7.79 |            500 |          |
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
|         4 | mMTC    |     8 |        1   |          8.64 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        1.6 |         12.46 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3704, completion~817, total~4521

Detailed Slice Utilization Values:
eMBB utils: [22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 44.44, 44.44, 66.67, 88.89, 88.89, 88.89, 88.89, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 16.67, 30.0, 30.0, 46.67, 46.67, 63.33, 76.67, 76.67, 76.67, 93.33, 93.33, 93.33, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
mMTC utils: [0.0, 0.0, 0.0, 19.0, 19.0, 28.0, 28.0, 28.0, 47.0, 66.0, 66.0, 85.0, 85.0, 85.0, 85.0, 85.0, 85.0, 94.0, 94.0, 94.0, 94.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |     8 |       20   |        172.78 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | URLLC   | URLLC          | Yes            |     5 |        5   |         30.97 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | URLLC   | URLLC          | Yes            |    11 |        4   |         45.33 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | mMTC    | mMTC           | Yes            |     8 |        1.9 |         16.41 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | URLLC   | URLLC          | Yes            |    14 |        5   |         70.85 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |     6 |        5   |         34.87 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | URLLC          | Yes            |     8 |        4   |         34.56 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | mMTC    | mMTC           | Yes            |     7 |        1.9 |         14.8  |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | mMTC    | mMTC           | Yes            |     9 |        1.9 |         18.08 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |    15 |        5   |         75.68 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | mMTC    | mMTC           | Yes            |     7 |        1.9 |         14.8  |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | eMBB    | eMBB           | Yes            |     8 |       20   |        172.78 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | URLLC   | URLLC          | Yes            |     9 |        2   |         19.03 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |     9 |       20   |        190.3  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | URLLC   | URLLC          | Yes            |     7 |        1   |          7.79 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | mMTC    | mMTC           | Yes            |     8 |        0.9 |          7.78 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | URLLC   | URLLC          | Yes            |     7 |        1   |          7.79 |              3 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    11 |       20   |        226.64 |             50 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             50 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | mMTC    | mMTC           | Yes            |    12 |        0.9 |         11.04 |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | mMTC    | mMTC           | Yes            |    15 |        0.9 |         13.62 |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     8 |        1   |          8.64 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | URLLC          | Yes            |     9 |        1   |          9.51 |              3 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | URLLC   | URLLC          | Yes            |     9 |        1   |          9.51 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | mMTC    | mMTC           | Yes            |     8 |        0.9 |          7.78 |           1000 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | mMTC    | mMTC           | Yes            |     8 |        0.9 |          7.78 |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Success  | URLLC   | URLLC          | Yes            |     7 |        1   |          7.79 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 30/30 (100.0%)

Intent Understanding Evaluation:
Correctly identified intents: 30/30
Intent understanding rate: 100.0%

Workload Balancing Statistics:
Users with workload balancing: 0/30
Workload balancing rate: 0.0%

Slice Utilization Statistics:
Average eMBB utilization: 62.59%
Average URLLC utilization: 81.44%
Average mMTC utilization: 69.57%

Weighted Average Utilization: 67.48%

Transmission Rate Statistics:
Final eMBB total rate: 863.31 Mbps
Final URLLC total rate: 277.24 Mbps
Final mMTC total rate: 93.41 Mbps

Resource Utilization:
Average resource utilization: 100.00%

Results exported to F:\code\wirelessagent\run_results\batch_run\fkb\kimi-k2.5\network_slicing_results_TJU_north_kimi-k2.5.csv

✓ TJU_north 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\fkb\kimi-k2.5\network_slicing_results_TJU_north_kimi-k2.5.csv