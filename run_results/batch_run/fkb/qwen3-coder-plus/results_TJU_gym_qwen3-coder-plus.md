============================================================
场景 5/5: TJU_gym
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_gym.csv
输出: F:\code\wirelessagent\run_results\batch_run\fkb\qwen3-coder-plus\network_slicing_results_TJU_gym_qwen3-coder-plus.csv
------------------------------------------------------------
Starting network slice management system with CSV-based user testing...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~513, total~1887
[Token] Slice_Type_Determination: prompt~2333, completion~80, total~2413
[Token] Beamforming_Bandwidth: prompt~214, completion~111, total~325
[Token] Allocate_Resources: prompt~2564, completion~143, total~2707

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-16 16:13:32
Total Users: 1
Average Resource Utilization: 1.46%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 9.05 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          0  0/30 MHz          0%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 1 → mMTC Slice
CQI: 3, Bandwidth: 1.9 MHz, Rate: 9.05 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2927, completion~222, total~3149

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I want to browse websites and check email"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~610, total~1980
[Token] Slice_Type_Determination: prompt~2408, completion~99, total~2507
[Token] Beamforming_Bandwidth: prompt~213, completion~111, total~324
[Token] Allocate_Resources: prompt~2645, completion~356, total~3001

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-16 16:14:00
Total Users: 2
Average Resource Utilization: 16.85%
eMBB Total Rate: 226.64 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 9.05 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          0  0/30 MHz          0%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 2 → eMBB Slice
CQI: 11, Bandwidth: 20.0 MHz, Rate: 226.64 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3176, completion~290, total~3466

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to monitor water level in a reservoir"
CQI: 13
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor water level in a reservoir", "mmtc"),, ("i need to monitor water level in a reservoir periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~496, total~1868
[Token] Slice_Type_Determination: prompt~2312, completion~75, total~2387
[Token] Beamforming_Bandwidth: prompt~215, completion~102, total~317
[Token] Allocate_Resources: prompt~2539, completion~131, total~2670

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-16 16:14:24
Total Users: 3
Average Resource Utilization: 17.54%
eMBB Total Rate: 226.64 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 20.94 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          0  0/30 MHz          0%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 3 → mMTC Slice
CQI: 13, Bandwidth: 0.9 MHz, Rate: 11.89 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2897, completion~184, total~3081

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my wearable device needs to upload health data periodically", "mmtc"),, ("my wearable device needs to upload health data periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~451, total~1823
[Token] Slice_Type_Determination: prompt~2268, completion~71, total~2339
[Token] Beamforming_Bandwidth: prompt~215, completion~101, total~316
[Token] Allocate_Resources: prompt~2495, completion~329, total~2824

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-16 16:14:47
Total Users: 4
Average Resource Utilization: 18.23%
eMBB Total Rate: 226.64 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 25.85 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           1  20.0/90 MHz                22.22%
URLLC          0  0/30 MHz                   0%
mMTC           3  3.6999999999999997/10 MHz  37.00%

New User Allocation:
User 4 → mMTC Slice
CQI: 4, Bandwidth: 0.9 MHz, Rate: 4.91 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3044, completion~338, total~3382

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[Token] Intent_Analysis: prompt~1376, completion~579, total~1955
[Token] Slice_Type_Determination: prompt~2384, completion~78, total~2462
[Token] Beamforming_Bandwidth: prompt~218, completion~106, total~324
[Token] Allocate_Resources: prompt~2572, completion~263, total~2835

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-16 16:15:16
Total Users: 5
Average Resource Utilization: 29.77%
eMBB Total Rate: 356.22 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 25.85 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           2  35.0/90 MHz                38.89%
URLLC          0  0/30 MHz                   0%
mMTC           3  3.6999999999999997/10 MHz  37.00%

New User Allocation:
User 5 → eMBB Slice
CQI: 8, Bandwidth: 15.0 MHz, Rate: 129.58 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3054, completion~277, total~3331

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I need to stream 8K video content"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to stream 8k video content", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~586, total~1958
[Token] Slice_Type_Determination: prompt~2383, completion~98, total~2481
[Token] Beamforming_Bandwidth: prompt~216, completion~114, total~330
[Token] Allocate_Resources: prompt~2591, completion~252, total~2843

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-16 16:15:45
Total Users: 6
Average Resource Utilization: 45.15%
eMBB Total Rate: 480.09 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 25.85 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           3  55.0/90 MHz                61.11%
URLLC          0  0/30 MHz                   0%
mMTC           3  3.6999999999999997/10 MHz  37.00%

New User Allocation:
User 6 → eMBB Slice
CQI: 5, Bandwidth: 20.0 MHz, Rate: 123.87 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3068, completion~403, total~3471

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to track the location of a shipping container"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to track the location of a shipping container", "mmtc"),, ("i need to track the location of a shipping container", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~465, total~1839
[Token] Slice_Type_Determination: prompt~2285, completion~68, total~2353
[Token] Beamforming_Bandwidth: prompt~220, completion~82, total~302
[Token] Allocate_Resources: prompt~2509, completion~323, total~2832

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-16 16:16:12
Total Users: 7
Average Resource Utilization: 45.85%
eMBB Total Rate: 480.09 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 32.13 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  55.0/90 MHz       61.11%
URLLC          0  0/30 MHz          0%
mMTC           4  4.6/10 MHz        46.00%

New User Allocation:
User 7 → mMTC Slice
CQI: 6, Bandwidth: 0.9 MHz, Rate: 6.28 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3049, completion~321, total~3370

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my wearable device needs to upload health data periodically", "mmtc"),, ("my wearable device needs to upload health data periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~444, total~1816
[Token] Slice_Type_Determination: prompt~2261, completion~95, total~2356
[Token] Beamforming_Bandwidth: prompt~215, completion~239, total~454
[Token] Allocate_Resources: prompt~2508, completion~150, total~2658

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-16 16:16:42
Total Users: 8
Average Resource Utilization: 46.54%
eMBB Total Rate: 480.09 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 45.75 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  55.0/90 MHz       61.11%
URLLC          0  0/30 MHz          0%
mMTC           5  5.5/10 MHz        55.00%

New User Allocation:
User 8 → mMTC Slice
CQI: 15, Bandwidth: 0.9 MHz, Rate: 13.62 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2885, completion~266, total~3151

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I need reliable connectivity for implanted medical devices"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable connectivity for implanted medical devices", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~451, total~1821
[Token] Slice_Type_Determination: prompt~2268, completion~75, total~2343
[Token] Beamforming_Bandwidth: prompt~213, completion~101, total~314
[Token] Allocate_Resources: prompt~2459, completion~230, total~2689

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-16 16:17:08
Total Users: 9
Average Resource Utilization: 49.62%
eMBB Total Rate: 480.09 Mbps, URLLC Total Rate: 31.16 Mbps, mMTC Total Rate: 45.75 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  55.0/90 MHz       61.11%
URLLC          1  4.0/30 MHz        13.33%
mMTC           5  5.5/10 MHz        55.00%

New User Allocation:
User 9 → URLLC Slice
CQI: 7, Bandwidth: 4.0 MHz, Rate: 31.16 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2919, completion~325, total~3244

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to check weather forecasts"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to check weather forecasts", "embb"),)
[Token] Intent_Analysis: prompt~1366, completion~508, total~1874
[Token] Slice_Type_Determination: prompt~2315, completion~86, total~2401
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~2465, completion~68, total~2533
[Token] Beamforming_Bandwidth: prompt~213, completion~105, total~318
[Token] Allocate_Resources: prompt~2677, completion~91, total~2768

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-16 16:17:36
Total Users: 10
Average Resource Utilization: 65.0%
eMBB Total Rate: 603.96 Mbps, URLLC Total Rate: 31.16 Mbps, mMTC Total Rate: 45.75 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  75.0/90 MHz       83.33%
URLLC          1  4.0/30 MHz        13.33%
mMTC           5  5.5/10 MHz        55.00%

New User Allocation:
User 10 → eMBB Slice
CQI: 5, Bandwidth: 20.0 MHz, Rate: 123.87 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3048, completion~286, total~3334

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of city-wide smart streetlights", "mmtc"),, ("i need to check the status of city-wide smart streetlights", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~471, total~1849
[Token] Slice_Type_Determination: prompt~2295, completion~72, total~2367
[Token] Beamforming_Bandwidth: prompt~218, completion~215, total~433
[Token] Allocate_Resources: prompt~2523, completion~372, total~2895

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-16 16:18:06
Total Users: 11
Average Resource Utilization: 65.69%
eMBB Total Rate: 603.96 Mbps, URLLC Total Rate: 31.16 Mbps, mMTC Total Rate: 52.03 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  75.0/90 MHz       83.33%
URLLC          1  4.0/30 MHz        13.33%
mMTC           6  6.4/10 MHz        64.00%

New User Allocation:
User 11 → mMTC Slice
CQI: 6, Bandwidth: 0.9 MHz, Rate: 6.28 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3111, completion~346, total~3457

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need microsecond-level latency for high-frequency tradin"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need microsecond-level latency for high-frequency trading", "urllc"),)
[Token] Intent_Analysis: prompt~1376, completion~547, total~1923
[Token] Slice_Type_Determination: prompt~2365, completion~79, total~2444
[Token] Beamforming_Bandwidth: prompt~218, completion~121, total~339
[Token] Allocate_Resources: prompt~2560, completion~244, total~2804

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-16 16:18:34
Total Users: 12
Average Resource Utilization: 68.77%
eMBB Total Rate: 603.96 Mbps, URLLC Total Rate: 50.22 Mbps, mMTC Total Rate: 52.03 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  75.0/90 MHz       83.33%
URLLC          2  8.0/30 MHz        26.67%
mMTC           6  6.4/10 MHz        64.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 3, Bandwidth: 4.0 MHz, Rate: 19.06 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3027, completion~300, total~3327

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I want to use holographic communication"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use holographic communication", "embb"),)
[Token] Intent_Analysis: prompt~1368, completion~602, total~1970
[Token] Slice_Type_Determination: prompt~2397, completion~99, total~2496
[Token] Beamforming_Bandwidth: prompt~214, completion~102, total~316
[Token] Allocate_Resources: prompt~2607, completion~280, total~2887

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-16 16:19:03
Total Users: 13
Average Resource Utilization: 78.0%
eMBB Total Rate: 718.14 Mbps, URLLC Total Rate: 50.22 Mbps, mMTC Total Rate: 52.03 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  87.0/90 MHz       96.67%
URLLC          2  8.0/30 MHz        26.67%
mMTC           6  6.4/10 MHz        64.00%

New User Allocation:
User 13 → eMBB Slice
CQI: 9, Bandwidth: 12.0 MHz, Rate: 114.18 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3093, completion~410, total~3503

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 12
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control critical infrastructure with zero downtime", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~434, total~1806
[Token] Slice_Type_Determination: prompt~2250, completion~74, total~2324
[Token] Beamforming_Bandwidth: prompt~216, completion~115, total~331
[Token] Allocate_Resources: prompt~2440, completion~272, total~2712

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-16 16:19:31
Total Users: 14
Average Resource Utilization: 81.08%
eMBB Total Rate: 718.14 Mbps, URLLC Total Rate: 99.28 Mbps, mMTC Total Rate: 52.03 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  87.0/90 MHz       96.67%
URLLC          3  12.0/30 MHz       40.00%
mMTC           6  6.4/10 MHz        64.00%

New User Allocation:
User 14 → URLLC Slice
CQI: 12, Bandwidth: 4.0 MHz, Rate: 49.06 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        4   |         49.06 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2936, completion~405, total~3341

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need real-time fraud detection for financial transactions"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need real-time fraud detection for financial transactions", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~376, total~1748
[Token] Slice_Type_Determination: prompt~2197, completion~75, total~2272
[Token] Beamforming_Bandwidth: prompt~216, completion~101, total~317
[Token] Allocate_Resources: prompt~2388, completion~227, total~2615

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-16 16:19:55
Total Users: 15
Average Resource Utilization: 84.15%
eMBB Total Rate: 718.14 Mbps, URLLC Total Rate: 130.44 Mbps, mMTC Total Rate: 52.03 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  87.0/90 MHz       96.67%
URLLC          4  16.0/30 MHz       53.33%
mMTC           6  6.4/10 MHz        64.00%

New User Allocation:
User 15 → URLLC Slice
CQI: 7, Bandwidth: 4.0 MHz, Rate: 31.16 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        4   |         49.06 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        4   |         31.16 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2844, completion~306, total~3150

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need instant facial recognition for public security threats"
CQI: 10
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need instant facial recognition for public security threats", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~442, total~1814
[Token] Slice_Type_Determination: prompt~2257, completion~77, total~2334
[Token] Beamforming_Bandwidth: prompt~216, completion~112, total~328
[Token] Allocate_Resources: prompt~2449, completion~251, total~2700

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-16 16:20:23
Total Users: 16
Average Resource Utilization: 87.23%
eMBB Total Rate: 718.14 Mbps, URLLC Total Rate: 172.10 Mbps, mMTC Total Rate: 52.03 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  87.0/90 MHz       96.67%
URLLC          5  20.0/30 MHz       66.67%
mMTC           6  6.4/10 MHz        64.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 10, Bandwidth: 4.0 MHz, Rate: 41.66 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        4   |         49.06 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        4   |         31.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        4   |         41.66 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2925, completion~350, total~3275

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "My smart trash can needs to signal that it's full"
CQI: 13
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart trash can needs to signal that it's full", "mmtc"),)
[Token] Intent_Analysis: prompt~1376, completion~504, total~1880
[Token] Slice_Type_Determination: prompt~2325, completion~70, total~2395
[Token] Beamforming_Bandwidth: prompt~222, completion~196, total~418
[Token] Allocate_Resources: prompt~2551, completion~139, total~2690

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-16 16:20:50
Total Users: 17
Average Resource Utilization: 87.92%
eMBB Total Rate: 718.14 Mbps, URLLC Total Rate: 172.10 Mbps, mMTC Total Rate: 63.92 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  87.0/90 MHz               96.67%
URLLC          5  20.0/30 MHz               66.67%
mMTC           7  7.300000000000001/10 MHz  73.00%

New User Allocation:
User 17 → mMTC Slice
CQI: 13, Bandwidth: 0.9 MHz, Rate: 11.89 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        4   |         49.06 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        4   |         31.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        4   |         41.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2925, completion~302, total~3227

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~479, total~1849
[Token] Slice_Type_Determination: prompt~2289, completion~78, total~2367
[Token] Beamforming_Bandwidth: prompt~215, completion~87, total~302
[Token] Allocate_Resources: prompt~2483, completion~282, total~2765

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-16 16:21:19
Total Users: 18
Average Resource Utilization: 91.0%
eMBB Total Rate: 718.14 Mbps, URLLC Total Rate: 191.16 Mbps, mMTC Total Rate: 63.92 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  87.0/90 MHz               96.67%
URLLC          6  24.0/30 MHz               80.00%
mMTC           7  7.300000000000001/10 MHz  73.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 3, Bandwidth: 4.0 MHz, Rate: 19.06 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        4   |         49.06 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        4   |         31.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        4   |         41.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        4   |         19.06 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2997, completion~352, total~3349

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to monitor water level in a reservoir"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor water level in a reservoir", "mmtc"),, ("i need to monitor water level in a reservoir periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~440, total~1812
[Token] Slice_Type_Determination: prompt~2262, completion~73, total~2335
[Token] Beamforming_Bandwidth: prompt~235, completion~105, total~340
[Token] Allocate_Resources: prompt~2491, completion~489, total~2980

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-16 16:21:49
Total Users: 19
Average Resource Utilization: 91.69%
eMBB Total Rate: 718.14 Mbps, URLLC Total Rate: 191.16 Mbps, mMTC Total Rate: 68.21 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  87.0/90 MHz               96.67%
URLLC          6  24.0/30 MHz               80.00%
mMTC           8  8.200000000000001/10 MHz  82.00%

New User Allocation:
User 19 → mMTC Slice
CQI: 3, Bandwidth: 0.9 MHz, Rate: 4.29 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        4   |         49.06 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        4   |         31.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        4   |         41.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       20   |        226.64 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       15   |        129.58 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3183, completion~430, total~3613

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I need to make a high-quality voice call"
CQI: 10
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to make a high-quality voice call", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~612, total~1984
[Token] Slice_Type_Determination: prompt~2419, completion~73, total~2492
LLM recommended URLLC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~2560, completion~91, total~2651
[Token] Beamforming_Bandwidth: prompt~216, completion~112, total~328
[Token] Allocate_Resources: prompt~2901, completion~279, total~3180

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-16 16:22:23
Total Users: 20
Average Resource Utilization: 94.0%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 191.16 Mbps, mMTC Total Rate: 68.21 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC          6  24.0/30 MHz               80.00%
mMTC           8  8.200000000000001/10 MHz  82.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 10, Bandwidth: 20.0 MHz, Rate: 208.28 Mbps, Latency: 30.0 ms

Dynamic Resource Adjustments:
Users adjusted: 3, Bandwidth freed: 17.0 MHz
  User 2: 20.0 → 9.0 MHz, Rate: 226.64 → 101.99 Mbps, User 5: 15.0 → 12.0 MHz, Rate: 129.58 → 103.67 Mbps, User 6: 20.0 → 17.0 MHz, Rate: 123.87 → 105.29 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        4   |         49.06 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        4   |         31.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        4   |         41.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3592, completion~299, total~3891

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to use holographic communication"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use holographic communication", "embb"),)
[Token] Intent_Analysis: prompt~1368, completion~491, total~1859
[Token] Slice_Type_Determination: prompt~2295, completion~83, total~2378
[Token] Beamforming_Bandwidth: prompt~214, completion~203, total~417
[Token] Allocate_Resources: prompt~2564, completion~113, total~2677
[Token] Failure_Evaluation: prompt~2754, completion~214, total~2968

----------------------------------------
ALLOCATION FAILED FOR USER 21
----------------------------------------
Request: I want to use holographic communication
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "I need to check the status of my smart home sensors"
CQI: 11
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of my smart home sensors", "mmtc"),, ("i need to check the status of my smart home sensors", "mmtc"),)
[Token] Intent_Analysis: prompt~1376, completion~473, total~1849
[Token] Slice_Type_Determination: prompt~2293, completion~96, total~2389
[Token] Beamforming_Bandwidth: prompt~233, completion~1, total~234
[Token] Allocate_Resources: prompt~2544, completion~183, total~2727

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-16 16:23:14
Total Users: 21
Average Resource Utilization: 94.69%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 191.16 Mbps, mMTC Total Rate: 78.41 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC          6  24.0/30 MHz               80.00%
mMTC           9  9.100000000000001/10 MHz  91.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 11, Bandwidth: 0.9 MHz, Rate: 10.20 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        4   |         49.06 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        4   |         31.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        4   |         41.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |        0.9 |         10.2  |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2956, completion~324, total~3280

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 1
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to detect and isolate power grid faults instantly", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~429, total~1803
[Token] Slice_Type_Determination: prompt~2252, completion~77, total~2329
[Token] Beamforming_Bandwidth: prompt~217, completion~133, total~350
[Token] Allocate_Resources: prompt~2445, completion~249, total~2694

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-16 16:23:43
Total Users: 22
Average Resource Utilization: 97.77%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 205.32 Mbps, mMTC Total Rate: 78.41 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC          7  28.0/30 MHz               93.33%
mMTC           9  9.100000000000001/10 MHz  91.00%

New User Allocation:
User 23 → URLLC Slice
CQI: 1, Bandwidth: 4.0 MHz, Rate: 14.16 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        4   |         49.06 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        4   |         31.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        4   |         41.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        4   |         14.16 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |        0.9 |         10.2  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2925, completion~324, total~3249

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I want to use remote surgery equipment"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to use remote surgery equipment", "urllc"),)
[Token] Intent_Analysis: prompt~1368, completion~450, total~1818
[Token] Slice_Type_Determination: prompt~2267, completion~83, total~2350
[Token] Beamforming_Bandwidth: prompt~214, completion~113, total~327
[Token] Allocate_Resources: prompt~2466, completion~215, total~2681

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-16 16:24:09
Total Users: 23
Average Resource Utilization: 99.31%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 217.71 Mbps, mMTC Total Rate: 78.41 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC          8  30.0/30 MHz               100.00%
mMTC           9  9.100000000000001/10 MHz  91.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 5, Bandwidth: 2.0 MHz, Rate: 12.39 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        4   |         49.06 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        4   |         31.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        4   |         41.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        4   |         14.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        2   |         12.39 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |        0.9 |         10.2  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2920, completion~366, total~3286

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I want to stream music while browsing social media"
CQI: 2
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to stream music while browsing social media", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~499, total~1871
[Token] Slice_Type_Determination: prompt~2313, completion~84, total~2397
[Token] Beamforming_Bandwidth: prompt~216, completion~177, total~393
[Token] Allocate_Resources: prompt~2584, completion~259, total~2843
[Token] Failure_Evaluation: prompt~2927, completion~331, total~3258

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
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to balance electrical load in real-time across microgrids", "urllc"),)
[Token] Intent_Analysis: prompt~1380, completion~458, total~1838
[Token] Slice_Type_Determination: prompt~2287, completion~64, total~2351
[Token] Beamforming_Bandwidth: prompt~220, completion~52, total~272
[Token] Allocate_Resources: prompt~2516, completion~188, total~2704

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-16 16:25:04
Total Users: 24
Average Resource Utilization: 99.31%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 212.42 Mbps, mMTC Total Rate: 78.41 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC          9  30.0/30 MHz               100.00%
mMTC           9  9.100000000000001/10 MHz  91.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 6.97 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 14: 4.0 → 3.0 MHz, Rate: 49.06 → 36.80 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        3   |         36.8  |              3 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        4   |         31.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        4   |         41.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        4   |         14.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        2   |         12.39 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        1   |          6.97 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |        0.9 |         10.2  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3010, completion~273, total~3283

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need to transmit real-time patient vital signs during critical care"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to transmit real-time patient vital signs during critical care", "urllc"),)
[Token] Intent_Analysis: prompt~1378, completion~466, total~1844
[Token] Slice_Type_Determination: prompt~2292, completion~73, total~2365
[Token] Beamforming_Bandwidth: prompt~219, completion~76, total~295
[Token] Allocate_Resources: prompt~2529, completion~298, total~2827

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-16 16:25:32
Total Users: 25
Average Resource Utilization: 99.31%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 206.76 Mbps, mMTC Total Rate: 78.41 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC         10  30.0/30 MHz               100.00%
mMTC           9  9.100000000000001/10 MHz  91.00%

New User Allocation:
User 27 → URLLC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 4.76 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 16: 4.0 → 3.0 MHz, Rate: 41.66 → 31.24 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        3   |         36.8  |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        4   |         31.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        3   |         31.24 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        4   |         14.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        2   |         12.39 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        1   |          4.76 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |        0.9 |         10.2  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3129, completion~411, total~3540

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~455, total~1825
[Token] Slice_Type_Determination: prompt~2274, completion~74, total~2348
[Token] Beamforming_Bandwidth: prompt~215, completion~51, total~266
[Token] Allocate_Resources: prompt~2512, completion~163, total~2675

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-16 16:26:00
Total Users: 26
Average Resource Utilization: 99.31%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 199.95 Mbps, mMTC Total Rate: 78.41 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC         11  30.0/30 MHz               100.00%
mMTC           9  9.100000000000001/10 MHz  91.00%

New User Allocation:
User 28 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 14: 3.0 → 2.0 MHz, Rate: 36.80 → 24.53 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        4   |         31.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        3   |         31.24 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        4   |         14.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        2   |         12.39 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | URLLC   |     4 |        1   |          5.46 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |        0.9 |         10.2  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2981, completion~279, total~3260

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need to participate in an online multiplayer game"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to participate in an online multiplayer game", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~463, total~1835
[Token] Slice_Type_Determination: prompt~2278, completion~60, total~2338
[Token] Beamforming_Bandwidth: prompt~216, completion~229, total~445
[Token] Allocate_Resources: prompt~2503, completion~177, total~2680

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-16 16:26:29
Total Users: 27
Average Resource Utilization: 99.31%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 204.68 Mbps, mMTC Total Rate: 78.41 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC         12  30.0/30 MHz               100.00%
mMTC           9  9.100000000000001/10 MHz  91.00%

New User Allocation:
User 29 → URLLC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 15.14 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 16: 3.0 → 2.0 MHz, Rate: 31.24 → 20.83 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        4   |         31.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        2   |         20.83 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        4   |         14.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        2   |         12.39 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | URLLC   |    15 |        1   |         15.14 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       17   |        105.29 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |        0.9 |         10.2  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        0.9 |         13.62 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2986, completion~305, total~3291

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I want to browse websites and check email"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~499, total~1869
[Token] Slice_Type_Determination: prompt~2308, completion~104, total~2412
[Token] Beamforming_Bandwidth: prompt~215, completion~213, total~428
[Token] Allocate_Resources: prompt~2594, completion~403, total~2997
[Token] Failure_Evaluation: prompt~3057, completion~237, total~3294

----------------------------------------
ALLOCATION FAILED FOR USER 30
----------------------------------------
Request: I want to browse websites and check email
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

Detailed Slice Utilization Values:
eMBB utils: [0.0, 22.22, 22.22, 22.22, 38.89, 61.11, 61.11, 61.11, 61.11, 83.33, 83.33, 83.33, 96.67, 96.67, 96.67, 96.67, 96.67, 96.67, 96.67, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.33, 13.33, 13.33, 26.67, 26.67, 40.0, 53.33, 66.67, 66.67, 80.0, 80.0, 80.0, 80.0, 93.33, 100.0, 100.0, 100.0, 100.0, 100.0]
mMTC utils: [19.0, 19.0, 28.0, 37.0, 37.0, 37.0, 46.0, 55.0, 55.0, 55.0, 64.0, 64.0, 64.0, 64.0, 64.0, 64.0, 73.0, 73.0, 82.0, 82.0, 91.0, 91.0, 91.0, 91.0, 91.0, 91.0, 91.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | mMTC    | mMTC           | Yes            |     3 |        1.9 |          9.05 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | eMBB    | eMBB           | Yes            |    11 |       20   |        226.64 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | mMTC    | mMTC           | Yes            |    13 |        0.9 |         11.89 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | mMTC    | mMTC           | Yes            |     4 |        0.9 |          4.91 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | eMBB    | eMBB           | Yes            |     8 |       15   |        129.58 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | eMBB    | eMBB           | Yes            |     5 |       20   |        123.87 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | mMTC    | mMTC           | Yes            |     6 |        0.9 |          6.28 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | mMTC    | mMTC           | Yes            |    15 |        0.9 |         13.62 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | URLLC   | URLLC          | Yes            |     7 |        4   |         31.16 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | eMBB    | eMBB           | Yes            |     5 |       20   |        123.87 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | mMTC    | mMTC           | Yes            |     6 |        0.9 |          6.28 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     3 |        4   |         19.06 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | eMBB    | eMBB           | Yes            |     9 |       12   |        114.18 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | URLLC   | URLLC          | Yes            |    12 |        4   |         49.06 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | URLLC   | URLLC          | Yes            |     7 |        4   |         31.16 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |    10 |        4   |         41.66 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | mMTC    | mMTC           | Yes            |    13 |        0.9 |         11.89 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     3 |        4   |         19.06 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | mMTC    | mMTC           | Yes            |     3 |        0.9 |          4.29 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    10 |       20   |        208.28 |             30 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Failed   | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | mMTC    | mMTC           | Yes            |    11 |        0.9 |         10.2  |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | URLLC   | URLLC          | Yes            |     1 |        4   |         14.16 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     5 |        2   |         12.39 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Failed   | eMBB    | eMBB           | Yes            |     2 |       20   |         82.49 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | URLLC          | Yes            |     6 |        1   |          6.97 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | URLLC   | URLLC          | Yes            |     3 |        1   |          4.76 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          5.46 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | URLLC   | URLLC          | Yes            |    15 |        1   |         15.14 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Failed   | eMBB    | eMBB           | Yes            |     8 |       20   |        172.78 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 27/30 (90.0%)

Intent Understanding Evaluation:
Correctly identified intents: 30/30
Intent understanding rate: 100.0%

Workload Balancing Statistics:
Users with workload balancing: 0/30
Workload balancing rate: 0.0%

Slice Utilization Statistics:
Average eMBB utilization: 76.91%
Average URLLC utilization: 45.68%
Average mMTC utilization: 63.67%

Weighted Average Utilization: 68.69%

Transmission Rate Statistics:
Final eMBB total rate: 757.28 Mbps
Final URLLC total rate: 204.68 Mbps
Final mMTC total rate: 78.41 Mbps

Resource Utilization:
Average resource utilization: 99.31%

Results exported to F:\code\wirelessagent\run_results\batch_run\fkb\qwen3-coder-plus\network_slicing_results_TJU_gym_qwen3-coder-plus.csv

✓ TJU_gym 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\fkb\qwen3-coder-plus\network_slicing_results_TJU_gym_qwen3-coder-plus.csv