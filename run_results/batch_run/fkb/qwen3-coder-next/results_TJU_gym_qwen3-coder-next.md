============================================================
场景 5/5: TJU_gym
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_gym.csv
输出: F:\code\wirelessagent\run_results\batch_run\fkb\qwen3-coder-next\network_slicing_results_TJU_gym_qwen3-coder-next.csv
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
[Token] Intent_Analysis: prompt~1374, completion~637, total~2011
[Token] Slice_Type_Determination: prompt~2446, completion~108, total~2554
[Token] Beamforming_Bandwidth: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~2704, completion~321, total~3025

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-16 14:37:01
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
[Token] Network_Evaluation: prompt~3235, completion~788, total~4023

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I want to browse websites and check email"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~638, total~2008
[Token] Slice_Type_Determination: prompt~2427, completion~139, total~2566
[Token] Beamforming_Bandwidth: prompt~213, completion~3, total~216
[Token] Allocate_Resources: prompt~2673, completion~195, total~2868

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-16 14:37:29
Total Users: 2
Average Resource Utilization: 10.69%
eMBB Total Rate: 135.98 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 9.05 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          0  0/30 MHz          0%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 2 → eMBB Slice
CQI: 11, Bandwidth: 12.0 MHz, Rate: 135.98 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3081, completion~348, total~3429

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to monitor water level in a reservoir"
CQI: 13
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor water level in a reservoir", "mmtc"),, ("i need to monitor water level in a reservoir periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~1142, total~2514
[Token] Slice_Type_Determination: prompt~2920, completion~111, total~3031
[Token] Beamforming_Bandwidth: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3182, completion~483, total~3665

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-16 14:37:45
Total Users: 3
Average Resource Utilization: 12.15%
eMBB Total Rate: 135.98 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 34.15 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          0  0/30 MHz          0%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 3 → mMTC Slice
CQI: 13, Bandwidth: 1.9 MHz, Rate: 25.10 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3879, completion~581, total~4460

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my wearable device needs to upload health data periodically", "mmtc"),, ("my wearable device needs to upload health data periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~941, total~2313
[Token] Slice_Type_Determination: prompt~2729, completion~124, total~2853
[Token] Beamforming_Bandwidth: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3008, completion~250, total~3258

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-16 14:37:59
Total Users: 4
Average Resource Utilization: 12.85%
eMBB Total Rate: 135.98 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 39.06 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          0  0/30 MHz          0%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 4 → mMTC Slice
CQI: 4, Bandwidth: 0.9 MHz, Rate: 4.91 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3484, completion~310, total~3794

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[Token] Intent_Analysis: prompt~1376, completion~1113, total~2489
[Token] Slice_Type_Determination: prompt~2889, completion~158, total~3047
[Token] Beamforming_Bandwidth: prompt~218, completion~3, total~221
[Token] Allocate_Resources: prompt~3153, completion~242, total~3395

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-16 14:38:11
Total Users: 5
Average Resource Utilization: 22.08%
eMBB Total Rate: 239.65 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 39.06 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  24.0/90 MHz       26.67%
URLLC          0  0/30 MHz          0%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 5 → eMBB Slice
CQI: 8, Bandwidth: 12.0 MHz, Rate: 103.67 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3597, completion~300, total~3897

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I need to stream 8K video content"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to stream 8k video content", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~1012, total~2384
[Token] Slice_Type_Determination: prompt~2768, completion~165, total~2933
[Token] Beamforming_Bandwidth: prompt~216, completion~3, total~219
[Token] Allocate_Resources: prompt~3067, completion~197, total~3264

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-16 14:38:22
Total Users: 6
Average Resource Utilization: 37.46%
eMBB Total Rate: 363.52 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 39.06 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  44.0/90 MHz       48.89%
URLLC          0  0/30 MHz          0%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 6 → eMBB Slice
CQI: 5, Bandwidth: 20.0 MHz, Rate: 123.87 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3472, completion~248, total~3720

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to track the location of a shipping container"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to track the location of a shipping container", "mmtc"),, ("i need to track the location of a shipping container", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~912, total~2286
[Token] Slice_Type_Determination: prompt~2719, completion~123, total~2842
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2993, completion~328, total~3321

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-16 14:38:37
Total Users: 7
Average Resource Utilization: 38.15%
eMBB Total Rate: 363.52 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 45.34 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           3  44.0/90 MHz                48.89%
URLLC          0  0/30 MHz                   0%
mMTC           4  5.6000000000000005/10 MHz  56.00%

New User Allocation:
User 7 → mMTC Slice
CQI: 6, Bandwidth: 0.9 MHz, Rate: 6.28 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3546, completion~502, total~4048

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my wearable device needs to upload health data periodically", "mmtc"),, ("my wearable device needs to upload health data periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~831, total~2203
[Token] Slice_Type_Determination: prompt~2626, completion~118, total~2744
[Token] Beamforming_Bandwidth: prompt~220, completion~2, total~222
[Token] Allocate_Resources: prompt~2891, completion~516, total~3407

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-16 14:38:51
Total Users: 8
Average Resource Utilization: 39.62%
eMBB Total Rate: 363.52 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 74.10 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  44.0/90 MHz       48.89%
URLLC          0  0/30 MHz          0%
mMTC           5  7.5/10 MHz        75.00%

New User Allocation:
User 8 → mMTC Slice
CQI: 15, Bandwidth: 1.9 MHz, Rate: 28.76 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1.9 |         28.76 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3633, completion~365, total~3998

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I need reliable connectivity for implanted medical devices"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable connectivity for implanted medical devices", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~1043, total~2413
[Token] Slice_Type_Determination: prompt~2827, completion~126, total~2953
[Token] Beamforming_Bandwidth: prompt~213, completion~2, total~215
[Token] Allocate_Resources: prompt~3066, completion~222, total~3288

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-16 14:39:04
Total Users: 9
Average Resource Utilization: 41.15%
eMBB Total Rate: 363.52 Mbps, URLLC Total Rate: 15.58 Mbps, mMTC Total Rate: 74.10 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  44.0/90 MHz       48.89%
URLLC          1  2.0/30 MHz        6.67%
mMTC           5  7.5/10 MHz        75.00%

New User Allocation:
User 9 → URLLC Slice
CQI: 7, Bandwidth: 2.0 MHz, Rate: 15.58 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1.9 |         28.76 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3516, completion~363, total~3879

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to check weather forecasts"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to check weather forecasts", "embb"),)
[Token] Intent_Analysis: prompt~1366, completion~1012, total~2378
[Token] Slice_Type_Determination: prompt~2801, completion~121, total~2922
WARNING: LLM didn't provide explicit slice recommendation. Using knowledge base recommendation.
[Token] Beamforming_Bandwidth: prompt~213, completion~2, total~215
[Token] Allocate_Resources: prompt~3059, completion~167, total~3226

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-16 14:39:26
Total Users: 10
Average Resource Utilization: 56.54%
eMBB Total Rate: 487.39 Mbps, URLLC Total Rate: 15.58 Mbps, mMTC Total Rate: 74.10 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  64.0/90 MHz       71.11%
URLLC          1  2.0/30 MHz        6.67%
mMTC           5  7.5/10 MHz        75.00%

New User Allocation:
User 10 → eMBB Slice
CQI: 5, Bandwidth: 20.0 MHz, Rate: 123.87 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1.9 |         28.76 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3499, completion~679, total~4178

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of city-wide smart streetlights", "mmtc"),, ("i need to check the status of city-wide smart streetlights", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~907, total~2285
[Token] Slice_Type_Determination: prompt~2711, completion~111, total~2822
[Token] Beamforming_Bandwidth: prompt~218, completion~2, total~220
[Token] Allocate_Resources: prompt~2977, completion~400, total~3377

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-16 14:40:00
Total Users: 11
Average Resource Utilization: 57.23%
eMBB Total Rate: 487.39 Mbps, URLLC Total Rate: 15.58 Mbps, mMTC Total Rate: 80.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  64.0/90 MHz       71.11%
URLLC          1  2.0/30 MHz        6.67%
mMTC           6  8.4/10 MHz        84.00%

New User Allocation:
User 11 → mMTC Slice
CQI: 6, Bandwidth: 0.9 MHz, Rate: 6.28 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1.9 |         28.76 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3601, completion~738, total~4339

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need microsecond-level latency for high-frequency tradin"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need microsecond-level latency for high-frequency trading", "urllc"),)
[Token] Intent_Analysis: prompt~1376, completion~990, total~2366
[Token] Slice_Type_Determination: prompt~2792, completion~125, total~2917
[Token] Beamforming_Bandwidth: prompt~218, completion~2, total~220
[Token] Allocate_Resources: prompt~3032, completion~202, total~3234

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-16 14:40:27
Total Users: 12
Average Resource Utilization: 58.77%
eMBB Total Rate: 487.39 Mbps, URLLC Total Rate: 25.11 Mbps, mMTC Total Rate: 80.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  64.0/90 MHz       71.11%
URLLC          2  4.0/30 MHz        13.33%
mMTC           6  8.4/10 MHz        84.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 3, Bandwidth: 2.0 MHz, Rate: 9.53 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1.9 |         28.76 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3463, completion~245, total~3708

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I want to use holographic communication"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use holographic communication", "embb"),)
[Token] Intent_Analysis: prompt~1368, completion~1202, total~2570
[Token] Slice_Type_Determination: prompt~2967, completion~114, total~3081
[Token] Beamforming_Bandwidth: prompt~214, completion~3, total~217
[Token] Allocate_Resources: prompt~3190, completion~202, total~3392

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-16 14:41:15
Total Users: 13
Average Resource Utilization: 68.0%
eMBB Total Rate: 601.57 Mbps, URLLC Total Rate: 25.11 Mbps, mMTC Total Rate: 80.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  76.0/90 MHz       84.44%
URLLC          2  4.0/30 MHz        13.33%
mMTC           6  8.4/10 MHz        84.00%

New User Allocation:
User 13 → eMBB Slice
CQI: 9, Bandwidth: 12.0 MHz, Rate: 114.18 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1.9 |         28.76 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3609, completion~299, total~3908

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 12
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control critical infrastructure with zero downtime", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~808, total~2180
[Token] Slice_Type_Determination: prompt~2599, completion~105, total~2704
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2819, completion~190, total~3009

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-16 14:41:45
Total Users: 14
Average Resource Utilization: 69.54%
eMBB Total Rate: 601.57 Mbps, URLLC Total Rate: 49.64 Mbps, mMTC Total Rate: 80.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  76.0/90 MHz       84.44%
URLLC          3  6.0/30 MHz        20.00%
mMTC           6  8.4/10 MHz        84.00%

New User Allocation:
User 14 → URLLC Slice
CQI: 12, Bandwidth: 2.0 MHz, Rate: 24.53 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1.9 |         28.76 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3234, completion~355, total~3589

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need real-time fraud detection for financial transactions"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need real-time fraud detection for financial transactions", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~859, total~2231
[Token] Slice_Type_Determination: prompt~2651, completion~76, total~2727
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2843, completion~157, total~3000

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-16 14:41:56
Total Users: 15
Average Resource Utilization: 71.08%
eMBB Total Rate: 601.57 Mbps, URLLC Total Rate: 65.22 Mbps, mMTC Total Rate: 80.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  76.0/90 MHz       84.44%
URLLC          4  8.0/30 MHz        26.67%
mMTC           6  8.4/10 MHz        84.00%

New User Allocation:
User 15 → URLLC Slice
CQI: 7, Bandwidth: 2.0 MHz, Rate: 15.58 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        2   |         15.58 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1.9 |         28.76 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3229, completion~260, total~3489

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need instant facial recognition for public security threats"
CQI: 10
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need instant facial recognition for public security threats", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~993, total~2365
[Token] Slice_Type_Determination: prompt~2777, completion~120, total~2897
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3008, completion~177, total~3185

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-16 14:42:13
Total Users: 16
Average Resource Utilization: 72.62%
eMBB Total Rate: 601.57 Mbps, URLLC Total Rate: 86.05 Mbps, mMTC Total Rate: 80.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  76.0/90 MHz       84.44%
URLLC          5  10.0/30 MHz       33.33%
mMTC           6  8.4/10 MHz        84.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 10, Bandwidth: 2.0 MHz, Rate: 20.83 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        2   |         20.83 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1.9 |         28.76 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3410, completion~362, total~3772

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "My smart trash can needs to signal that it's full"
CQI: 13
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart trash can needs to signal that it's full", "mmtc"),)
[Token] Intent_Analysis: prompt~1376, completion~806, total~2182
[Token] Slice_Type_Determination: prompt~2622, completion~111, total~2733
[Token] Beamforming_Bandwidth: prompt~237, completion~2, total~239
[Token] Allocate_Resources: prompt~2886, completion~487, total~3373

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-16 14:42:32
Total Users: 17
Average Resource Utilization: 73.31%
eMBB Total Rate: 601.57 Mbps, URLLC Total Rate: 86.05 Mbps, mMTC Total Rate: 92.27 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  76.0/90 MHz       84.44%
URLLC          5  10.0/30 MHz       33.33%
mMTC           7  9.3/10 MHz        93.00%

New User Allocation:
User 17 → mMTC Slice
CQI: 13, Bandwidth: 0.9 MHz, Rate: 11.89 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        2   |         20.83 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1.9 |         28.76 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3593, completion~866, total~4459

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~1076, total~2446
[Token] Slice_Type_Determination: prompt~2866, completion~125, total~2991
[Token] Beamforming_Bandwidth: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3099, completion~190, total~3289

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-16 14:42:50
Total Users: 18
Average Resource Utilization: 75.62%
eMBB Total Rate: 601.57 Mbps, URLLC Total Rate: 100.34 Mbps, mMTC Total Rate: 92.27 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  76.0/90 MHz       84.44%
URLLC          6  13.0/30 MHz       43.33%
mMTC           7  9.3/10 MHz        93.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 3, Bandwidth: 3.0 MHz, Rate: 14.29 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        2   |         20.83 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        1.9 |          9.05 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1.9 |         28.76 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3517, completion~366, total~3883

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to monitor water level in a reservoir"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor water level in a reservoir", "mmtc"),, ("i need to monitor water level in a reservoir periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~912, total~2284
[Token] Slice_Type_Determination: prompt~2717, completion~119, total~2836
[Token] Beamforming_Bandwidth: prompt~220, completion~4, total~224
[Token] Allocate_Resources: prompt~3045, completion~357, total~3402

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-16 14:43:06
Total Users: 19
Average Resource Utilization: 76.15%
eMBB Total Rate: 601.57 Mbps, URLLC Total Rate: 100.34 Mbps, mMTC Total Rate: 93.53 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  76.0/90 MHz       84.44%
URLLC          6  13.0/30 MHz       43.33%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 19 → mMTC Slice
CQI: 3, Bandwidth: 0.9 MHz, Rate: 4.29 Mbps, Latency: 1000.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.20000000000000073 MHz
  User 8: 1.9 → 1.6999999999999993 MHz, Rate: 28.76 → 25.73 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        2   |         20.83 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       12   |        135.98 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
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
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1.7 |         25.73 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3713, completion~343, total~4056

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I need to make a high-quality voice call"
CQI: 10
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to make a high-quality voice call", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~1329, total~2701
[Token] Slice_Type_Determination: prompt~3098, completion~152, total~3250
LLM recommended URLLC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~3309, completion~203, total~3512
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3732, completion~177, total~3909

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-16 14:43:26
Total Users: 20
Average Resource Utilization: 86.92%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 100.34 Mbps, mMTC Total Rate: 93.53 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          6  13.0/30 MHz       43.33%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 10, Bandwidth: 20.0 MHz, Rate: 208.28 Mbps, Latency: 30.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 6.0 MHz
  User 2: 12.0 → 9.0 MHz, Rate: 135.98 → 101.99 Mbps, User 6: 20.0 → 17.0 MHz, Rate: 123.87 → 105.29 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        2   |         20.83 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        114.18 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
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
|         3 | mMTC    |    13 |        1.9 |         25.1  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1.7 |         25.73 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4299, completion~254, total~4553

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to use holographic communication"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use holographic communication", "embb"),)
[Token] Intent_Analysis: prompt~1368, completion~1227, total~2595
[Token] Slice_Type_Determination: prompt~2969, completion~133, total~3102
[Token] Beamforming_Bandwidth: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~3281, completion~215, total~3496
[Token] Failure_Evaluation: prompt~3563, completion~398, total~3961

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
[Token] Intent_Analysis: prompt~1376, completion~798, total~2174
[Token] Slice_Type_Determination: prompt~2613, completion~123, total~2736
[Token] Beamforming_Bandwidth: prompt~217, completion~4, total~221
[Token] Allocate_Resources: prompt~2975, completion~292, total~3267

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-16 14:44:10
Total Users: 21
Average Resource Utilization: 86.92%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 100.34 Mbps, mMTC Total Rate: 90.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          6  13.0/30 MHz       43.33%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 11, Bandwidth: 0.9 MHz, Rate: 10.20 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 0.9 MHz
  User 8: 1.6999999999999993 → 1.0 MHz, Rate: 25.73 → 15.14 Mbps, User 3: 1.9 → 1.6999999999999993 MHz, Rate: 25.10 → 22.46 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        2   |         20.83 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
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
|         3 | mMTC    |    13 |        1.7 |         22.46 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |         15.14 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3631, completion~308, total~3939

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 1
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to detect and isolate power grid faults instantly", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~1217, total~2591
[Token] Slice_Type_Determination: prompt~3005, completion~120, total~3125
[Token] Beamforming_Bandwidth: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~3240, completion~164, total~3404

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-16 14:44:43
Total Users: 22
Average Resource Utilization: 88.46%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 107.42 Mbps, mMTC Total Rate: 90.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          7  15.0/30 MHz       50.00%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 23 → URLLC Slice
CQI: 1, Bandwidth: 2.0 MHz, Rate: 7.08 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        2   |         20.83 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        2   |          7.08 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
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
|         3 | mMTC    |    13 |        1.7 |         22.46 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |         15.14 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3630, completion~225, total~3855

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I want to use remote surgery equipment"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to use remote surgery equipment", "urllc"),)
[Token] Intent_Analysis: prompt~1368, completion~792, total~2160
[Token] Slice_Type_Determination: prompt~2589, completion~151, total~2740
[Token] Beamforming_Bandwidth: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~2849, completion~176, total~3025

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-16 14:44:58
Total Users: 23
Average Resource Utilization: 92.31%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 138.39 Mbps, mMTC Total Rate: 90.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          8  20.0/30 MHz       66.67%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 5, Bandwidth: 5.0 MHz, Rate: 30.97 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        2   |         20.83 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        2   |          7.08 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         30.97 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
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
|         3 | mMTC    |    13 |        1.7 |         22.46 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |         15.14 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3251, completion~299, total~3550

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I want to stream music while browsing social media"
CQI: 2
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to stream music while browsing social media", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~1156, total~2528
[Token] Slice_Type_Determination: prompt~2922, completion~145, total~3067
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3251, completion~470, total~3721
[Token] Failure_Evaluation: prompt~3782, completion~414, total~4196

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
[Token] Intent_Analysis: prompt~1380, completion~1129, total~2509
[Token] Slice_Type_Determination: prompt~2920, completion~113, total~3033
[Token] Beamforming_Bandwidth: prompt~220, completion~2, total~222
[Token] Allocate_Resources: prompt~3144, completion~175, total~3319

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-16 14:45:35
Total Users: 24
Average Resource Utilization: 94.62%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 159.31 Mbps, mMTC Total Rate: 90.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          9  23.0/30 MHz       76.67%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 6, Bandwidth: 3.0 MHz, Rate: 20.92 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        2   |         20.83 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        2   |          7.08 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         30.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        3   |         20.92 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
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
|         3 | mMTC    |    13 |        1.7 |         22.46 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |         15.14 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3543, completion~389, total~3932

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need to transmit real-time patient vital signs during critical care"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to transmit real-time patient vital signs during critical care", "urllc"),)
[Token] Intent_Analysis: prompt~1378, completion~1110, total~2488
[Token] Slice_Type_Determination: prompt~2910, completion~96, total~3006
[Token] Beamforming_Bandwidth: prompt~219, completion~2, total~221
[Token] Allocate_Resources: prompt~3119, completion~177, total~3296

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-16 14:45:59
Total Users: 25
Average Resource Utilization: 96.15%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 168.84 Mbps, mMTC Total Rate: 90.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         10  25.0/30 MHz       83.33%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 27 → URLLC Slice
CQI: 3, Bandwidth: 2.0 MHz, Rate: 9.53 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        2   |         20.83 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        2   |          7.08 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         30.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        2   |          9.53 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
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
|         3 | mMTC    |    13 |        1.7 |         22.46 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |         15.14 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3524, completion~418, total~3942

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~1079, total~2449
[Token] Slice_Type_Determination: prompt~2881, completion~109, total~2990
[Token] Beamforming_Bandwidth: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3102, completion~191, total~3293

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-16 14:46:29
Total Users: 26
Average Resource Utilization: 97.69%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 179.75 Mbps, mMTC Total Rate: 90.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         11  27.0/30 MHz       90.00%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 28 → URLLC Slice
CQI: 4, Bandwidth: 2.0 MHz, Rate: 10.91 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        2   |         20.83 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        2   |          7.08 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         30.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | URLLC   |     4 |        2   |         10.91 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
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
|         3 | mMTC    |    13 |        1.7 |         22.46 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |         15.14 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3522, completion~415, total~3937

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need to participate in an online multiplayer game"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to participate in an online multiplayer game", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~1147, total~2519
[Token] Slice_Type_Determination: prompt~2920, completion~149, total~3069
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3178, completion~189, total~3367

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-16 14:46:50
Total Users: 27
Average Resource Utilization: 99.23%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 210.02 Mbps, mMTC Total Rate: 90.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         12  29.0/30 MHz       96.67%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 29 → URLLC Slice
CQI: 15, Bandwidth: 2.0 MHz, Rate: 30.27 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        2   |         24.53 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        2   |         20.83 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        2   |          7.08 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         30.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | URLLC   |    15 |        2   |         30.27 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        2   |         15.58 |              1 |          |
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
|         3 | mMTC    |    13 |        1.7 |         22.46 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |         15.14 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3593, completion~322, total~3915

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I want to browse websites and check email"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~965, total~2335
[Token] Slice_Type_Determination: prompt~2741, completion~98, total~2839
[Token] Beamforming_Bandwidth: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3026, completion~193, total~3219
[Token] Failure_Evaluation: prompt~3294, completion~355, total~3649

----------------------------------------
ALLOCATION FAILED FOR USER 30
----------------------------------------
Request: I want to browse websites and check email
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

Detailed Slice Utilization Values:
eMBB utils: [0.0, 13.33, 13.33, 13.33, 26.67, 48.89, 48.89, 48.89, 48.89, 71.11, 71.11, 71.11, 84.44, 84.44, 84.44, 84.44, 84.44, 84.44, 84.44, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.67, 6.67, 6.67, 13.33, 13.33, 20.0, 26.67, 33.33, 33.33, 43.33, 43.33, 43.33, 43.33, 50.0, 66.67, 76.67, 83.33, 90.0, 96.67]
mMTC utils: [19.0, 19.0, 38.0, 47.0, 47.0, 47.0, 56.0, 75.0, 75.0, 75.0, 84.0, 84.0, 84.0, 84.0, 84.0, 84.0, 93.0, 93.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | mMTC    | mMTC           | Yes            |     3 |        1.9 |          9.05 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | eMBB    | eMBB           | Yes            |    11 |       12   |        135.98 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | mMTC    | mMTC           | Yes            |    13 |        1.9 |         25.1  |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | mMTC    | mMTC           | Yes            |     4 |        0.9 |          4.91 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | eMBB    | eMBB           | Yes            |     8 |       12   |        103.67 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | eMBB    | eMBB           | Yes            |     5 |       20   |        123.87 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | mMTC    | mMTC           | Yes            |     6 |        0.9 |          6.28 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | mMTC    | mMTC           | Yes            |    15 |        1.9 |         28.76 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | URLLC   | URLLC          | Yes            |     7 |        2   |         15.58 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | eMBB    | eMBB           | Yes            |     5 |       20   |        123.87 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | mMTC    | mMTC           | Yes            |     6 |        0.9 |          6.28 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     3 |        2   |          9.53 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | eMBB    | eMBB           | Yes            |     9 |       12   |        114.18 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | URLLC   | URLLC          | Yes            |    12 |        2   |         24.53 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | URLLC   | URLLC          | Yes            |     7 |        2   |         15.58 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |    10 |        2   |         20.83 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | mMTC    | mMTC           | Yes            |    13 |        0.9 |         11.89 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     3 |        3   |         14.29 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | mMTC    | mMTC           | Yes            |     3 |        0.9 |          4.29 |           1000 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    10 |       20   |        208.28 |             30 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Failed   | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | mMTC    | mMTC           | Yes            |    11 |        0.9 |         10.2  |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | URLLC   | URLLC          | Yes            |     1 |        2   |          7.08 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     5 |        5   |         30.97 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Failed   | eMBB    | eMBB           | Yes            |     2 |       20   |         82.49 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | URLLC          | Yes            |     6 |        3   |         20.92 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | URLLC   | URLLC          | Yes            |     3 |        2   |          9.53 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | URLLC   | URLLC          | Yes            |     4 |        2   |         10.91 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | URLLC   | URLLC          | Yes            |    15 |        2   |         30.27 |              5 | No         |
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
Average eMBB utilization: 69.13%
Average URLLC utilization: 29.51%
Average mMTC utilization: 77.33%

Weighted Average Utilization: 60.62%

Transmission Rate Statistics:
Final eMBB total rate: 757.28 Mbps
Final URLLC total rate: 210.02 Mbps
Final mMTC total rate: 90.50 Mbps

Resource Utilization:
Average resource utilization: 99.23%

Results exported to F:\code\wirelessagent\run_results\batch_run\fkb\qwen3-coder-next\network_slicing_results_TJU_gym_qwen3-coder-next.csv

✓ TJU_gym 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\fkb\qwen3-coder-next\network_slicing_results_TJU_gym_qwen3-coder-next.csv