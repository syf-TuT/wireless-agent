============================================================
场景 4/5: TJU_west
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_west.csv
输出: F:\code\wirelessagent\run_results\batch_run\fkb\qwen3-coder-plus\network_slicing_results_TJU_west_qwen3-coder-plus.csv
------------------------------------------------------------
Starting network slice management system with CSV-based user testing...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I need to download large files"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to download large files", "embb"),)
[Token] Intent_Analysis: prompt~1366, completion~413, total~1779
[Token] Slice_Type_Determination: prompt~2221, completion~59, total~2280
[Token] Beamforming_Bandwidth: prompt~211, completion~108, total~319
[Token] Allocate_Resources: prompt~2423, completion~110, total~2533

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-16 15:58:27
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
[Token] Network_Evaluation: prompt~2750, completion~240, total~2990

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~495, total~1869
[Token] Slice_Type_Determination: prompt~2315, completion~63, total~2378
[Token] Beamforming_Bandwidth: prompt~214, completion~102, total~316
[Token] Allocate_Resources: prompt~2533, completion~173, total~2706

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-16 15:58:54
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
[Token] Network_Evaluation: prompt~2929, completion~357, total~3286

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control precision cnc machines with zero tolerance for delay", "urllc"),)
[Token] Intent_Analysis: prompt~1378, completion~437, total~1815
[Token] Slice_Type_Determination: prompt~2260, completion~76, total~2336
[Token] Beamforming_Bandwidth: prompt~217, completion~115, total~332
[Token] Allocate_Resources: prompt~2451, completion~205, total~2656

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-16 15:59:24
Total Users: 3
Average Resource Utilization: 19.92%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 31.16 Mbps, mMTC Total Rate: 10.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          1  4.0/30 MHz        13.33%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 3 → URLLC Slice
CQI: 7, Bandwidth: 4.0 MHz, Rate: 31.16 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2884, completion~279, total~3163

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to send text messages and use messaging apps"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to send text messages and use messaging apps", "embb"),)
[Token] Intent_Analysis: prompt~1374, completion~546, total~1920
[Token] Slice_Type_Determination: prompt~2363, completion~71, total~2434
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~2502, completion~69, total~2571
[Token] Beamforming_Bandwidth: prompt~217, completion~113, total~330
[Token] Allocate_Resources: prompt~2715, completion~349, total~3064

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-16 15:59:57
Total Users: 4
Average Resource Utilization: 35.31%
eMBB Total Rate: 264.91 Mbps, URLLC Total Rate: 31.16 Mbps, mMTC Total Rate: 10.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          1  4.0/30 MHz        13.33%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 4 → eMBB Slice
CQI: 7, Bandwidth: 20.0 MHz, Rate: 155.80 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3302, completion~281, total~3583

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I need to use maps for basic navigation"
CQI: 3
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use maps for basic navigation", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~538, total~1908
[Token] Slice_Type_Determination: prompt~2345, completion~100, total~2445
[Token] Beamforming_Bandwidth: prompt~215, completion~88, total~303
[Token] Allocate_Resources: prompt~2585, completion~114, total~2699

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-16 16:00:28
Total Users: 5
Average Resource Utilization: 50.69%
eMBB Total Rate: 360.20 Mbps, URLLC Total Rate: 31.16 Mbps, mMTC Total Rate: 10.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          1  4.0/30 MHz        13.33%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 5 → eMBB Slice
CQI: 3, Bandwidth: 20.0 MHz, Rate: 95.29 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2927, completion~255, total~3182

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~470, total~1844
[Token] Slice_Type_Determination: prompt~2289, completion~84, total~2373
[Token] Beamforming_Bandwidth: prompt~216, completion~124, total~340
[Token] Allocate_Resources: prompt~2528, completion~557, total~3085

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-16 16:01:03
Total Users: 6
Average Resource Utilization: 51.38%
eMBB Total Rate: 360.20 Mbps, URLLC Total Rate: 31.16 Mbps, mMTC Total Rate: 16.65 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          1  4.0/30 MHz        13.33%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 6 → mMTC Slice
CQI: 6, Bandwidth: 0.9 MHz, Rate: 6.28 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3280, completion~368, total~3648

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need reliable connectivity for implanted medical devices"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable connectivity for implanted medical devices", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~454, total~1824
[Token] Slice_Type_Determination: prompt~2273, completion~66, total~2339
[Token] Beamforming_Bandwidth: prompt~215, completion~100, total~315
[Token] Allocate_Resources: prompt~2455, completion~278, total~2733

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-16 16:01:31
Total Users: 7
Average Resource Utilization: 54.46%
eMBB Total Rate: 360.20 Mbps, URLLC Total Rate: 59.05 Mbps, mMTC Total Rate: 16.65 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          2  8.0/30 MHz        26.67%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 6, Bandwidth: 4.0 MHz, Rate: 27.89 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2956, completion~415, total~3371

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I want to remotely access my work computer"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to remotely access my work computer", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~458, total~1828
[Token] Slice_Type_Determination: prompt~2273, completion~89, total~2362
LLM recommended URLLC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~2427, completion~95, total~2522
[Token] Beamforming_Bandwidth: prompt~215, completion~71, total~286
[Token] Allocate_Resources: prompt~2658, completion~233, total~2891

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-16 16:02:03
Total Users: 8
Average Resource Utilization: 69.85%
eMBB Total Rate: 469.31 Mbps, URLLC Total Rate: 59.05 Mbps, mMTC Total Rate: 16.65 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          2  8.0/30 MHz        26.67%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 8 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3167, completion~329, total~3496

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to use holographic communication"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use holographic communication", "embb"),)
[Token] Intent_Analysis: prompt~1368, completion~609, total~1977
[Token] Slice_Type_Determination: prompt~2406, completion~103, total~2509
[Token] Beamforming_Bandwidth: prompt~214, completion~120, total~334
[Token] Allocate_Resources: prompt~2620, completion~223, total~2843

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-16 16:02:33
Total Users: 9
Average Resource Utilization: 76.0%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 59.05 Mbps, mMTC Total Rate: 16.65 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC          2  8.0/30 MHz        26.67%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 9 → eMBB Slice
CQI: 15, Bandwidth: 8.0 MHz, Rate: 121.08 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3063, completion~398, total~3461

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control precision cnc machines with zero tolerance for delay", "urllc"),)
[Token] Intent_Analysis: prompt~1378, completion~399, total~1777
[Token] Slice_Type_Determination: prompt~2224, completion~68, total~2292
[Token] Beamforming_Bandwidth: prompt~219, completion~92, total~311
[Token] Allocate_Resources: prompt~2408, completion~211, total~2619

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-16 16:03:00
Total Users: 10
Average Resource Utilization: 79.08%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 119.59 Mbps, mMTC Total Rate: 16.65 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC          3  12.0/30 MHz       40.00%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 10 → URLLC Slice
CQI: 15, Bandwidth: 4.0 MHz, Rate: 60.54 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        4   |         60.54 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2845, completion~292, total~3137

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need emergency response coordination during a disaste"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need emergency response coordination during a disaster", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~396, total~1768
[Token] Slice_Type_Determination: prompt~2218, completion~65, total~2283
[Token] Beamforming_Bandwidth: prompt~216, completion~176, total~392
[Token] Allocate_Resources: prompt~2400, completion~290, total~2690

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-16 16:03:30
Total Users: 11
Average Resource Utilization: 82.15%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 138.65 Mbps, mMTC Total Rate: 16.65 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC          4  16.0/30 MHz       53.33%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 3, Bandwidth: 4.0 MHz, Rate: 19.06 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        4   |         60.54 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2913, completion~340, total~3253

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I want to use remote surgery equipment"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to use remote surgery equipment", "urllc"),)
[Token] Intent_Analysis: prompt~1368, completion~465, total~1833
[Token] Slice_Type_Determination: prompt~2278, completion~77, total~2355
[Token] Beamforming_Bandwidth: prompt~214, completion~122, total~336
[Token] Allocate_Resources: prompt~2471, completion~237, total~2708

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-16 16:03:58
Total Users: 12
Average Resource Utilization: 85.23%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 169.81 Mbps, mMTC Total Rate: 16.65 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC          5  20.0/30 MHz       66.67%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 7, Bandwidth: 4.0 MHz, Rate: 31.16 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        4   |         60.54 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2942, completion~331, total~3273

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need to participate in an online multiplayer game"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to participate in an online multiplayer game", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~611, total~1983
[Token] Slice_Type_Determination: prompt~2406, completion~97, total~2503
LLM recommended eMBB but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~2564, completion~59, total~2623
[Token] Beamforming_Bandwidth: prompt~216, completion~80, total~296
[Token] Allocate_Resources: prompt~2739, completion~197, total~2936

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-16 16:04:32
Total Users: 13
Average Resource Utilization: 87.54%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 193.18 Mbps, mMTC Total Rate: 16.65 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC          6  23.0/30 MHz       76.67%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 7, Bandwidth: 3.0 MHz, Rate: 23.37 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        4   |         60.54 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3223, completion~350, total~3573

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need to track the location of a shipping container"
CQI: 14
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to track the location of a shipping container", "mmtc"),, ("i need to track the location of a shipping container", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~511, total~1885
[Token] Slice_Type_Determination: prompt~2326, completion~86, total~2412
[Token] Beamforming_Bandwidth: prompt~216, completion~130, total~346
[Token] Allocate_Resources: prompt~2568, completion~142, total~2710

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-16 16:05:04
Total Users: 14
Average Resource Utilization: 88.23%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 193.18 Mbps, mMTC Total Rate: 29.40 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           5  88.0/90 MHz                97.78%
URLLC          6  23.0/30 MHz                76.67%
mMTC           3  3.6999999999999997/10 MHz  37.00%

New User Allocation:
User 14 → mMTC Slice
CQI: 14, Bandwidth: 0.9 MHz, Rate: 12.75 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        4   |         60.54 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2946, completion~291, total~3237

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[Token] Intent_Analysis: prompt~1376, completion~579, total~1955
[Token] Slice_Type_Determination: prompt~2378, completion~88, total~2466
[Token] Beamforming_Bandwidth: prompt~218, completion~106, total~324
[Token] Allocate_Resources: prompt~2648, completion~433, total~3081
[Token] Failure_Evaluation: prompt~3113, completion~300, total~3413

----------------------------------------
ALLOCATION FAILED FOR USER 15
----------------------------------------
Request: I need to use cloud-based AI services for image processing
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need to monitor water level in a reservoir"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor water level in a reservoir", "mmtc"),, ("i need to monitor water level in a reservoir periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~457, total~1829
[Token] Slice_Type_Determination: prompt~2276, completion~86, total~2362
[Token] Beamforming_Bandwidth: prompt~219, completion~120, total~339
[Token] Allocate_Resources: prompt~2518, completion~121, total~2639

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-16 16:06:06
Total Users: 15
Average Resource Utilization: 88.92%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 193.18 Mbps, mMTC Total Rate: 37.96 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC          6  23.0/30 MHz       76.67%
mMTC           4  4.6/10 MHz        46.00%

New User Allocation:
User 16 → mMTC Slice
CQI: 9, Bandwidth: 0.9 MHz, Rate: 8.56 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        4   |         60.54 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2871, completion~187, total~3058

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to monitor soil moisture levels in a large farm"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor soil moisture levels in a large farm", "mmtc"),)
[Token] Intent_Analysis: prompt~1376, completion~486, total~1862
[Token] Slice_Type_Determination: prompt~2308, completion~64, total~2372
[Token] Beamforming_Bandwidth: prompt~217, completion~96, total~313
[Token] Allocate_Resources: prompt~2528, completion~147, total~2675

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-16 16:06:31
Total Users: 16
Average Resource Utilization: 89.62%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 193.18 Mbps, mMTC Total Rate: 42.87 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC          6  23.0/30 MHz       76.67%
mMTC           5  5.5/10 MHz        55.00%

New User Allocation:
User 17 → mMTC Slice
CQI: 4, Bandwidth: 0.9 MHz, Rate: 4.91 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        4   |         60.54 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2908, completion~207, total~3115

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need instant alerts for life-threatening patient conditions"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need instant alerts for life-threatening patient conditions", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~427, total~1799
[Token] Slice_Type_Determination: prompt~2246, completion~59, total~2305
[Token] Beamforming_Bandwidth: prompt~216, completion~123, total~339
[Token] Allocate_Resources: prompt~2422, completion~257, total~2679

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-16 16:06:58
Total Users: 17
Average Resource Utilization: 91.92%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 214.10 Mbps, mMTC Total Rate: 42.87 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC          7  26.0/30 MHz       86.67%
mMTC           5  5.5/10 MHz        55.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 6, Bandwidth: 3.0 MHz, Rate: 20.92 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        4   |         60.54 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        3   |         20.92 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2904, completion~302, total~3206

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to download large files"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to download large files", "embb"),)
[Token] Intent_Analysis: prompt~1366, completion~505, total~1871
[Token] Slice_Type_Determination: prompt~2302, completion~96, total~2398
[Token] Beamforming_Bandwidth: prompt~213, completion~124, total~337
[Token] Allocate_Resources: prompt~2584, completion~385, total~2969
[Token] Failure_Evaluation: prompt~3020, completion~230, total~3250

----------------------------------------
ALLOCATION FAILED FOR USER 19
----------------------------------------
Request: I need to download large files
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I need to synchronize distributed financial ledgers instantly"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to synchronize distributed financial ledgers instantly", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~446, total~1818
[Token] Slice_Type_Determination: prompt~2268, completion~73, total~2341
[Token] Beamforming_Bandwidth: prompt~216, completion~114, total~330
[Token] Allocate_Resources: prompt~2457, completion~202, total~2659

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-16 16:07:59
Total Users: 18
Average Resource Utilization: 93.46%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 233.13 Mbps, mMTC Total Rate: 42.87 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC          8  28.0/30 MHz       93.33%
mMTC           5  5.5/10 MHz        55.00%

New User Allocation:
User 20 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 19.03 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        4   |         60.54 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2888, completion~354, total~3242

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~460, total~1830
[Token] Slice_Type_Determination: prompt~2276, completion~69, total~2345
[Token] Beamforming_Bandwidth: prompt~215, completion~179, total~394
[Token] Allocate_Resources: prompt~2461, completion~243, total~2704

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-16 16:08:31
Total Users: 19
Average Resource Utilization: 94.23%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 240.10 Mbps, mMTC Total Rate: 42.87 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC          9  29.0/30 MHz       96.67%
mMTC           5  5.5/10 MHz        55.00%

New User Allocation:
User 21 → URLLC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 6.97 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        4   |         60.54 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        1   |          6.97 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2937, completion~267, total~3204

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "Smart city parking sensor reporting availability"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("smart city parking sensor reporting availability", "mmtc"),)
[Token] Intent_Analysis: prompt~1366, completion~483, total~1849
[Token] Slice_Type_Determination: prompt~2294, completion~108, total~2402
[Token] Beamforming_Bandwidth: prompt~212, completion~98, total~310
[Token] Allocate_Resources: prompt~2553, completion~168, total~2721

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-16 16:09:00
Total Users: 20
Average Resource Utilization: 95.69%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 240.10 Mbps, mMTC Total Rate: 57.67 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC          9  29.0/30 MHz       96.67%
mMTC           6  7.4/10 MHz        74.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 7, Bandwidth: 1.9 MHz, Rate: 14.80 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        4   |         60.54 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        1.9 |         14.8  |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2948, completion~322, total~3270

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control critical infrastructure with zero downtime", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~419, total~1791
[Token] Slice_Type_Determination: prompt~2240, completion~78, total~2318
[Token] Beamforming_Bandwidth: prompt~216, completion~112, total~328
[Token] Allocate_Resources: prompt~2434, completion~247, total~2681

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-16 16:09:29
Total Users: 21
Average Resource Utilization: 96.46%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 245.56 Mbps, mMTC Total Rate: 57.67 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC         10  30.0/30 MHz       100.00%
mMTC           6  7.4/10 MHz        74.00%

New User Allocation:
User 23 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        4   |         60.54 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~2907, completion~329, total~3236

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to monitor IoT sensors in real-time"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to monitor iot sensors in real-time", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~438, total~1810
[Token] Slice_Type_Determination: prompt~2255, completion~65, total~2320
LLM recommended mMTC but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~2389, completion~79, total~2468
[Token] Beamforming_Bandwidth: prompt~216, completion~91, total~307
[Token] Allocate_Resources: prompt~2633, completion~264, total~2897

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-16 16:09:59
Total Users: 22
Average Resource Utilization: 96.46%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 235.89 Mbps, mMTC Total Rate: 57.67 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC         11  30.0/30 MHz       100.00%
mMTC           6  7.4/10 MHz        74.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 10: 4.0 → 3.0 MHz, Rate: 60.54 → 45.41 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        3   |         45.41 |              3 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3257, completion~296, total~3553

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "My smart trash can needs to signal that it's full"
CQI: 5
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart trash can needs to signal that it's full", "mmtc"),)
[Token] Intent_Analysis: prompt~1376, completion~440, total~1816
[Token] Slice_Type_Determination: prompt~2265, completion~67, total~2332
[Token] Beamforming_Bandwidth: prompt~237, completion~217, total~454
[Token] Allocate_Resources: prompt~2488, completion~436, total~2924

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-16 16:10:34
Total Users: 23
Average Resource Utilization: 97.15%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 235.89 Mbps, mMTC Total Rate: 63.24 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC         11  30.0/30 MHz       100.00%
mMTC           7  8.3/10 MHz        83.00%

New User Allocation:
User 25 → mMTC Slice
CQI: 5, Bandwidth: 0.9 MHz, Rate: 5.57 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        3   |         45.41 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3139, completion~352, total~3491

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to monitor my home security cameras remotely", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~441, total~1813
[Token] Slice_Type_Determination: prompt~2253, completion~96, total~2349
LLM recommended eMBB but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~2418, completion~93, total~2511
[Token] Beamforming_Bandwidth: prompt~216, completion~83, total~299
[Token] Allocate_Resources: prompt~2675, completion~179, total~2854

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-16 16:11:03
Total Users: 24
Average Resource Utilization: 97.15%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 228.54 Mbps, mMTC Total Rate: 63.24 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC         12  30.0/30 MHz       100.00%
mMTC           7  8.3/10 MHz        83.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 10: 3.0 → 2.0 MHz, Rate: 45.41 → 30.27 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     7 |        1   |          7.79 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3218, completion~289, total~3507

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("industrial equipment monitoring sensor data", "mmtc"))
[Token] Intent_Analysis: prompt~1364, completion~500, total~1864
[Token] Slice_Type_Determination: prompt~2310, completion~88, total~2398
[Token] Beamforming_Bandwidth: prompt~231, completion~80, total~311
[Token] Allocate_Resources: prompt~2549, completion~610, total~3159

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-16 16:11:37
Total Users: 25
Average Resource Utilization: 97.85%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 228.54 Mbps, mMTC Total Rate: 68.15 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  88.0/90 MHz               97.78%
URLLC         12  30.0/30 MHz               100.00%
mMTC           8  9.200000000000001/10 MHz  92.00%

New User Allocation:
User 27 → mMTC Slice
CQI: 4, Bandwidth: 0.9 MHz, Rate: 4.91 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        1.9 |         14.8  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     4 |        0.9 |          4.91 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3352, completion~385, total~3737

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart parking sensor needs to report if the spot is free", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~423, total~1801
[Token] Slice_Type_Determination: prompt~2247, completion~86, total~2333
[Token] Beamforming_Bandwidth: prompt~223, completion~311, total~534
[Token] Allocate_Resources: prompt~2545, completion~173, total~2718

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-16 16:12:10
Total Users: 26
Average Resource Utilization: 98.46%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 228.54 Mbps, mMTC Total Rate: 75.93 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC         12  30.0/30 MHz       100.00%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 9, Bandwidth: 0.9 MHz, Rate: 8.56 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.10000000000000109 MHz
  User 22: 1.9 → 1.799999999999999 MHz, Rate: 14.80 → 14.02 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.9 |         10.37 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        1.8 |         14.02 |            500 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     9 |        0.9 |          8.56 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3037, completion~267, total~3304

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I want to stream music while browsing social media"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to stream music while browsing social media", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~583, total~1955
[Token] Slice_Type_Determination: prompt~2380, completion~82, total~2462
[Token] Beamforming_Bandwidth: prompt~216, completion~93, total~309
[Token] Allocate_Resources: prompt~2645, completion~370, total~3015
[Token] Failure_Evaluation: prompt~3071, completion~251, total~3322

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
Knowledge Base recommended slice: mMTC (Knowledge base match: ("industrial equipment monitoring sensor data", "mmtc"))
[Token] Intent_Analysis: prompt~1364, completion~453, total~1817
[Token] Slice_Type_Determination: prompt~2265, completion~79, total~2344
LLM recommended URLLC but knowledge base recommended mMTC, using knowledge base recommendation
[Token] Intent_Override: prompt~2408, completion~83, total~2491
[Token] Beamforming_Bandwidth: prompt~211, completion~65, total~276
[Token] Allocate_Resources: prompt~2733, completion~349, total~3082

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-16 16:13:05
Total Users: 27
Average Resource Utilization: 98.46%
eMBB Total Rate: 590.39 Mbps, URLLC Total Rate: 228.54 Mbps, mMTC Total Rate: 79.35 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  88.0/90 MHz       97.78%
URLLC         12  30.0/30 MHz       100.00%
mMTC          10  10.0/10 MHz       100.00%

New User Allocation:
User 30 → mMTC Slice
CQI: 11, Bandwidth: 0.9 MHz, Rate: 10.20 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 0.9 MHz
  User 22: 1.799999999999999 → 1.0 MHz, Rate: 14.02 → 7.79 Mbps, User 2: 1.9 → 1.799999999999999 MHz, Rate: 10.37 → 9.82 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        2   |         30.27 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        4   |         19.06 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        4   |         31.16 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        3   |         23.37 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        4   |         31.16 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        4   |         27.89 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |        8   |        121.08 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        1.8 |          9.82 |            500 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        1   |          7.79 |            500 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | mMTC    |    11 |        0.9 |         10.2  |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3463, completion~472, total~3935

Detailed Slice Utilization Values:
eMBB utils: [22.22, 22.22, 22.22, 44.44, 66.67, 66.67, 66.67, 88.89, 97.78, 97.78, 97.78, 97.78, 97.78, 97.78, 97.78, 97.78, 97.78, 97.78, 97.78, 97.78, 97.78, 97.78, 97.78, 97.78, 97.78, 97.78, 97.78]
URLLC utils: [0.0, 0.0, 13.33, 13.33, 13.33, 13.33, 26.67, 26.67, 26.67, 40.0, 53.33, 66.67, 76.67, 76.67, 76.67, 76.67, 86.67, 93.33, 96.67, 96.67, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
mMTC utils: [0.0, 19.0, 19.0, 19.0, 19.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 37.0, 46.0, 55.0, 55.0, 55.0, 55.0, 74.0, 74.0, 74.0, 83.0, 83.0, 92.0, 100.0, 100.0]

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
|         3 | Success  | URLLC   | URLLC          | Yes            |     7 |        4   |         31.16 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | eMBB    | eMBB           | Yes            |     3 |       20   |         95.29 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | mMTC    | mMTC           | Yes            |     6 |        0.9 |          6.28 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |     6 |        4   |         27.89 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | eMBB    | eMBB           | Yes            |    15 |        8   |        121.08 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | URLLC   | URLLC          | Yes            |    15 |        4   |         60.54 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |     3 |        4   |         19.06 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     7 |        4   |         31.16 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | URLLC          | Yes            |     7 |        3   |         23.37 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | mMTC    | mMTC           | Yes            |    14 |        0.9 |         12.75 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Failed   | eMBB    | eMBB           | Yes            |    15 |       20   |        302.7  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | mMTC    | mMTC           | Yes            |     9 |        0.9 |          8.56 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | mMTC    | mMTC           | Yes            |     4 |        0.9 |          4.91 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     6 |        3   |         20.92 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Failed   | eMBB    | eMBB           | Yes            |     8 |       20   |        172.78 |             80 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | URLLC   | URLLC          | Yes            |     9 |        2   |         19.03 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | URLLC   | URLLC          | Yes            |     6 |        1   |          6.97 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | mMTC    | mMTC           | Yes            |     7 |        1.9 |         14.8  |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          5.46 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          5.46 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | mMTC    | mMTC           | Yes            |     5 |        0.9 |          5.57 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | eMBB           | No             |     7 |        1   |          7.79 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | mMTC    | mMTC           | Yes            |     4 |        0.9 |          4.91 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | mMTC    | mMTC           | Yes            |     9 |        0.9 |          8.56 |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Failed   | eMBB    | eMBB           | Yes            |     5 |       20   |        123.87 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Success  | mMTC    | mMTC           | Yes            |    11 |        0.9 |         10.2  |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 27/30 (90.0%)

Intent Understanding Evaluation:
Correctly identified intents: 29/30
Intent understanding rate: 96.7%

Workload Balancing Statistics:
Users with workload balancing: 0/30
Workload balancing rate: 0.0%

Slice Utilization Statistics:
Average eMBB utilization: 83.62%
Average URLLC utilization: 61.98%
Average mMTC utilization: 47.52%

Weighted Average Utilization: 75.85%

Transmission Rate Statistics:
Final eMBB total rate: 590.39 Mbps
Final URLLC total rate: 228.54 Mbps
Final mMTC total rate: 79.35 Mbps

Resource Utilization:
Average resource utilization: 98.46%

Results exported to F:\code\wirelessagent\run_results\batch_run\fkb\qwen3-coder-plus\network_slicing_results_TJU_west_qwen3-coder-plus.csv

✓ TJU_west 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\fkb\qwen3-coder-plus\network_slicing_results_TJU_west_qwen3-coder-plus.csv