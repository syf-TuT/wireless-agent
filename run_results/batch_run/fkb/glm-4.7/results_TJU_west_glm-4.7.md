
============================================================
场景 3/5: TJU_west
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_west.csv
输出: F:\code\wirelessagent\run_results\batch_run\fkb\glm-4.7\network_slicing_results_TJU_west_glm-4.7.csv
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
[Token] Intent_Analysis: prompt~1366, completion~1192, total~2558
[Token] Slice_Type_Determination: prompt~3037, completion~248, total~3285
WARNING: LLM didn't provide explicit slice recommendation. Using knowledge base recommendation.
[Token] Beamforming_Bandwidth: prompt~211, completion~428, total~639
[Token] Allocate_Resources: prompt~3434, completion~881, total~4315

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-21 20:05:13
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
[Token] Network_Evaluation: prompt~4625, completion~784, total~5409

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~1119, total~2493
[Token] Slice_Type_Determination: prompt~2949, completion~244, total~3193
[Token] Beamforming_Bandwidth: prompt~214, completion~880, total~1094
[Token] Allocate_Resources: prompt~3348, completion~1051, total~4399

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-21 20:06:21
Total Users: 2
Average Resource Utilization: 16.08%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 4.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          0  0/30 MHz          0%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 2 → mMTC Slice
CQI: 4, Bandwidth: 0.9 MHz, Rate: 4.91 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4641, completion~964, total~5605

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control precision cnc machines with zero tolerance for delay", "urllc"),)
[Token] Intent_Analysis: prompt~1378, completion~1093, total~2471
[Token] Slice_Type_Determination: prompt~2963, completion~223, total~3186
[Token] Beamforming_Bandwidth: prompt~217, completion~752, total~969
[Token] Allocate_Resources: prompt~3313, completion~850, total~4163

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-21 20:07:22
Total Users: 3
Average Resource Utilization: 16.85%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 7.79 Mbps, mMTC Total Rate: 4.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          1  1.0/30 MHz        3.33%
mMTC           1  0.9/10 MHz        9.00%

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
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4441, completion~1144, total~5585

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to send text messages and use messaging apps"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to send text messages and use messaging apps", "embb"),)
[Token] Intent_Analysis: prompt~1374, completion~1060, total~2434
[Token] Slice_Type_Determination: prompt~2898, completion~231, total~3129
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~3201, completion~391, total~3592
[Token] Beamforming_Bandwidth: prompt~217, completion~481, total~698
[Token] Allocate_Resources: prompt~3744, completion~752, total~4496

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-21 20:08:31
Total Users: 4
Average Resource Utilization: 32.23%
eMBB Total Rate: 264.91 Mbps, URLLC Total Rate: 7.79 Mbps, mMTC Total Rate: 4.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          1  1.0/30 MHz        3.33%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 4 → eMBB Slice
CQI: 7, Bandwidth: 20.0 MHz, Rate: 155.80 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4804, completion~1239, total~6043

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I need to use maps for basic navigation"
CQI: 3
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use maps for basic navigation", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~2103, total~3473
[Token] Slice_Type_Determination: prompt~3952, completion~617, total~4569
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~4640, completion~492, total~5132
[Token] Beamforming_Bandwidth: prompt~215, completion~630, total~845
[Token] Allocate_Resources: prompt~5281, completion~878, total~6159

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-21 20:10:02
Total Users: 5
Average Resource Utilization: 47.62%
eMBB Total Rate: 360.20 Mbps, URLLC Total Rate: 7.79 Mbps, mMTC Total Rate: 4.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          1  1.0/30 MHz        3.33%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 5 → eMBB Slice
CQI: 3, Bandwidth: 20.0 MHz, Rate: 95.29 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~6478, completion~1519, total~7997

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~1330, total~2704
[Token] Slice_Type_Determination: prompt~3171, completion~279, total~3450
[Token] Beamforming_Bandwidth: prompt~216, completion~620, total~836
[Token] Allocate_Resources: prompt~3607, completion~1062, total~4669

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-21 20:11:16
Total Users: 6
Average Resource Utilization: 48.31%
eMBB Total Rate: 360.20 Mbps, URLLC Total Rate: 7.79 Mbps, mMTC Total Rate: 11.19 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          1  1.0/30 MHz        3.33%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 6 → mMTC Slice
CQI: 6, Bandwidth: 0.9 MHz, Rate: 6.28 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4925, completion~993, total~5918

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need reliable connectivity for implanted medical devices"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable connectivity for implanted medical devices", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~1232, total~2602
[Token] Slice_Type_Determination: prompt~3084, completion~394, total~3478
[Token] Beamforming_Bandwidth: prompt~215, completion~720, total~935
[Token] Allocate_Resources: prompt~3608, completion~795, total~4403

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-21 20:12:21
Total Users: 7
Average Resource Utilization: 49.08%
eMBB Total Rate: 360.20 Mbps, URLLC Total Rate: 14.76 Mbps, mMTC Total Rate: 11.19 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          2  2.0/30 MHz        6.67%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 6.97 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4671, completion~1088, total~5759

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I want to remotely access my work computer"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to remotely access my work computer", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~1273, total~2643
[Token] Slice_Type_Determination: prompt~3133, completion~341, total~3474
LLM recommended URLLC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~3553, completion~594, total~4147
[Token] Beamforming_Bandwidth: prompt~215, completion~833, total~1048
[Token] Allocate_Resources: prompt~4310, completion~931, total~5241

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-21 20:13:40
Total Users: 8
Average Resource Utilization: 64.46%
eMBB Total Rate: 469.31 Mbps, URLLC Total Rate: 14.76 Mbps, mMTC Total Rate: 11.19 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          2  2.0/30 MHz        6.67%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 8 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5551, completion~1331, total~6882

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to use holographic communication"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use holographic communication", "embb"),)
[Token] Intent_Analysis: prompt~1368, completion~1186, total~2554
[Token] Slice_Type_Determination: prompt~3040, completion~361, total~3401
[Token] Beamforming_Bandwidth: prompt~214, completion~657, total~871
[Token] Allocate_Resources: prompt~3528, completion~858, total~4386

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-21 20:14:45
Total Users: 9
Average Resource Utilization: 72.15%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 14.76 Mbps, mMTC Total Rate: 11.19 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          2  2.0/30 MHz        6.67%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 9 → eMBB Slice
CQI: 15, Bandwidth: 10.0 MHz, Rate: 151.35 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4648, completion~958, total~5606

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control precision cnc machines with zero tolerance for delay", "urllc"),)
[Token] Intent_Analysis: prompt~1378, completion~1129, total~2507
[Token] Slice_Type_Determination: prompt~2997, completion~283, total~3280
[Token] Beamforming_Bandwidth: prompt~219, completion~457, total~676
[Token] Allocate_Resources: prompt~3409, completion~818, total~4227

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-21 20:15:42
Total Users: 10
Average Resource Utilization: 72.92%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 29.90 Mbps, mMTC Total Rate: 11.19 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          3  3.0/30 MHz        10.00%
mMTC           2  1.8/10 MHz        18.00%

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
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4494, completion~929, total~5423

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need emergency response coordination during a disaste"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need emergency response coordination during a disaster", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~1009, total~2381
[Token] Slice_Type_Determination: prompt~2867, completion~255, total~3122
[Token] Beamforming_Bandwidth: prompt~216, completion~323, total~539
[Token] Allocate_Resources: prompt~3251, completion~889, total~4140

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-21 20:16:38
Total Users: 11
Average Resource Utilization: 73.69%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 34.66 Mbps, mMTC Total Rate: 11.19 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          4  4.0/30 MHz        13.33%
mMTC           2  1.8/10 MHz        18.00%

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
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4405, completion~1125, total~5530

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I want to use remote surgery equipment"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to use remote surgery equipment", "urllc"),)
[Token] Intent_Analysis: prompt~1368, completion~893, total~2261
[Token] Slice_Type_Determination: prompt~2750, completion~357, total~3107
[Token] Beamforming_Bandwidth: prompt~214, completion~672, total~886
[Token] Allocate_Resources: prompt~3236, completion~857, total~4093

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-21 20:17:37
Total Users: 12
Average Resource Utilization: 74.46%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 42.45 Mbps, mMTC Total Rate: 11.19 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          5  5.0/30 MHz        16.67%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        1   |          7.79 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4350, completion~944, total~5294

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need to participate in an online multiplayer game"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to participate in an online multiplayer game", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~1114, total~2486
[Token] Slice_Type_Determination: prompt~2973, completion~476, total~3449
[Token] Beamforming_Bandwidth: prompt~216, completion~822, total~1038
[Token] Allocate_Resources: prompt~3576, completion~862, total~4438

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-21 20:18:42
Total Users: 13
Average Resource Utilization: 75.23%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 50.24 Mbps, mMTC Total Rate: 11.19 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          6  6.0/30 MHz        20.00%
mMTC           2  1.8/10 MHz        18.00%

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
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4693, completion~844, total~5537

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need to track the location of a shipping container"
CQI: 14
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to track the location of a shipping container", "mmtc"),, ("i need to track the location of a shipping container", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~1098, total~2472
[Token] Slice_Type_Determination: prompt~2938, completion~243, total~3181
[Token] Beamforming_Bandwidth: prompt~216, completion~829, total~1045
[Token] Allocate_Resources: prompt~3336, completion~1049, total~4385

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-21 20:19:46
Total Users: 14
Average Resource Utilization: 75.92%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 50.24 Mbps, mMTC Total Rate: 23.94 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          6  6.0/30 MHz        20.00%
mMTC           3  2.7/10 MHz        27.00%

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
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4641, completion~910, total~5551

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[Token] Intent_Analysis: prompt~1376, completion~1213, total~2589
[Token] Slice_Type_Determination: prompt~3070, completion~265, total~3335
[Token] Beamforming_Bandwidth: prompt~218, completion~466, total~684
[Token] Allocate_Resources: prompt~3528, completion~891, total~4419
[Token] Failure_Evaluation: prompt~4526, completion~647, total~5173

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
[Token] Intent_Analysis: prompt~1372, completion~1189, total~2561
[Token] Slice_Type_Determination: prompt~3043, completion~251, total~3294
[Token] Beamforming_Bandwidth: prompt~215, completion~529, total~744
[Token] Allocate_Resources: prompt~3447, completion~1517, total~4964

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-21 20:21:49
Total Users: 15
Average Resource Utilization: 76.62%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 50.24 Mbps, mMTC Total Rate: 32.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          6  6.0/30 MHz        20.00%
mMTC           4  3.6/10 MHz        36.00%

New User Allocation:
User 16 → mMTC Slice
CQI: 9, Bandwidth: 0.9 MHz, Rate: 8.56 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5219, completion~1174, total~6393

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to monitor soil moisture levels in a large farm"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor soil moisture levels in a large farm", "mmtc"),)
[Token] Intent_Analysis: prompt~1376, completion~1017, total~2393
[Token] Slice_Type_Determination: prompt~2849, completion~342, total~3191
[Token] Beamforming_Bandwidth: prompt~217, completion~366, total~583
[Token] Allocate_Resources: prompt~3349, completion~1322, total~4671

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-21 20:22:54
Total Users: 16
Average Resource Utilization: 77.31%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 50.24 Mbps, mMTC Total Rate: 37.41 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          6  6.0/30 MHz        20.00%
mMTC           5  4.5/10 MHz        45.00%

New User Allocation:
User 17 → mMTC Slice
CQI: 4, Bandwidth: 0.9 MHz, Rate: 4.91 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4918, completion~1283, total~6201

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need instant alerts for life-threatening patient conditions"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need instant alerts for life-threatening patient conditions", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~1001, total~2373
[Token] Slice_Type_Determination: prompt~2859, completion~367, total~3226
[Token] Beamforming_Bandwidth: prompt~216, completion~591, total~807
[Token] Allocate_Resources: prompt~3361, completion~814, total~4175

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-21 20:23:55
Total Users: 17
Average Resource Utilization: 78.08%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 57.21 Mbps, mMTC Total Rate: 37.41 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          7  7.0/30 MHz        23.33%
mMTC           5  4.5/10 MHz        45.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 6.97 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        1   |          6.97 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4441, completion~956, total~5397

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to download large files"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to download large files", "embb"),)
[Token] Intent_Analysis: prompt~1366, completion~1009, total~2375
[Token] Slice_Type_Determination: prompt~2845, completion~200, total~3045
[Token] Beamforming_Bandwidth: prompt~213, completion~697, total~910
[Token] Allocate_Resources: prompt~3238, completion~865, total~4103
[Token] Failure_Evaluation: prompt~4210, completion~731, total~4941

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
[Token] Intent_Analysis: prompt~1372, completion~1075, total~2447
[Token] Slice_Type_Determination: prompt~2921, completion~227, total~3148
[Token] Beamforming_Bandwidth: prompt~216, completion~474, total~690
[Token] Allocate_Resources: prompt~3272, completion~845, total~4117

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-21 20:25:45
Total Users: 18
Average Resource Utilization: 78.85%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 66.72 Mbps, mMTC Total Rate: 37.41 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          8  8.0/30 MHz        26.67%
mMTC           5  4.5/10 MHz        45.00%

New User Allocation:
User 20 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4392, completion~1017, total~5409

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~1064, total~2434
[Token] Slice_Type_Determination: prompt~2927, completion~379, total~3306
[Token] Beamforming_Bandwidth: prompt~215, completion~788, total~1003
[Token] Allocate_Resources: prompt~3437, completion~897, total~4334

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-21 20:26:54
Total Users: 19
Average Resource Utilization: 82.69%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 101.59 Mbps, mMTC Total Rate: 37.41 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          9  13.0/30 MHz       43.33%
mMTC           5  4.5/10 MHz        45.00%

New User Allocation:
User 21 → URLLC Slice
CQI: 6, Bandwidth: 5.0 MHz, Rate: 34.87 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        5   |         34.87 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4593, completion~1012, total~5605

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "Smart city parking sensor reporting availability"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("smart city parking sensor reporting availability", "mmtc"),)
[Token] Intent_Analysis: prompt~1366, completion~922, total~2288
[Token] Slice_Type_Determination: prompt~2741, completion~596, total~3337
[Token] Beamforming_Bandwidth: prompt~212, completion~734, total~946
[Token] Allocate_Resources: prompt~3496, completion~1097, total~4593

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-21 20:27:57
Total Users: 20
Average Resource Utilization: 84.92%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 101.59 Mbps, mMTC Total Rate: 60.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          9  13.0/30 MHz       43.33%
mMTC           6  7.4/10 MHz        74.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 7, Bandwidth: 2.9 MHz, Rate: 22.59 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        2.9 |         22.59 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4833, completion~989, total~5822

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control critical infrastructure with zero downtime", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~1133, total~2505
[Token] Slice_Type_Determination: prompt~3001, completion~300, total~3301
[Token] Beamforming_Bandwidth: prompt~216, completion~446, total~662
[Token] Allocate_Resources: prompt~3432, completion~876, total~4308

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-21 20:28:55
Total Users: 21
Average Resource Utilization: 85.69%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 107.05 Mbps, mMTC Total Rate: 60.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         10  14.0/30 MHz       46.67%
mMTC           6  7.4/10 MHz        74.00%

New User Allocation:
User 23 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        2.9 |         22.59 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4569, completion~794, total~5363

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to monitor IoT sensors in real-time"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to monitor iot sensors in real-time", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~1301, total~2673
[Token] Slice_Type_Determination: prompt~3144, completion~326, total~3470
[Token] Beamforming_Bandwidth: prompt~216, completion~415, total~631
[Token] Allocate_Resources: prompt~3592, completion~918, total~4510

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-21 20:29:52
Total Users: 22
Average Resource Utilization: 86.46%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 112.51 Mbps, mMTC Total Rate: 60.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         11  15.0/30 MHz       50.00%
mMTC           6  7.4/10 MHz        74.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        2.9 |         22.59 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4769, completion~1134, total~5903

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "My smart trash can needs to signal that it's full"
CQI: 5
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart trash can needs to signal that it's full", "mmtc"),)
[Token] Intent_Analysis: prompt~1376, completion~1128, total~2504
[Token] Slice_Type_Determination: prompt~2972, completion~303, total~3275
[Token] Beamforming_Bandwidth: prompt~237, completion~716, total~953
[Token] Allocate_Resources: prompt~3425, completion~1640, total~5065

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-21 20:31:14
Total Users: 23
Average Resource Utilization: 87.15%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 112.51 Mbps, mMTC Total Rate: 65.57 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         11  15.0/30 MHz       50.00%
mMTC           7  8.3/10 MHz        83.00%

New User Allocation:
User 25 → mMTC Slice
CQI: 5, Bandwidth: 0.9 MHz, Rate: 5.57 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        2.9 |         22.59 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5296, completion~841, total~6137

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to monitor my home security cameras remotely", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~1154, total~2526
[Token] Slice_Type_Determination: prompt~3007, completion~804, total~3811
WARNING: LLM didn't provide explicit slice recommendation. Using knowledge base recommendation.
[Token] Beamforming_Bandwidth: prompt~216, completion~594, total~810
[Token] Allocate_Resources: prompt~3943, completion~890, total~4833

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-21 20:32:26
Total Users: 24
Average Resource Utilization: 87.92%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 120.30 Mbps, mMTC Total Rate: 65.57 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         12  16.0/30 MHz       53.33%
mMTC           7  8.3/10 MHz        83.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     7 |        1   |          7.79 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        2.9 |         22.59 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5152, completion~1081, total~6233

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("industrial equipment monitoring sensor data", "mmtc"))
[Token] Intent_Analysis: prompt~1364, completion~1260, total~2624
[Token] Slice_Type_Determination: prompt~3107, completion~409, total~3516
LLM recommended URLLC but knowledge base recommended mMTC, using knowledge base recommendation
[Token] Intent_Override: prompt~3589, completion~690, total~4279
[Token] Beamforming_Bandwidth: prompt~231, completion~642, total~873
[Token] Allocate_Resources: prompt~4439, completion~1303, total~5742

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-21 20:33:55
Total Users: 25
Average Resource Utilization: 88.62%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 120.30 Mbps, mMTC Total Rate: 70.48 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         12  16.0/30 MHz               53.33%
mMTC           8  9.200000000000001/10 MHz  92.00%

New User Allocation:
User 27 → mMTC Slice
CQI: 4, Bandwidth: 0.9 MHz, Rate: 4.91 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        2.9 |         22.59 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     4 |        0.9 |          4.91 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~6056, completion~1063, total~7119

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart parking sensor needs to report if the spot is free", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~916, total~2294
[Token] Slice_Type_Determination: prompt~2753, completion~405, total~3158
[Token] Beamforming_Bandwidth: prompt~223, completion~925, total~1148
[Token] Allocate_Resources: prompt~3373, completion~995, total~4368

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-21 20:35:01
Total Users: 26
Average Resource Utilization: 89.23%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 120.30 Mbps, mMTC Total Rate: 78.26 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         12  16.0/30 MHz       53.33%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 9, Bandwidth: 0.9 MHz, Rate: 8.56 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.10000000000000109 MHz
  User 22: 2.9 → 2.799999999999999 MHz, Rate: 22.59 → 21.81 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        2.8 |         21.81 |            500 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |        0.9 |          5.57 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     9 |        0.9 |          8.56 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |        0.9 |          6.28 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4711, completion~862, total~5573

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I want to stream music while browsing social media"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to stream music while browsing social media", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~1388, total~2760
[Token] Slice_Type_Determination: prompt~3242, completion~390, total~3632
[Token] Beamforming_Bandwidth: prompt~216, completion~548, total~764
[Token] Allocate_Resources: prompt~3828, completion~1174, total~5002
[Token] Failure_Evaluation: prompt~5124, completion~879, total~6003

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
[Token] Intent_Analysis: prompt~1364, completion~1603, total~2967
[Token] Slice_Type_Determination: prompt~3470, completion~527, total~3997
LLM recommended URLLC but knowledge base recommended mMTC, using knowledge base recommendation
[Token] Intent_Override: prompt~4074, completion~681, total~4755
[Token] Beamforming_Bandwidth: prompt~211, completion~661, total~872
[Token] Allocate_Resources: prompt~4965, completion~1204, total~6169

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-21 20:37:59
Total Users: 27
Average Resource Utilization: 89.23%
eMBB Total Rate: 620.66 Mbps, URLLC Total Rate: 120.30 Mbps, mMTC Total Rate: 81.45 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC         12  16.0/30 MHz       53.33%
mMTC          10  10.0/10 MHz       100.00%

New User Allocation:
User 30 → mMTC Slice
CQI: 11, Bandwidth: 0.9 MHz, Rate: 10.20 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.9 MHz
  User 22: 2.799999999999999 → 1.899999999999999 MHz, Rate: 21.81 → 14.80 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |        1   |          4.76 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     7 |        1   |          7.79 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |        1   |          7.79 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |       20   |        109.11 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |       10   |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |        0.9 |         12.75 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |        0.9 |          8.56 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |        0.9 |          4.91 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |        1.9 |         14.8  |            500 | ADJUSTED |
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
[Token] Network_Evaluation: prompt~6555, completion~1070, total~7625

Detailed Slice Utilization Values:
eMBB utils: [22.22, 22.22, 22.22, 44.44, 66.67, 66.67, 66.67, 88.89, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 0.0, 3.33, 3.33, 3.33, 3.33, 6.67, 6.67, 6.67, 10.0, 13.33, 16.67, 20.0, 20.0, 20.0, 20.0, 23.33, 26.67, 43.33, 43.33, 46.67, 50.0, 50.0, 53.33, 53.33, 53.33, 53.33]
mMTC utils: [0.0, 9.0, 9.0, 9.0, 9.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 27.0, 36.0, 45.0, 45.0, 45.0, 45.0, 74.0, 74.0, 74.0, 83.0, 83.0, 92.0, 100.0, 100.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             80 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | mMTC    | mMTC           | Yes            |     4 |        0.9 |          4.91 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | URLLC   | URLLC          | Yes            |     7 |        1   |          7.79 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | eMBB    | eMBB           | Yes            |     3 |       20   |         95.29 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | mMTC    | mMTC           | Yes            |     6 |        0.9 |          6.28 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |     6 |        1   |          6.97 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | eMBB    | eMBB           | Yes            |    15 |       10   |        151.35 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | URLLC   | URLLC          | Yes            |    15 |        1   |         15.14 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |     3 |        1   |          4.76 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     7 |        1   |          7.79 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | URLLC          | Yes            |     7 |        1   |          7.79 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | mMTC    | mMTC           | Yes            |    14 |        0.9 |         12.75 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Failed   | eMBB    | eMBB           | Yes            |    15 |       20   |        302.7  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | mMTC    | mMTC           | Yes            |     9 |        0.9 |          8.56 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | mMTC    | mMTC           | Yes            |     4 |        0.9 |          4.91 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     6 |        1   |          6.97 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Failed   | eMBB    | eMBB           | Yes            |     8 |       20   |        172.78 |             80 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | URLLC   | URLLC          | Yes            |     9 |        1   |          9.51 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | URLLC   | URLLC          | Yes            |     6 |        5   |         34.87 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | mMTC    | mMTC           | Yes            |     7 |        2.9 |         22.59 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          5.46 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          5.46 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | mMTC    | mMTC           | Yes            |     5 |        0.9 |          5.57 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | eMBB           | No             |     7 |        1   |          7.79 |              5 | No         |
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
Average eMBB utilization: 85.19%
Average URLLC utilization: 24.07%
Average mMTC utilization: 40.85%

Weighted Average Utilization: 67.67%

Transmission Rate Statistics:
Final eMBB total rate: 620.66 Mbps
Final URLLC total rate: 120.30 Mbps
Final mMTC total rate: 81.45 Mbps

Resource Utilization:
Average resource utilization: 89.23%

Results exported to F:\code\wirelessagent\run_results\batch_run\fkb\glm-4.7\network_slicing_results_TJU_west_glm-4.7.csv

✓ TJU_west 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\fkb\glm-4.7\network_slicing_results_TJU_west_glm-4.7.csv