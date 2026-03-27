场景 3/5: TJU_east
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_east.csv
输出: F:\code\wirelessagent\run_results\batch_run\fkb\kimi-k2.5\network_slicing_results_TJU_east_kimi-k2.5.csv
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
[Token] Intent_Analysis: prompt~1366, completion~804, total~2170
[Token] Slice_Type_Determination: prompt~2642, completion~72, total~2714
[Token] Beamforming_Bandwidth: prompt~211, completion~2, total~213
[Token] Allocate_Resources: prompt~2831, completion~306, total~3137

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-16 12:02:54
Total Users: 1
Average Resource Utilization: 7.69%
eMBB Total Rate: 151.35 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  10.0/90 MHz       11.11%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → eMBB Slice
CQI: 15, Bandwidth: 10.0 MHz, Rate: 151.35 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         10 |        151.35 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3365, completion~678, total~4043

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need to stream 8K video content"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to stream 8k video content", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~781, total~2153
[Token] Slice_Type_Determination: prompt~2623, completion~106, total~2729
[Token] Beamforming_Bandwidth: prompt~216, completion~311, total~527
[Token] Allocate_Resources: prompt~2847, completion~396, total~3243

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-16 12:04:02
Total Users: 2
Average Resource Utilization: 23.08%
eMBB Total Rate: 260.46 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  30.0/90 MHz       33.33%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         10 |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3472, completion~682, total~4154

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I want to listen to low-quality audio streaming"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to listen to low-quality audio streaming", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~812, total~2184
[Token] Slice_Type_Determination: prompt~2653, completion~106, total~2759
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2909, completion~549, total~3458

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-16 12:05:15
Total Users: 3
Average Resource Utilization: 38.46%
eMBB Total Rate: 563.16 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  50.0/90 MHz       55.56%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 3 → eMBB Slice
CQI: 15, Bandwidth: 20.0 MHz, Rate: 302.70 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         10 |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3707, completion~822, total~4529

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I want to browse websites and check email"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~777, total~2147
[Token] Slice_Type_Determination: prompt~2608, completion~116, total~2724
[Token] Beamforming_Bandwidth: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~2871, completion~586, total~3457

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-16 12:06:11
Total Users: 4
Average Resource Utilization: 53.85%
eMBB Total Rate: 753.46 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  70.0/90 MHz       77.78%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 4 → eMBB Slice
CQI: 9, Bandwidth: 20.0 MHz, Rate: 190.30 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         10 |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        190.3  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3692, completion~760, total~4452

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to monitor my home security cameras remotely", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~640, total~2012
[Token] Slice_Type_Determination: prompt~2474, completion~67, total~2541
[Token] Beamforming_Bandwidth: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~2658, completion~340, total~2998

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-16 12:07:13
Total Users: 5
Average Resource Utilization: 56.92%
eMBB Total Rate: 753.46 Mbps, URLLC Total Rate: 45.33 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  70.0/90 MHz       77.78%
URLLC          1  4.0/30 MHz        13.33%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 5 → URLLC Slice
CQI: 11, Bandwidth: 4.0 MHz, Rate: 45.33 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |          4 |         45.33 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3231, completion~517, total~3748

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I need to participate in a video conference meeting"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to participate in a video conference meeting", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~768, total~2140
[Token] Slice_Type_Determination: prompt~2615, completion~115, total~2730
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2875, completion~575, total~3450

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-16 12:08:13
Total Users: 6
Average Resource Utilization: 72.31%
eMBB Total Rate: 998.77 Mbps, URLLC Total Rate: 45.33 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          1  4.0/30 MHz        13.33%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 6 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 245.31 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |          4 |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        151.35 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |        245.31 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3680, completion~792, total~4472

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to monitor water level in a reservoir"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor water level in a reservoir", "mmtc"),, ("i need to monitor water level in a reservoir periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~759, total~2131
[Token] Slice_Type_Determination: prompt~2585, completion~63, total~2648
WARNING: LLM didn't provide explicit slice recommendation. Using knowledge base recommendation.
[Token] Beamforming_Bandwidth: prompt~213, completion~2, total~215
[Token] Allocate_Resources: prompt~2803, completion~503, total~3306

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-16 12:09:07
Total Users: 7
Average Resource Utilization: 73.77%
eMBB Total Rate: 998.77 Mbps, URLLC Total Rate: 45.33 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          1  4.0/30 MHz        13.33%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 7 → mMTC Slice
CQI: 6, Bandwidth: 1.9 MHz, Rate: 13.25 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
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
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3618, completion~753, total~4371

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~729, total~2099
[Token] Slice_Type_Determination: prompt~2557, completion~89, total~2646
[Token] Beamforming_Bandwidth: prompt~215, completion~257, total~472
[Token] Allocate_Resources: prompt~2761, completion~337, total~3098

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-16 12:09:47
Total Users: 8
Average Resource Utilization: 77.62%
eMBB Total Rate: 998.77 Mbps, URLLC Total Rate: 92.90 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          2  9.0/30 MHz        30.00%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 9, Bandwidth: 5.0 MHz, Rate: 47.57 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 | NEW      |
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
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3330, completion~650, total~3980

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to update my social media status"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~738, total~2108
[Token] Slice_Type_Determination: prompt~2565, completion~74, total~2639
[Token] Beamforming_Bandwidth: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~2860, completion~326, total~3186

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-16 12:10:30
Total Users: 9
Average Resource Utilization: 77.62%
eMBB Total Rate: 961.47 Mbps, URLLC Total Rate: 92.90 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          2  9.0/30 MHz        30.00%
mMTC           1  1.9/10 MHz        19.00%

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
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3540, completion~702, total~4242

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I want to update my social media status"
CQI: 13
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~611, total~1981
[Token] Slice_Type_Determination: prompt~2441, completion~81, total~2522
[Token] Beamforming_Bandwidth: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~2747, completion~308, total~3055

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-16 12:16:17
Total Users: 10
Average Resource Utilization: 77.62%
eMBB Total Rate: 1005.16 Mbps, URLLC Total Rate: 92.90 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC          2  9.0/30 MHz        30.00%
mMTC           1  1.9/10 MHz        19.00%

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
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3413, completion~726, total~4139

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need vehicle-to-vehicle collision avoidance systems"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need vehicle-to-vehicle collision avoidance systems", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~934, total~2306
[Token] Slice_Type_Determination: prompt~2768, completion~92, total~2860
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2977, completion~371, total~3348

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-16 12:17:14
Total Users: 11
Average Resource Utilization: 80.69%
eMBB Total Rate: 1005.16 Mbps, URLLC Total Rate: 149.58 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC          3  13.0/30 MHz       43.33%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 14, Bandwidth: 4.0 MHz, Rate: 56.68 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3592, completion~616, total~4208

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need to sync my calendar and contacts"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to sync my calendar and contacts", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~635, total~2005
[Token] Slice_Type_Determination: prompt~2461, completion~110, total~2571
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~2633, completion~106, total~2739
[Token] Beamforming_Bandwidth: prompt~215, completion~4, total~219
[Token] Allocate_Resources: prompt~3027, completion~363, total~3390

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-16 12:18:04
Total Users: 12
Average Resource Utilization: 80.69%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 149.58 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          3  13.0/30 MHz       43.33%
mMTC           1  1.9/10 MHz        19.00%

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
|        11 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3882, completion~726, total~4608

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~651, total~2023
[Token] Slice_Type_Determination: prompt~2485, completion~108, total~2593
[Token] Beamforming_Bandwidth: prompt~216, completion~305, total~521
[Token] Allocate_Resources: prompt~2711, completion~401, total~3112

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-16 12:19:00
Total Users: 13
Average Resource Utilization: 84.54%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 176.86 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          4  18.0/30 MHz       60.00%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 4, Bandwidth: 5.0 MHz, Rate: 27.28 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        5   |         27.28 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3353, completion~760, total~4113

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I want to use holographic communication"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use holographic communication", "embb"),)
[Token] Intent_Analysis: prompt~1368, completion~699, total~2067
[Token] Slice_Type_Determination: prompt~2535, completion~95, total~2630
[Token] Beamforming_Bandwidth: prompt~214, completion~4, total~218
[Token] Allocate_Resources: prompt~2823, completion~470, total~3293
[Token] Failure_Evaluation: prompt~3387, completion~362, total~3749

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
[Token] Intent_Analysis: prompt~1370, completion~658, total~2028
[Token] Slice_Type_Determination: prompt~2489, completion~96, total~2585
[Token] Beamforming_Bandwidth: prompt~215, completion~4, total~219
[Token] Allocate_Resources: prompt~2776, completion~551, total~3327
[Token] Failure_Evaluation: prompt~3420, completion~410, total~3830

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
[Token] Intent_Analysis: prompt~1368, completion~718, total~2086
[Token] Slice_Type_Determination: prompt~2553, completion~116, total~2669
[Token] Beamforming_Bandwidth: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~2787, completion~356, total~3143

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-16 12:21:13
Total Users: 14
Average Resource Utilization: 87.62%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 211.42 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          5  22.0/30 MHz       73.33%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 8, Bandwidth: 4.0 MHz, Rate: 34.56 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        4   |         34.56 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3384, completion~563, total~3947

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to participate in a video conference meeting"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to participate in a video conference meeting", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~789, total~2161
[Token] Slice_Type_Determination: prompt~2633, completion~91, total~2724
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2916, completion~679, total~3595
[Token] Failure_Evaluation: prompt~3688, completion~667, total~4355

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
[Token] Intent_Analysis: prompt~1378, completion~643, total~2021
[Token] Slice_Type_Determination: prompt~2484, completion~90, total~2574
[Token] Beamforming_Bandwidth: prompt~219, completion~343, total~562
[Token] Allocate_Resources: prompt~2690, completion~323, total~3013

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-16 12:23:24
Total Users: 15
Average Resource Utilization: 88.38%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 216.88 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          6  23.0/30 MHz       76.67%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        4   |         34.56 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3251, completion~564, total~3815

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[Token] Intent_Analysis: prompt~1376, completion~559, total~1935
[Token] Slice_Type_Determination: prompt~2395, completion~84, total~2479
[Token] Beamforming_Bandwidth: prompt~218, completion~332, total~550
[Token] Allocate_Resources: prompt~2671, completion~443, total~3114
[Token] Failure_Evaluation: prompt~3203, completion~465, total~3668

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
[Token] Intent_Analysis: prompt~1372, completion~473, total~1845
[Token] Slice_Type_Determination: prompt~2304, completion~83, total~2387
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2502, completion~356, total~2858

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-16 12:25:01
Total Users: 16
Average Resource Utilization: 90.69%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 245.42 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          7  26.0/30 MHz       86.67%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 20 → URLLC Slice
CQI: 9, Bandwidth: 3.0 MHz, Rate: 28.54 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        4   |         34.56 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        3   |         28.54 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3094, completion~559, total~3653

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart agriculture sensor needs to report soil temperature", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~636, total~2008
[Token] Slice_Type_Determination: prompt~2461, completion~78, total~2539
[Token] Beamforming_Bandwidth: prompt~215, completion~4, total~219
[Token] Allocate_Resources: prompt~2692, completion~543, total~3235

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-16 12:25:59
Total Users: 17
Average Resource Utilization: 92.0%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 245.42 Mbps, mMTC Total Rate: 26.49 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           8  90.0/90 MHz                100.00%
URLLC          7  26.0/30 MHz                86.67%
mMTC           2  3.5999999999999996/10 MHz  36.00%

New User Allocation:
User 21 → mMTC Slice
CQI: 7, Bandwidth: 1.7 MHz, Rate: 13.24 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        4   |         34.56 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        3   |         28.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|        21 | mMTC    |     7 |        1.7 |         13.24 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3471, completion~776, total~4247

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "I need to synchronize multiple robots on a factory floor"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to synchronize multiple robots on a factory floor", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~880, total~2254
[Token] Slice_Type_Determination: prompt~2717, completion~136, total~2853
[Token] Beamforming_Bandwidth: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~2969, completion~310, total~3279

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-16 12:26:56
Total Users: 18
Average Resource Utilization: 93.54%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 264.45 Mbps, mMTC Total Rate: 26.49 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           8  90.0/90 MHz                100.00%
URLLC          8  28.0/30 MHz                93.33%
mMTC           2  3.5999999999999996/10 MHz  36.00%

New User Allocation:
User 22 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 19.03 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        4   |         34.56 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        3   |         28.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|        21 | mMTC    |     7 |        1.7 |         13.24 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3527, completion~904, total~4431

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to control a robotic arm in real time"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control a robotic arm in real time", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~707, total~2081
[Token] Slice_Type_Determination: prompt~2549, completion~91, total~2640
[Token] Beamforming_Bandwidth: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~2760, completion~333, total~3093

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-16 12:28:04
Total Users: 19
Average Resource Utilization: 94.31%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 269.91 Mbps, mMTC Total Rate: 26.49 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           8  90.0/90 MHz                100.00%
URLLC          9  29.0/30 MHz                96.67%
mMTC           2  3.5999999999999996/10 MHz  36.00%

New User Allocation:
User 23 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        4   |         34.56 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        3   |         28.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|        21 | mMTC    |     7 |        1.7 |         13.24 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3334, completion~913, total~4247

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to detect and isolate power grid faults instantly", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~845, total~2219
[Token] Slice_Type_Determination: prompt~2683, completion~129, total~2812
[Token] Beamforming_Bandwidth: prompt~217, completion~4, total~221
[Token] Allocate_Resources: prompt~2930, completion~426, total~3356

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-16 12:29:11
Total Users: 20
Average Resource Utilization: 95.08%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 276.10 Mbps, mMTC Total Rate: 26.49 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           8  90.0/90 MHz                100.00%
URLLC         10  30.0/30 MHz                100.00%
mMTC           2  3.5999999999999996/10 MHz  36.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 5, Bandwidth: 1.0 MHz, Rate: 6.19 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        4   |         34.56 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        3   |         28.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|        21 | mMTC    |     7 |        1.7 |         13.24 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3601, completion~708, total~4309

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 2
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~895, total~2269
[Token] Slice_Type_Determination: prompt~2716, completion~68, total~2784
[Token] Beamforming_Bandwidth: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2936, completion~565, total~3501

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-16 12:30:15
Total Users: 21
Average Resource Utilization: 95.77%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 276.10 Mbps, mMTC Total Rate: 30.20 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         10  30.0/30 MHz       100.00%
mMTC           3  4.5/10 MHz        45.00%

New User Allocation:
User 25 → mMTC Slice
CQI: 2, Bandwidth: 0.9 MHz, Rate: 3.71 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        4   |         34.56 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        3   |         28.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|        21 | mMTC    |     7 |        1.7 |         13.24 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        0.9 |          3.71 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3740, completion~734, total~4474

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 1
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a fleet of delivery drones needs to send low-rate telemetry data", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~858, total~2236
[Token] Slice_Type_Determination: prompt~2696, completion~79, total~2775
[Token] Beamforming_Bandwidth: prompt~218, completion~2, total~220
[Token] Allocate_Resources: prompt~2929, completion~525, total~3454

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-16 12:31:19
Total Users: 22
Average Resource Utilization: 97.23%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 276.10 Mbps, mMTC Total Rate: 36.92 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         10  30.0/30 MHz       100.00%
mMTC           4  6.4/10 MHz        64.00%

New User Allocation:
User 26 → mMTC Slice
CQI: 1, Bandwidth: 1.9 MHz, Rate: 6.72 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        4   |         34.56 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        3   |         28.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|        21 | mMTC    |     7 |        1.7 |         13.24 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        1.9 |          6.72 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3686, completion~604, total~4290

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "My smart meter needs to report its reading"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart meter needs to report its reading", "mmtc"),)
[Token] Intent_Analysis: prompt~1370, completion~846, total~2216
[Token] Slice_Type_Determination: prompt~2661, completion~62, total~2723
[Token] Beamforming_Bandwidth: prompt~219, completion~2, total~221
[Token] Allocate_Resources: prompt~2875, completion~336, total~3211

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-16 12:32:04
Total Users: 23
Average Resource Utilization: 97.92%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 276.10 Mbps, mMTC Total Rate: 43.93 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           8  90.0/90 MHz               100.00%
URLLC         10  30.0/30 MHz               100.00%
mMTC           5  7.300000000000001/10 MHz  73.00%

New User Allocation:
User 27 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        4   |         34.56 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        3   |         28.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|        21 | mMTC    |     7 |        1.7 |         13.24 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        1.9 |          6.72 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3452, completion~599, total~4051

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("industrial equipment monitoring sensor data", "mmtc"))
[Token] Intent_Analysis: prompt~1364, completion~708, total~2072
[Token] Slice_Type_Determination: prompt~2522, completion~67, total~2589
[Token] Beamforming_Bandwidth: prompt~231, completion~2, total~233
[Token] Allocate_Resources: prompt~2742, completion~532, total~3274

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-16 12:32:50
Total Users: 24
Average Resource Utilization: 98.62%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 276.10 Mbps, mMTC Total Rate: 48.22 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           8  90.0/90 MHz               100.00%
URLLC         10  30.0/30 MHz               100.00%
mMTC           6  8.200000000000001/10 MHz  82.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 3, Bandwidth: 0.9 MHz, Rate: 4.29 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        4   |         34.56 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        3   |         28.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|        21 | mMTC    |     7 |        1.7 |         13.24 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        1.9 |          6.72 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     3 |        0.9 |          4.29 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3512, completion~548, total~4060

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart agriculture sensor needs to report soil temperature", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~650, total~2022
[Token] Slice_Type_Determination: prompt~2469, completion~74, total~2543
[Token] Beamforming_Bandwidth: prompt~231, completion~2, total~233
[Token] Allocate_Resources: prompt~2695, completion~654, total~3349

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-16 12:33:42
Total Users: 25
Average Resource Utilization: 99.31%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 276.10 Mbps, mMTC Total Rate: 56.78 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           8  90.0/90 MHz               100.00%
URLLC         10  30.0/30 MHz               100.00%
mMTC           7  9.100000000000001/10 MHz  91.00%

New User Allocation:
User 29 → mMTC Slice
CQI: 9, Bandwidth: 0.9 MHz, Rate: 8.56 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        4   |         34.56 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        3   |         28.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|        21 | mMTC    |     7 |        1.7 |         13.24 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        1.9 |          6.72 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | mMTC    |     9 |        0.9 |          8.56 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3594, completion~679, total~4273

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I want to use remote surgery equipment"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to use remote surgery equipment", "urllc"),)
[Token] Intent_Analysis: prompt~1368, completion~820, total~2188
[Token] Slice_Type_Determination: prompt~2652, completion~91, total~2743
[Token] Beamforming_Bandwidth: prompt~214, completion~4, total~218
[Token] Allocate_Resources: prompt~2908, completion~398, total~3306

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-16 12:34:38
Total Users: 26
Average Resource Utilization: 99.31%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 270.57 Mbps, mMTC Total Rate: 56.78 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           8  90.0/90 MHz               100.00%
URLLC         11  30.0/30 MHz               100.00%
mMTC           7  9.100000000000001/10 MHz  91.00%

New User Allocation:
User 30 → URLLC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 8.64 Mbps, Latency: 1.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 11: 4.0 → 3.0 MHz, Rate: 56.68 → 42.51 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        3   |         42.51 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        4   |         34.56 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        3   |         28.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | URLLC   |     8 |        1   |          8.64 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        5   |         47.57 |              5 |          |
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
|        21 | mMTC    |     7 |        1.7 |         13.24 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        1.9 |          6.72 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3623, completion~868, total~4491

Detailed Slice Utilization Values:
eMBB utils: [11.11, 33.33, 55.56, 77.78, 77.78, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 0.0, 0.0, 0.0, 13.33, 13.33, 13.33, 30.0, 30.0, 30.0, 43.33, 43.33, 60.0, 73.33, 76.67, 86.67, 86.67, 93.33, 96.67, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
mMTC utils: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 36.0, 36.0, 36.0, 36.0, 45.0, 64.0, 73.0, 82.0, 91.0, 91.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |    15 |       10   |        151.35 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | eMBB    | eMBB           | Yes            |    15 |       20   |        302.7  |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | eMBB    | eMBB           | Yes            |     9 |       20   |        190.3  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | URLLC   | eMBB           | No             |    11 |        4   |         45.33 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | mMTC    | mMTC           | Yes            |     6 |        1.9 |         13.25 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | URLLC          | Yes            |     9 |        5   |         47.57 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | eMBB    | eMBB           | Yes            |    13 |       20   |        264.25 |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |    14 |        4   |         56.68 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | eMBB    | eMBB           | Yes            |     5 |       20   |        123.87 |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | URLLC          | Yes            |     4 |        5   |         27.28 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Failed   | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Failed   | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |     8 |        4   |         34.56 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Failed   | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          5.46 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Failed   | eMBB    | eMBB           | Yes            |    11 |       20   |        226.64 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | URLLC   | URLLC          | Yes            |     9 |        3   |         28.54 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | mMTC    | mMTC           | Yes            |     7 |        1.7 |         13.24 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | URLLC   | URLLC          | Yes            |     9 |        2   |         19.03 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          5.46 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     5 |        1   |          6.19 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | mMTC    | mMTC           | Yes            |     2 |        0.9 |          3.71 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | mMTC    | mMTC           | Yes            |     1 |        1.9 |          6.72 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | mMTC    | mMTC           | Yes            |     3 |        0.9 |          4.29 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | mMTC    | mMTC           | Yes            |     9 |        0.9 |          8.56 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Success  | URLLC   | URLLC          | Yes            |     8 |        1   |          8.64 |              1 | Yes        |
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
Average eMBB utilization: 90.60%
Average URLLC utilization: 57.31%
Average mMTC utilization: 30.00%

Weighted Average Utilization: 78.25%

Transmission Rate Statistics:
Final eMBB total rate: 870.56 Mbps
Final URLLC total rate: 270.57 Mbps
Final mMTC total rate: 56.78 Mbps

Resource Utilization:
Average resource utilization: 99.31%

Results exported to F:\code\wirelessagent\run_results\batch_run\fkb\kimi-k2.5\network_slicing_results_TJU_east_kimi-k2.5.csv

✓ TJU_east 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\fkb\kimi-k2.5\network_slicing_results_TJU_east_kimi-k2.5.csv