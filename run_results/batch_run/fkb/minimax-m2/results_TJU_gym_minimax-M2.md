F:\code\wirelessagent\.venv\Scripts\python.exe F:\code\wirelessagent\with_knowledge_base\WA_DS_V3_FKB.py 
Starting network slice management system with CSV-based user testing...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~1095, total~2469
[Token] Slice_Type_Determination: prompt~2933, completion~222, total~3155
[Token] Beamforming_Bandwidth: prompt~214, completion~1137, total~1351
[Token] Allocate_Resources: prompt~3311, completion~825, total~4136

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-15 16:49:47
Total Users: 1
Average Resource Utilization: 2.23%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 13.82 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          0  0/30 MHz          0%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 1 → mMTC Slice
CQI: 3, Bandwidth: 2.9 MHz, Rate: 13.82 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4379, completion~733, total~5112

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I want to browse websites and check email"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~1038, total~2408
[Token] Slice_Type_Determination: prompt~2869, completion~308, total~3177
[Token] Beamforming_Bandwidth: prompt~213, completion~320, total~533
[Token] Allocate_Resources: prompt~3296, completion~786, total~4082

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-15 16:50:47
Total Users: 2
Average Resource Utilization: 10.69%
eMBB Total Rate: 124.65 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 13.82 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  11.0/90 MHz       12.22%
URLLC          0  0/30 MHz          0%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 2 → eMBB Slice
CQI: 11, Bandwidth: 11.0 MHz, Rate: 124.65 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4338, completion~745, total~5083

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to monitor water level in a reservoir"
CQI: 13
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor water level in a reservoir", "mmtc"),, ("i need to monitor water level in a reservoir periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~1156, total~2528
[Token] Slice_Type_Determination: prompt~3004, completion~243, total~3247
[Token] Beamforming_Bandwidth: prompt~215, completion~394, total~609
[Token] Allocate_Resources: prompt~3404, completion~814, total~4218

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-15 16:52:11
Total Users: 3
Average Resource Utilization: 12.92%
eMBB Total Rate: 124.65 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 52.14 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  11.0/90 MHz       12.22%
URLLC          0  0/30 MHz          0%
mMTC           2  5.8/10 MHz        58.00%

New User Allocation:
User 3 → mMTC Slice
CQI: 13, Bandwidth: 2.9 MHz, Rate: 38.32 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2.9 |         38.32 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4455, completion~888, total~5343

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my wearable device needs to upload health data periodically", "mmtc"),, ("my wearable device needs to upload health data periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~1121, total~2493
[Token] Slice_Type_Determination: prompt~2962, completion~309, total~3271
[Token] Beamforming_Bandwidth: prompt~215, completion~438, total~653
[Token] Allocate_Resources: prompt~3428, completion~1057, total~4485

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-15 16:53:32
Total Users: 4
Average Resource Utilization: 13.62%
eMBB Total Rate: 124.65 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 57.05 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  11.0/90 MHz       12.22%
URLLC          0  0/30 MHz          0%
mMTC           3  6.7/10 MHz        67.00%

New User Allocation:
User 4 → mMTC Slice
CQI: 4, Bandwidth: 0.9 MHz, Rate: 4.91 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2.9 |         38.32 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4733, completion~858, total~5591

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[Token] Intent_Analysis: prompt~1376, completion~904, total~2280
[Token] Slice_Type_Determination: prompt~2745, completion~390, total~3135
[Token] Beamforming_Bandwidth: prompt~218, completion~416, total~634
[Token] Allocate_Resources: prompt~3281, completion~830, total~4111

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-15 16:54:49
Total Users: 5
Average Resource Utilization: 29.0%
eMBB Total Rate: 297.43 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 57.05 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  31.0/90 MHz       34.44%
URLLC          0  0/30 MHz          0%
mMTC           3  6.7/10 MHz        67.00%

New User Allocation:
User 5 → eMBB Slice
CQI: 8, Bandwidth: 20.0 MHz, Rate: 172.78 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2.9 |         38.32 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4353, completion~961, total~5314

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I need to stream 8K video content"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to stream 8k video content", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~766, total~2138
[Token] Slice_Type_Determination: prompt~2601, completion~207, total~2808
[Token] Beamforming_Bandwidth: prompt~216, completion~460, total~676
[Token] Allocate_Resources: prompt~2956, completion~806, total~3762

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-15 16:56:08
Total Users: 6
Average Resource Utilization: 44.38%
eMBB Total Rate: 421.30 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 57.05 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  51.0/90 MHz       56.67%
URLLC          0  0/30 MHz          0%
mMTC           3  6.7/10 MHz        67.00%

New User Allocation:
User 6 → eMBB Slice
CQI: 5, Bandwidth: 20.0 MHz, Rate: 123.87 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2.9 |         38.32 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4010, completion~957, total~4967

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to track the location of a shipping container"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to track the location of a shipping container", "mmtc"),, ("i need to track the location of a shipping container", "mmtc"),)
[Token] Intent_Analysis: prompt~1374, completion~901, total~2275
[Token] Slice_Type_Determination: prompt~2730, completion~240, total~2970
[Token] Beamforming_Bandwidth: prompt~216, completion~507, total~723
[Token] Allocate_Resources: prompt~3124, completion~906, total~4030

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-15 16:57:31
Total Users: 7
Average Resource Utilization: 46.62%
eMBB Total Rate: 421.30 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 77.27 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  51.0/90 MHz       56.67%
URLLC          0  0/30 MHz          0%
mMTC           4  9.6/10 MHz        96.00%

New User Allocation:
User 7 → mMTC Slice
CQI: 6, Bandwidth: 2.9 MHz, Rate: 20.22 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2.9 |         38.32 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4259, completion~710, total~4969

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my wearable device needs to upload health data periodically", "mmtc"),, ("my wearable device needs to upload health data periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~1242, total~2614
[Token] Slice_Type_Determination: prompt~3080, completion~388, total~3468
[Token] Workload_Balance: prompt~3565, completion~398, total~3963
[Token] Beamforming_Bandwidth: prompt~214, completion~489, total~703
[Token] Allocate_Resources: prompt~4092, completion~839, total~4931

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-15 16:59:22
Total Users: 8
Average Resource Utilization: 50.46%
eMBB Total Rate: 421.30 Mbps, URLLC Total Rate: 75.68 Mbps, mMTC Total Rate: 77.27 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  51.0/90 MHz       56.67%
URLLC          1  5.0/30 MHz        16.67%
mMTC           4  9.6/10 MHz        96.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 15, Bandwidth: 5.0 MHz, Rate: 75.68 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2.9 |         38.32 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5280, completion~814, total~6094

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I need reliable connectivity for implanted medical devices"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable connectivity for implanted medical devices", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~1085, total~2455
[Token] Slice_Type_Determination: prompt~2927, completion~274, total~3201
[Token] Beamforming_Bandwidth: prompt~215, completion~1574, total~1789
[Token] Allocate_Resources: prompt~3328, completion~794, total~4122

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-15 17:01:24
Total Users: 9
Average Resource Utilization: 51.23%
eMBB Total Rate: 421.30 Mbps, URLLC Total Rate: 83.47 Mbps, mMTC Total Rate: 77.27 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  51.0/90 MHz       56.67%
URLLC          2  6.0/30 MHz        20.00%
mMTC           4  9.6/10 MHz        96.00%

New User Allocation:
User 9 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2.9 |         38.32 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4381, completion~904, total~5285

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to check weather forecasts"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to check weather forecasts", "embb"),)
[Token] Intent_Analysis: prompt~1366, completion~1143, total~2509
[Token] Slice_Type_Determination: prompt~2986, completion~329, total~3315
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~3384, completion~438, total~3822
[Token] Beamforming_Bandwidth: prompt~213, completion~417, total~630
[Token] Allocate_Resources: prompt~3974, completion~839, total~4813

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-15 17:03:08
Total Users: 10
Average Resource Utilization: 66.62%
eMBB Total Rate: 545.17 Mbps, URLLC Total Rate: 83.47 Mbps, mMTC Total Rate: 77.27 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  71.0/90 MHz       78.89%
URLLC          2  6.0/30 MHz        20.00%
mMTC           4  9.6/10 MHz        96.00%

New User Allocation:
User 10 → eMBB Slice
CQI: 5, Bandwidth: 20.0 MHz, Rate: 123.87 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2.9 |         38.32 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5120, completion~907, total~6027

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of city-wide smart streetlights", "mmtc"),, ("i need to check the status of city-wide smart streetlights", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~946, total~2324
[Token] Slice_Type_Determination: prompt~2783, completion~288, total~3071
[Token] Beamforming_Bandwidth: prompt~223, completion~2628, total~2851
[Token] Allocate_Resources: prompt~3285, completion~931, total~4216

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-15 17:05:51
Total Users: 11
Average Resource Utilization: 66.92%
eMBB Total Rate: 545.17 Mbps, URLLC Total Rate: 83.47 Mbps, mMTC Total Rate: 76.94 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  71.0/90 MHz       78.89%
URLLC          2  6.0/30 MHz        20.00%
mMTC           5  10.0/10 MHz       100.00%

New User Allocation:
User 11 → mMTC Slice
CQI: 6, Bandwidth: 0.9 MHz, Rate: 6.28 Mbps, Latency: 1000.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.49999999999999967 MHz
  User 3: 2.9 → 2.4000000000000004 MHz, Rate: 38.32 → 31.71 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2.4 |         31.71 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4555, completion~951, total~5506

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need microsecond-level latency for high-frequency tradin"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need microsecond-level latency for high-frequency trading", "urllc"),)
[Token] Intent_Analysis: prompt~1376, completion~871, total~2247
[Token] Slice_Type_Determination: prompt~2722, completion~375, total~3097
[Token] Beamforming_Bandwidth: prompt~218, completion~387, total~605
[Token] Allocate_Resources: prompt~3228, completion~765, total~3993

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-15 17:07:07
Total Users: 12
Average Resource Utilization: 67.69%
eMBB Total Rate: 545.17 Mbps, URLLC Total Rate: 88.23 Mbps, mMTC Total Rate: 76.94 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  71.0/90 MHz       78.89%
URLLC          3  7.0/30 MHz        23.33%
mMTC           5  10.0/10 MHz       100.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 4.76 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2.4 |         31.71 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4260, completion~810, total~5070

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I want to use holographic communication"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use holographic communication", "embb"),)
[Token] Intent_Analysis: prompt~1368, completion~712, total~2080
[Token] Slice_Type_Determination: prompt~2546, completion~412, total~2958
[Token] Beamforming_Bandwidth: prompt~214, completion~275, total~489
[Token] Allocate_Resources: prompt~3164, completion~740, total~3904

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-15 17:08:06
Total Users: 13
Average Resource Utilization: 82.31%
eMBB Total Rate: 726.83 Mbps, URLLC Total Rate: 88.23 Mbps, mMTC Total Rate: 76.94 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          3  7.0/30 MHz        23.33%
mMTC           5  10.0/10 MHz       100.00%

New User Allocation:
User 13 → eMBB Slice
CQI: 9, Bandwidth: 20.0 MHz, Rate: 190.30 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 5: 20.0 → 19.0 MHz, Rate: 172.78 → 164.14 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       19   |        164.14 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2.4 |         31.71 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4228, completion~679, total~4907

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 12
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control critical infrastructure with zero downtime", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~900, total~2272
[Token] Slice_Type_Determination: prompt~2759, completion~296, total~3055
[Token] Beamforming_Bandwidth: prompt~216, completion~824, total~1040
[Token] Allocate_Resources: prompt~3183, completion~793, total~3976

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-15 17:09:36
Total Users: 14
Average Resource Utilization: 86.15%
eMBB Total Rate: 726.83 Mbps, URLLC Total Rate: 149.56 Mbps, mMTC Total Rate: 76.94 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          4  12.0/30 MHz       40.00%
mMTC           5  10.0/10 MHz       100.00%

New User Allocation:
User 14 → URLLC Slice
CQI: 12, Bandwidth: 5.0 MHz, Rate: 61.33 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2.4 |         31.71 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4241, completion~1021, total~5262

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need real-time fraud detection for financial transactions"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need real-time fraud detection for financial transactions", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~600, total~1972
[Token] Slice_Type_Determination: prompt~2438, completion~310, total~2748
[Token] Beamforming_Bandwidth: prompt~216, completion~416, total~632
[Token] Allocate_Resources: prompt~2875, completion~810, total~3685

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-15 17:10:50
Total Users: 15
Average Resource Utilization: 90.0%
eMBB Total Rate: 726.83 Mbps, URLLC Total Rate: 188.51 Mbps, mMTC Total Rate: 76.94 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          5  17.0/30 MHz       56.67%
mMTC           5  10.0/10 MHz       100.00%

New User Allocation:
User 15 → URLLC Slice
CQI: 7, Bandwidth: 5.0 MHz, Rate: 38.95 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        5   |         38.95 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2.4 |         31.71 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3946, completion~679, total~4625

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need instant facial recognition for public security threats"
CQI: 10
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need instant facial recognition for public security threats", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~993, total~2365
[Token] Slice_Type_Determination: prompt~2839, completion~334, total~3173
[Token] Beamforming_Bandwidth: prompt~216, completion~738, total~954
[Token] Allocate_Resources: prompt~3302, completion~914, total~4216

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-15 17:12:23
Total Users: 16
Average Resource Utilization: 90.77%
eMBB Total Rate: 726.83 Mbps, URLLC Total Rate: 198.92 Mbps, mMTC Total Rate: 76.94 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          6  18.0/30 MHz       60.00%
mMTC           5  10.0/10 MHz       100.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 10, Bandwidth: 1.0 MHz, Rate: 10.41 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2.4 |         31.71 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4473, completion~646, total~5119

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "My smart trash can needs to signal that it's full"
CQI: 13
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart trash can needs to signal that it's full", "mmtc"),)
[Token] Intent_Analysis: prompt~1376, completion~890, total~2266
[Token] Slice_Type_Determination: prompt~2719, completion~274, total~2993
[Token] Beamforming_Bandwidth: prompt~217, completion~407, total~624
[Token] Allocate_Resources: prompt~3207, completion~827, total~4034

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-15 17:13:57
Total Users: 17
Average Resource Utilization: 90.77%
eMBB Total Rate: 726.83 Mbps, URLLC Total Rate: 198.92 Mbps, mMTC Total Rate: 76.94 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          6  18.0/30 MHz       60.00%
mMTC           6  10.0/10 MHz       100.00%

New User Allocation:
User 17 → mMTC Slice
CQI: 13, Bandwidth: 0.9 MHz, Rate: 11.89 Mbps, Latency: 1000.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.9 MHz
  User 3: 2.4000000000000004 → 1.5000000000000004 MHz, Rate: 31.71 → 19.82 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.5 |         19.82 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4358, completion~816, total~5174

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~984, total~2354
[Token] Slice_Type_Determination: prompt~2841, completion~284, total~3125
[Token] Beamforming_Bandwidth: prompt~215, completion~753, total~968
[Token] Allocate_Resources: prompt~3252, completion~813, total~4065

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-15 17:15:48
Total Users: 18
Average Resource Utilization: 93.08%
eMBB Total Rate: 726.83 Mbps, URLLC Total Rate: 213.21 Mbps, mMTC Total Rate: 76.94 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          7  21.0/30 MHz       70.00%
mMTC           6  10.0/10 MHz       100.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 3, Bandwidth: 3.0 MHz, Rate: 14.29 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.5 |         19.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4344, completion~824, total~5168

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to monitor water level in a reservoir"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor water level in a reservoir", "mmtc"),, ("i need to monitor water level in a reservoir periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~802, total~2174
[Token] Slice_Type_Determination: prompt~2642, completion~284, total~2926
[Token] Beamforming_Bandwidth: prompt~215, completion~287, total~502
[Token] Allocate_Resources: prompt~3129, completion~1202, total~4331

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-15 17:17:00
Total Users: 19
Average Resource Utilization: 93.08%
eMBB Total Rate: 726.83 Mbps, URLLC Total Rate: 213.21 Mbps, mMTC Total Rate: 74.96 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          7  21.0/30 MHz       70.00%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 19 → mMTC Slice
CQI: 3, Bandwidth: 0.9 MHz, Rate: 4.29 Mbps, Latency: 1000.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.9 MHz
  User 7: 2.9 → 2.0 MHz, Rate: 20.22 → 13.95 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       11   |        124.65 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       20   |        123.87 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.5 |         19.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2   |         13.95 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4662, completion~744, total~5406

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I need to make a high-quality voice call"
CQI: 10
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to make a high-quality voice call", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~988, total~2360
[Token] Slice_Type_Determination: prompt~2827, completion~554, total~3381
[Token] Beamforming_Bandwidth: prompt~216, completion~546, total~762
[Token] Allocate_Resources: prompt~3685, completion~787, total~4472

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-15 17:19:44
Total Users: 20
Average Resource Utilization: 93.08%
eMBB Total Rate: 753.95 Mbps, URLLC Total Rate: 213.21 Mbps, mMTC Total Rate: 74.96 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          7  21.0/30 MHz       70.00%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 10, Bandwidth: 20.0 MHz, Rate: 208.28 Mbps, Latency: 30.0 ms

Dynamic Resource Adjustments:
Users adjusted: 4, Bandwidth freed: 20.0 MHz
  User 13: 20.0 → 11.0 MHz, Rate: 190.30 → 104.66 Mbps, User 5: 19.0 → 12.0 MHz, Rate: 164.14 → 103.67 Mbps, User 2: 11.0 → 9.0 MHz, Rate: 124.65 → 101.99 Mbps, User 6: 20.0 → 18.0 MHz, Rate: 123.87 → 111.48 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       18   |        111.48 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1.5 |         19.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        2   |         13.95 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4919, completion~698, total~5617

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to use holographic communication"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use holographic communication", "embb"),)
[Token] Intent_Analysis: prompt~1368, completion~739, total~2107
[Token] Slice_Type_Determination: prompt~2579, completion~208, total~2787
[Token] Beamforming_Bandwidth: prompt~214, completion~329, total~543
[Token] Allocate_Resources: prompt~2987, completion~803, total~3790
[Token] Failure_Evaluation: prompt~3905, completion~486, total~4391

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
[Token] Intent_Analysis: prompt~1376, completion~970, total~2346
[Token] Slice_Type_Determination: prompt~2797, completion~268, total~3065
[Token] Beamforming_Bandwidth: prompt~217, completion~415, total~632
[Token] Allocate_Resources: prompt~3308, completion~949, total~4257

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-15 17:22:25
Total Users: 21
Average Resource Utilization: 93.08%
eMBB Total Rate: 753.95 Mbps, URLLC Total Rate: 213.21 Mbps, mMTC Total Rate: 75.76 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          7  21.0/30 MHz       70.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 11, Bandwidth: 0.9 MHz, Rate: 10.20 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 0.9 MHz
  User 3: 1.5000000000000004 → 1.0 MHz, Rate: 19.82 → 13.21 Mbps, User 7: 2.0 → 1.6000000000000005 MHz, Rate: 13.95 → 11.16 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       18   |        111.48 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |        0.9 |         10.2  |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1   |         13.21 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.6 |         11.16 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4635, completion~966, total~5601

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 1
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to detect and isolate power grid faults instantly", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~993, total~2367
[Token] Slice_Type_Determination: prompt~2859, completion~183, total~3042
 [Token] Beamforming_Bandwidth: prompt~217, completion~1349, total~1566
[Token] Allocate_Resources: prompt~3167, completion~760, total~3927

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-15 17:23:56
Total Users: 22
Average Resource Utilization: 93.85%
eMBB Total Rate: 753.95 Mbps, URLLC Total Rate: 216.75 Mbps, mMTC Total Rate: 75.76 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          8  22.0/30 MHz       73.33%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 23 → URLLC Slice
CQI: 1, Bandwidth: 1.0 MHz, Rate: 3.54 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        1   |          3.54 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       18   |        111.48 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |        0.9 |         10.2  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1   |         13.21 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.6 |         11.16 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4199, completion~768, total~4967

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I want to use remote surgery equipment"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to use remote surgery equipment", "urllc"),)
[Token] Intent_Analysis: prompt~1368, completion~887, total~2255
[Token] Slice_Type_Determination: prompt~2736, completion~278, total~3014
[Token] Beamforming_Bandwidth: prompt~214, completion~434, total~648
[Token] Allocate_Resources: prompt~3141, completion~812, total~3953

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-15 17:25:10
Total Users: 23
Average Resource Utilization: 97.69%
eMBB Total Rate: 753.95 Mbps, URLLC Total Rate: 247.72 Mbps, mMTC Total Rate: 75.76 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          9  27.0/30 MHz       90.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 5, Bandwidth: 5.0 MHz, Rate: 30.97 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        1   |          3.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         30.97 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       18   |        111.48 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |        0.9 |         10.2  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1   |         13.21 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.6 |         11.16 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4222, completion~871, total~5093

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I want to stream music while browsing social media"
CQI: 2
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to stream music while browsing social media", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~838, total~2210
[Token] Slice_Type_Determination: prompt~2673, completion~332, total~3005
[Token] Beamforming_Bandwidth: prompt~216, completion~516, total~732
[Token] Allocate_Resources: prompt~3197, completion~860, total~4057
[Token] Failure_Evaluation: prompt~4175, completion~557, total~4732

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
[Token] Intent_Analysis: prompt~1380, completion~913, total~2293
[Token] Slice_Type_Determination: prompt~2761, completion~418, total~3179
[Token] Beamforming_Bandwidth: prompt~220, completion~394, total~614
[Token] Allocate_Resources: prompt~3312, completion~844, total~4156

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-15 17:28:12
Total Users: 24
Average Resource Utilization: 100.0%
eMBB Total Rate: 753.95 Mbps, URLLC Total Rate: 268.64 Mbps, mMTC Total Rate: 75.76 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         10  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 6, Bandwidth: 3.0 MHz, Rate: 20.92 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        1   |          3.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         30.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        3   |         20.92 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       18   |        111.48 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |        0.9 |         10.2  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1   |         13.21 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.6 |         11.16 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4415, completion~647, total~5062

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need to transmit real-time patient vital signs during critical care"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to transmit real-time patient vital signs during critical care", "urllc"),)
[Token] Intent_Analysis: prompt~1378, completion~891, total~2269
[Token] Slice_Type_Determination: prompt~2745, completion~422, total~3167
[Token] Beamforming_Bandwidth: prompt~219, completion~335, total~554
[Token] Allocate_Resources: prompt~3341, completion~765, total~4106

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-15 17:29:25
Total Users: 25
Average Resource Utilization: 100.0%
eMBB Total Rate: 753.95 Mbps, URLLC Total Rate: 258.26 Mbps, mMTC Total Rate: 75.76 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         11  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 27 → URLLC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 4.76 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 8: 5.0 → 4.0 MHz, Rate: 75.68 → 60.54 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        1   |          3.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         30.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        1   |          4.76 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        4   |         60.54 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       18   |        111.48 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |        0.9 |         10.2  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1   |         13.21 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.6 |         11.16 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4437, completion~886, total~5323

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~583, total~1953
[Token] Slice_Type_Determination: prompt~2416, completion~379, total~2795
[Token] Beamforming_Bandwidth: prompt~215, completion~336, total~551
[Token] Allocate_Resources: prompt~2974, completion~771, total~3745

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-15 17:30:48
Total Users: 26
Average Resource Utilization: 100.0%
eMBB Total Rate: 753.95 Mbps, URLLC Total Rate: 251.45 Mbps, mMTC Total Rate: 75.76 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         12  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 28 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 14: 5.0 → 4.0 MHz, Rate: 61.33 → 49.06 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        4   |         49.06 |              3 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        1   |          3.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         30.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | URLLC   |     4 |        1   |          5.46 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       18   |        111.48 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |        0.9 |         10.2  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1   |         13.21 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.6 |         11.16 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4080, completion~968, total~5048

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need to participate in an online multiplayer game"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to participate in an online multiplayer game", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~936, total~2308
[Token] Slice_Type_Determination: prompt~2781, completion~599, total~3380
[Token] Beamforming_Bandwidth: prompt~216, completion~396, total~612
[Token] Allocate_Resources: prompt~3553, completion~793, total~4346

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-15 17:32:27
Total Users: 27
Average Resource Utilization: 100.0%
eMBB Total Rate: 753.95 Mbps, URLLC Total Rate: 251.46 Mbps, mMTC Total Rate: 75.76 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         13  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 29 → URLLC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 15.14 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 8: 4.0 → 3.0 MHz, Rate: 60.54 → 45.41 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        4   |         49.06 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        1   |         10.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     1 |        1   |          3.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         30.97 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | URLLC   |    15 |        1   |         15.14 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |    15 |        3   |         45.41 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     7 |        1   |          7.79 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     5 |       20   |        123.87 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |        9   |        101.99 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    10 |       20   |        208.28 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |     5 |       18   |        111.48 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2.9 |         13.82 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        0.9 |         11.89 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    11 |        0.9 |         10.2  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        1   |         13.21 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        0.9 |          4.91 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.6 |         11.16 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4679, completion~549, total~5228

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I want to browse websites and check email"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~899, total~2269
[Token] Slice_Type_Determination: prompt~2726, completion~257, total~2983
[Token] Beamforming_Bandwidth: prompt~215, completion~409, total~624
[Token] Allocate_Resources: prompt~3177, completion~807, total~3984
[Token] Failure_Evaluation: prompt~4081, completion~701, total~4782

----------------------------------------
ALLOCATION FAILED FOR USER 30
----------------------------------------
Request: I want to browse websites and check email
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

Detailed Slice Utilization Values:
eMBB utils: [0.0, 12.22, 12.22, 12.22, 34.44, 56.67, 56.67, 56.67, 56.67, 78.89, 78.89, 78.89, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16.67, 20.0, 20.0, 20.0, 23.33, 23.33, 40.0, 56.67, 60.0, 60.0, 70.0, 70.0, 70.0, 70.0, 73.33, 90.0, 100.0, 100.0, 100.0, 100.0]
mMTC utils: [29.0, 29.0, 58.0, 67.0, 67.0, 67.0, 96.0, 96.0, 96.0, 96.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | mMTC    | mMTC           | Yes            |     3 |        2.9 |         13.82 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | eMBB    | eMBB           | Yes            |    11 |       11   |        124.65 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | mMTC    | mMTC           | Yes            |    13 |        2.9 |         38.32 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | mMTC    | mMTC           | Yes            |     4 |        0.9 |          4.91 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | eMBB    | eMBB           | Yes            |     8 |       20   |        172.78 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | eMBB    | eMBB           | Yes            |     5 |       20   |        123.87 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | mMTC    | mMTC           | Yes            |     6 |        2.9 |         20.22 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | mMTC           | No             |    15 |        5   |         75.68 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | URLLC   | URLLC          | Yes            |     7 |        1   |          7.79 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | eMBB    | eMBB           | Yes            |     5 |       20   |        123.87 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | mMTC    | mMTC           | Yes            |     6 |        0.9 |          6.28 |           1000 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     3 |        1   |          4.76 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | eMBB    | eMBB           | Yes            |     9 |       20   |        190.3  |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | URLLC   | URLLC          | Yes            |    12 |        5   |         61.33 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | URLLC   | URLLC          | Yes            |     7 |        5   |         38.95 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |    10 |        1   |         10.41 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | mMTC    | mMTC           | Yes            |    13 |        0.9 |         11.89 |           1000 | Yes        |
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
|        23 | Success  | URLLC   | URLLC          | Yes            |     1 |        1   |          3.54 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     5 |        5   |         30.97 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Failed   | eMBB    | eMBB           | Yes            |     2 |       20   |         82.49 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | URLLC          | Yes            |     6 |        3   |         20.92 |              5 | No         |
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
Correctly identified intents: 29/30
Intent understanding rate: 96.7%

Workload Balancing Statistics:
Users with workload balancing: 0/30
Workload balancing rate: 0.0%

Slice Utilization Statistics:
Average eMBB utilization: 75.35%
Average URLLC utilization: 43.83%
Average mMTC utilization: 88.93%

Weighted Average Utilization: 69.12%

Transmission Rate Statistics:
Final eMBB total rate: 753.95 Mbps
Final URLLC total rate: 251.46 Mbps
Final mMTC total rate: 75.76 Mbps

Resource Utilization:
Average resource utilization: 100.00%

Results exported to F:\code\wirelessagent\run_results\with_fkb\network_slicing_results_TJU_gym_minimax-M2.csv

进程已结束，退出代码为 0
