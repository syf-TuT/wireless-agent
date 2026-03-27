============================================================
场景 2/5: TJU_south
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_south.csv
输出: F:\code\wirelessagent\run_results\batch_run\kb\glm-5\network_slicing_results_TJU_south_glm-5.csv
------------------------------------------------------------
Starting network slice management system with CSV-based user testing...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 2
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)
[RAG] Using knowledge base: F:\code\wirelessagent\with_knowledge_base\Intent_Understand.txt
[RAG] Initializing RAG system...
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 10296.33it/s]
BertModel LOAD REPORT from: F:\code\wirelessagent\models\all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
[RAG] Loaded existing vector index from F:\code\wirelessagent\with_knowledge_base\Intent_Understand_index
[RAG] BM25 retriever initialized with 122 documents
[RAG] RAG system initialized successfully!
[RAG] RAG system initialized successfully!
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1399, total~2771
[Token] Slice_Type_Determination: prompt~2462, completion~277, total~2739
[Token] Bandwidth_Analysis: prompt~214, completion~1019, total~1233
[Token] Allocate_Resources: prompt~2662, completion~992, total~3654

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-17 12:45:50
Total Users: 1
Average Resource Utilization: 3.85%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 20.62 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          1  5.0/30 MHz        16.67%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → URLLC Slice
CQI: 2, Bandwidth: 5.0 MHz, Rate: 20.62 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          5 |         20.62 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3310, completion~820, total~4130

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need my autonomous vehicle to communicate in real time"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need my autonomous vehicle to communicate in real time", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~1298, total~2672
[Token] Slice_Type_Determination: prompt~2483, completion~216, total~2699
[Token] Bandwidth_Analysis: prompt~217, completion~1317, total~1534
[Token] Allocate_Resources: prompt~2686, completion~1057, total~3743

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-17 12:47:50
Total Users: 2
Average Resource Utilization: 7.69%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 44.44 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          2  10.0/30 MHz       33.33%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → URLLC Slice
CQI: 3, Bandwidth: 5.0 MHz, Rate: 23.82 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          5 |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |          5 |         23.82 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3386, completion~967, total~4353

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart agriculture sensor needs to report soil temperature", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1544, total~2916
[Token] Slice_Type_Determination: prompt~2304, completion~130, total~2434
[Token] Bandwidth_Analysis: prompt~213, completion~873, total~1086
[Token] Allocate_Resources: prompt~2539, completion~1617, total~4156

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-17 12:50:00
Total Users: 3
Average Resource Utilization: 8.38%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 44.44 Mbps, mMTC Total Rate: 13.62 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          2  10.0/30 MHz       33.33%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 3 → mMTC Slice
CQI: 15, Bandwidth: 0.9 MHz, Rate: 13.62 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3253, completion~924, total~4177

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to participate in an online multiplayer game"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to participate in an online multiplayer game", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~2295, total~3667
[Token] Slice_Type_Determination: prompt~2289, completion~210, total~2499
LLM recommended eMBB but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~2457, completion~1004, total~3461
[Token] Bandwidth_Analysis: prompt~216, completion~3510, total~3726
[Token] Allocate_Resources: prompt~2962, completion~760, total~3722

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-17 12:53:36
Total Users: 4
Average Resource Utilization: 10.69%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 89.85 Mbps, mMTC Total Rate: 13.62 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          3  13.0/30 MHz       43.33%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 4 → URLLC Slice
CQI: 15, Bandwidth: 3.0 MHz, Rate: 45.41 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3603, completion~817, total~4420

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a fleet of delivery drones needs to send low-rate telemetry data", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~1849, total~3227
[Token] Slice_Type_Determination: prompt~2391, completion~190, total~2581
[Token] Bandwidth_Analysis: prompt~218, completion~1195, total~1413
[Token] Allocate_Resources: prompt~2629, completion~1626, total~4255

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-17 12:56:08
Total Users: 5
Average Resource Utilization: 12.92%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 89.85 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          3  13.0/30 MHz       43.33%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 5 → mMTC Slice
CQI: 3, Bandwidth: 2.9 MHz, Rate: 13.82 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3304, completion~1187, total~4491

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I want to play competitive mobile games with ultra-low latency"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to play competitive mobile games with ultra-low latency", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~991, total~2367
[Token] Slice_Type_Determination: prompt~2361, completion~298, total~2659
[Token] Bandwidth_Analysis: prompt~218, completion~1635, total~1853
[Token] Allocate_Resources: prompt~2570, completion~1020, total~3590

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-17 12:58:38
Total Users: 6
Average Resource Utilization: 16.77%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 117.13 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          4  18.0/30 MHz       60.00%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 6 → URLLC Slice
CQI: 4, Bandwidth: 5.0 MHz, Rate: 27.28 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3303, completion~943, total~4246

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to participate in an online multiplayer game"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to participate in an online multiplayer game", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~2049, total~3421
[Token] Slice_Type_Determination: prompt~2558, completion~548, total~3106
LLM recommended eMBB but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~2731, completion~822, total~3553
[Token] Bandwidth_Analysis: prompt~216, completion~2239, total~2455
[Token] Allocate_Resources: prompt~3280, completion~861, total~4141

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-17 13:02:08
Total Users: 7
Average Resource Utilization: 18.31%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 145.47 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          5  20.0/30 MHz       66.67%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 14, Bandwidth: 2.0 MHz, Rate: 28.34 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3972, completion~1564, total~5536

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I want to browse websites and check email"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~814, total~2184
[Token] Slice_Type_Determination: prompt~2299, completion~216, total~2515
WARNING: LLM didn't provide explicit slice recommendation. Using knowledge base recommendation.
[Token] Bandwidth_Analysis: prompt~213, completion~5576, total~5789
[Token] Allocate_Resources: prompt~2542, completion~1344, total~3886

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-17 13:06:42
Total Users: 8
Average Resource Utilization: 33.69%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 145.47 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          5  20.0/30 MHz       66.67%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 8 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3356, completion~1251, total~4607

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to monitor my home security cameras remotely", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~995, total~2367
[Token] Slice_Type_Determination: prompt~2329, completion~286, total~2615
LLM recommended eMBB but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~2517, completion~980, total~3497
[Token] Bandwidth_Analysis: prompt~216, completion~1887, total~2103
[Token] Allocate_Resources: prompt~3091, completion~962, total~4053

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-17 13:09:45
Total Users: 9
Average Resource Utilization: 37.54%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 180.34 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          6  25.0/30 MHz       83.33%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 9 → URLLC Slice
CQI: 6, Bandwidth: 5.0 MHz, Rate: 34.87 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3802, completion~888, total~4690

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to make a high-quality voice call"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to make a high-quality voice call", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~2668, total~4040
[Token] Slice_Type_Determination: prompt~2501, completion~1827, total~4328
LLM recommended URLLC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~2663, completion~971, total~3634
[Token] Bandwidth_Analysis: prompt~216, completion~1215, total~1431
[Token] Allocate_Resources: prompt~3208, completion~1263, total~4471

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-17 13:13:44
Total Users: 10
Average Resource Utilization: 52.92%
eMBB Total Rate: 218.22 Mbps, URLLC Total Rate: 180.34 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          6  25.0/30 MHz       83.33%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 10 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 30.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3961, completion~1180, total~5141

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to send text messages and use messaging apps"
CQI: 3
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to send text messages and use messaging apps", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~3266, total~4640
[Token] Slice_Type_Determination: prompt~2248, completion~340, total~2588
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~2406, completion~969, total~3375
[Token] Bandwidth_Analysis: prompt~217, completion~716, total~933
[Token] Allocate_Resources: prompt~3002, completion~1098, total~4100

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-17 13:17:10
Total Users: 11
Average Resource Utilization: 68.31%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 180.34 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          6  25.0/30 MHz       83.33%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 11 → eMBB Slice
CQI: 3, Bandwidth: 20.0 MHz, Rate: 95.29 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3800, completion~971, total~4771

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I want to play competitive mobile games with ultra-low latency"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to play competitive mobile games with ultra-low latency", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~1591, total~2967
[Token] Slice_Type_Determination: prompt~2628, completion~425, total~3053
[Token] Bandwidth_Analysis: prompt~218, completion~3212, total~3430
[Token] Allocate_Resources: prompt~2842, completion~1110, total~3952

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-17 13:20:45
Total Users: 12
Average Resource Utilization: 72.15%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 204.16 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          7  30.0/30 MHz       100.00%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 3, Bandwidth: 5.0 MHz, Rate: 23.82 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        5   |         23.82 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3572, completion~828, total~4400

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need real-time traffic updates for navigation"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need real-time traffic updates for navigation", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1841, total~3211
[Token] Slice_Type_Determination: prompt~2484, completion~515, total~2999
LLM recommended eMBB but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~2664, completion~648, total~3312
[Token] Bandwidth_Analysis: prompt~215, completion~407, total~622
[Token] Allocate_Resources: prompt~3136, completion~1171, total~4307

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-17 13:23:08
Total Users: 13
Average Resource Utilization: 72.15%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 198.53 Mbps, mMTC Total Rate: 27.44 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          8  30.0/30 MHz       100.00%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 4: 3.0 → 2.0 MHz, Rate: 45.41 → 30.27 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3926, completion~726, total~4652

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my wearable device needs to upload health data periodically", "mmtc"),, ("my wearable device needs to upload health data periodically", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1756, total~3128
[Token] Slice_Type_Determination: prompt~2580, completion~186, total~2766
[Token] Bandwidth_Analysis: prompt~215, completion~846, total~1061
[Token] Allocate_Resources: prompt~2807, completion~1198, total~4005

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-17 13:25:16
Total Users: 14
Average Resource Utilization: 72.85%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 198.53 Mbps, mMTC Total Rate: 33.72 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          8  30.0/30 MHz       100.00%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 14 → mMTC Slice
CQI: 6, Bandwidth: 0.9 MHz, Rate: 6.28 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3540, completion~1032, total~4572

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 13
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~1166, total~2542
[Token] Slice_Type_Determination: prompt~2441, completion~222, total~2663
[Token] Bandwidth_Analysis: prompt~218, completion~782, total~1000
[Token] Allocate_Resources: prompt~2657, completion~1132, total~3789

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-17 13:27:29
Total Users: 15
Average Resource Utilization: 88.23%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 198.53 Mbps, mMTC Total Rate: 33.72 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          8  30.0/30 MHz       100.00%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 13, Bandwidth: 20.0 MHz, Rate: 264.25 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3418, completion~1318, total~4736

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need instant facial recognition for public security threats"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need instant facial recognition for public security threats", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1200, total~2572
[Token] Slice_Type_Determination: prompt~2430, completion~186, total~2616
WARNING: LLM didn't provide explicit slice recommendation. Using knowledge base recommendation.
[Token] Bandwidth_Analysis: prompt~216, completion~623, total~839
[Token] Allocate_Resources: prompt~2678, completion~1322, total~4000

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-17 13:29:44
Total Users: 16
Average Resource Utilization: 88.23%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 201.06 Mbps, mMTC Total Rate: 33.72 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          9  30.0/30 MHz       100.00%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 9: 5.0 → 4.0 MHz, Rate: 34.87 → 27.89 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        4   |         27.89 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3526, completion~949, total~4475

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to monitor and control critical manufacturing processes in real-time"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to monitor and control critical manufacturing processes in real-time", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~1191, total~2569
[Token] Slice_Type_Determination: prompt~2331, completion~400, total~2731
[Token] Bandwidth_Analysis: prompt~219, completion~3426, total~3645
[Token] Allocate_Resources: prompt~2806, completion~880, total~3686

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-17 13:33:19
Total Users: 17
Average Resource Utilization: 88.23%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 201.07 Mbps, mMTC Total Rate: 33.72 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC         10  30.0/30 MHz       100.00%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 17 → URLLC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 15.14 Mbps, Latency: 3.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 4: 2.0 → 1.0 MHz, Rate: 30.27 → 15.14 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        4   |         27.89 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3495, completion~976, total~4471

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need to check the status of my smart home sensors"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of my smart home sensors", "mmtc"),, ("i need to check the status of my smart home sensors", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~1953, total~3329
[Token] Slice_Type_Determination: prompt~2348, completion~316, total~2664
[Token] Bandwidth_Analysis: prompt~217, completion~2833, total~3050
[Token] Allocate_Resources: prompt~2590, completion~1301, total~3891

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-17 13:37:01
Total Users: 18
Average Resource Utilization: 88.92%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 201.07 Mbps, mMTC Total Rate: 38.01 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           4  80.0/90 MHz                88.89%
URLLC         10  30.0/30 MHz                100.00%
mMTC           4  5.6000000000000005/10 MHz  56.00%

New User Allocation:
User 18 → mMTC Slice
CQI: 3, Bandwidth: 0.9 MHz, Rate: 4.29 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        4   |         27.89 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3333, completion~716, total~4049

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 2
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("industrial equipment monitoring sensor data", "mmtc"))
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1364, completion~1787, total~3151
[Token] Slice_Type_Determination: prompt~2464, completion~317, total~2781
[Token] Bandwidth_Analysis: prompt~216, completion~1531, total~1747
[Token] Allocate_Resources: prompt~2738, completion~2057, total~4795

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-17 13:40:24
Total Users: 19
Average Resource Utilization: 89.62%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 201.07 Mbps, mMTC Total Rate: 41.72 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           4  80.0/90 MHz               88.89%
URLLC         10  30.0/30 MHz               100.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 19 → mMTC Slice
CQI: 2, Bandwidth: 0.9 MHz, Rate: 3.71 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        4   |         27.89 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3454, completion~1061, total~4515

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I want to download a big game file"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to download a big game file", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1414, total~2784
[Token] Slice_Type_Determination: prompt~2418, completion~307, total~2725
[Token] Bandwidth_Analysis: prompt~215, completion~639, total~854
[Token] Allocate_Resources: prompt~2649, completion~1024, total~3673

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-17 13:42:38
Total Users: 20
Average Resource Utilization: 97.31%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 201.07 Mbps, mMTC Total Rate: 41.72 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         10  30.0/30 MHz               100.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 11, Bandwidth: 10.0 MHz, Rate: 113.32 Mbps, Latency: 80.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        4   |         27.89 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             80 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3305, completion~1110, total~4415

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to update my social media status"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~2329, total~3699
[Token] Slice_Type_Determination: prompt~2542, completion~892, total~3434
[Token] Bandwidth_Analysis: prompt~215, completion~596, total~811
[Token] Allocate_Resources: prompt~3110, completion~1637, total~4747
[Token] Failure_Evaluation: prompt~3849, completion~972, total~4821

----------------------------------------
ALLOCATION FAILED FOR USER 21
----------------------------------------
Request: I want to update my social media status
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 2
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control critical infrastructure with zero downtime", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1094, total~2466
[Token] Slice_Type_Determination: prompt~2365, completion~183, total~2548
[Token] Bandwidth_Analysis: prompt~216, completion~467, total~683
[Token] Allocate_Resources: prompt~2604, completion~890, total~3494

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-17 13:48:05
Total Users: 21
Average Resource Utilization: 97.31%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 191.02 Mbps, mMTC Total Rate: 41.72 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         11  30.0/30 MHz               100.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 22 → URLLC Slice
CQI: 2, Bandwidth: 1.0 MHz, Rate: 4.12 Mbps, Latency: 3.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 7: 2.0 → 1.0 MHz, Rate: 28.34 → 14.17 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        1   |          4.12 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        4   |         27.89 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3343, completion~794, total~4137

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I want to stream a webinar with interactive features"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to stream a webinar with interactive features", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1333, total~2705
[Token] Slice_Type_Determination: prompt~2374, completion~287, total~2661
WARNING: LLM didn't provide explicit slice recommendation. Using knowledge base recommendation.
[Token] Bandwidth_Analysis: prompt~216, completion~518, total~734
[Token] Allocate_Resources: prompt~2666, completion~1576, total~4242
[Token] Failure_Evaluation: prompt~3353, completion~900, total~4253

----------------------------------------
ALLOCATION FAILED FOR USER 23
----------------------------------------
Request: I want to stream a webinar with interactive features
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~1308, total~2684
[Token] Slice_Type_Determination: prompt~2355, completion~438, total~2793
[Token] Bandwidth_Analysis: prompt~218, completion~755, total~973
[Token] Allocate_Resources: prompt~2626, completion~2268, total~4894
[Token] Failure_Evaluation: prompt~3239, completion~670, total~3909

----------------------------------------
ALLOCATION FAILED FOR USER 24
----------------------------------------
Request: I need to use cloud-based AI services for image processing
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I need to synchronize distributed financial ledgers instantly"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to synchronize distributed financial ledgers instantly", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~963, total~2335
[Token] Slice_Type_Determination: prompt~2353, completion~252, total~2605
[Token] Bandwidth_Analysis: prompt~216, completion~492, total~708
[Token] Allocate_Resources: prompt~2632, completion~1043, total~3675

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-17 13:55:31
Total Users: 22
Average Resource Utilization: 97.31%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 198.22 Mbps, mMTC Total Rate: 41.72 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         12  30.0/30 MHz               100.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 25 → URLLC Slice
CQI: 14, Bandwidth: 1.0 MHz, Rate: 14.17 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 9: 4.0 → 3.0 MHz, Rate: 27.89 → 20.92 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        5   |         27.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        3   |         20.92 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3421, completion~978, total~4399

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to control a robotic arm in real time"
CQI: 2
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control a robotic arm in real time", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~1303, total~2677
[Token] Slice_Type_Determination: prompt~2489, completion~260, total~2749
[Token] Bandwidth_Analysis: prompt~217, completion~1424, total~1641
[Token] Allocate_Resources: prompt~2743, completion~1126, total~3869

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-17 13:57:56
Total Users: 23
Average Resource Utilization: 97.31%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 196.88 Mbps, mMTC Total Rate: 41.72 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         13  30.0/30 MHz               100.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 2, Bandwidth: 1.0 MHz, Rate: 4.12 Mbps, Latency: 3.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 6: 5.0 → 4.0 MHz, Rate: 27.28 → 21.82 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     2 |        1   |          4.12 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3621, completion~1050, total~4671

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need emergency response coordination during a disaste"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need emergency response coordination during a disaster", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~795, total~2167
[Token] Slice_Type_Determination: prompt~2259, completion~316, total~2575
[Token] Bandwidth_Analysis: prompt~216, completion~879, total~1095
[Token] Allocate_Resources: prompt~2505, completion~1008, total~3513

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-17 14:00:05
Total Users: 24
Average Resource Utilization: 97.31%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 206.29 Mbps, mMTC Total Rate: 41.72 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         14  30.0/30 MHz               100.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 27 → URLLC Slice
CQI: 14, Bandwidth: 1.0 MHz, Rate: 14.17 Mbps, Latency: 1.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 2: 5.0 → 4.0 MHz, Rate: 23.82 → 19.06 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        5   |         23.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        4   |         19.06 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |    14 |        1   |         14.17 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3280, completion~698, total~3978

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I want to use augmented reality navigation"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use augmented reality navigation", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1368, completion~2074, total~3442
[Token] Slice_Type_Determination: prompt~2650, completion~582, total~3232
[Token] Bandwidth_Analysis: prompt~214, completion~458, total~672
[Token] Allocate_Resources: prompt~3110, completion~1403, total~4513
[Token] Failure_Evaluation: prompt~3845, completion~581, total~4426

----------------------------------------
ALLOCATION FAILED FOR USER 28
----------------------------------------
Request: I want to use augmented reality navigation
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1031, total~2403
[Token] Slice_Type_Determination: prompt~2367, completion~237, total~2604
[Token] Bandwidth_Analysis: prompt~216, completion~2019, total~2235
[Token] Allocate_Resources: prompt~2649, completion~1235, total~3884

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-17 14:05:43
Total Users: 25
Average Resource Utilization: 97.31%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 206.29 Mbps, mMTC Total Rate: 41.72 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         15  30.0/30 MHz               100.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 29 → URLLC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 4.76 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 12: 5.0 → 4.0 MHz, Rate: 23.82 → 19.06 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | URLLC   |     3 |        1   |          4.76 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3551, completion~1148, total~4699

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 1
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart parking sensor needs to report if the spot is free", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~829, total~2207
[Token] Slice_Type_Determination: prompt~2276, completion~281, total~2557
[Token] Bandwidth_Analysis: prompt~222, completion~1445, total~1667
[Token] Allocate_Resources: prompt~2521, completion~1718, total~4239

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-17 14:08:36
Total Users: 26
Average Resource Utilization: 98.0%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 206.29 Mbps, mMTC Total Rate: 44.91 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         15  30.0/30 MHz               100.00%
mMTC           6  7.400000000000001/10 MHz  74.00%

New User Allocation:
User 30 → mMTC Slice
CQI: 1, Bandwidth: 0.9 MHz, Rate: 3.19 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        5   |         20.62 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        4   |         19.06 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     2 |        1   |          4.12 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        3   |         20.92 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             80 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | mMTC    |     1 |        0.9 |          3.19 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        2.9 |         13.82 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3232, completion~744, total~3976

Detailed Slice Utilization Values:
eMBB utils: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.22, 22.22, 44.44, 66.67, 66.67, 66.67, 66.67, 88.89, 88.89, 88.89, 88.89, 88.89, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [16.67, 33.33, 33.33, 43.33, 43.33, 60.0, 66.67, 66.67, 83.33, 83.33, 83.33, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
mMTC utils: [0.0, 0.0, 9.0, 9.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 38.0, 47.0, 47.0, 47.0, 47.0, 56.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 74.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | URLLC   | URLLC          | Yes            |     2 |        5   |         20.62 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | URLLC   | URLLC          | Yes            |     3 |        5   |         23.82 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | mMTC    | mMTC           | Yes            |    15 |        0.9 |         13.62 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | URLLC   | URLLC          | Yes            |    15 |        3   |         45.41 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | mMTC    | mMTC           | Yes            |     3 |        2.9 |         13.82 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | URLLC   | URLLC          | Yes            |     4 |        5   |         27.28 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |    14 |        2   |         28.34 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | URLLC   | eMBB           | No             |     6 |        5   |         34.87 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             30 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | eMBB    | eMBB           | Yes            |     3 |       20   |         95.29 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     3 |        5   |         23.82 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | eMBB           | No             |     9 |        1   |          9.51 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | mMTC    | mMTC           | Yes            |     6 |        0.9 |          6.28 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |    13 |       20   |        264.25 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |     9 |        1   |          9.51 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | URLLC   | URLLC          | Yes            |    15 |        1   |         15.14 |              3 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | mMTC    | mMTC           | Yes            |     3 |        0.9 |          4.29 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | mMTC    | mMTC           | Yes            |     2 |        0.9 |          3.71 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    11 |       10   |        113.32 |             80 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Failed   | eMBB    | eMBB           | Yes            |     6 |       20   |        139.46 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | URLLC   | URLLC          | Yes            |     2 |        1   |          4.12 |              3 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Failed   | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Failed   | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | URLLC   | URLLC          | Yes            |    14 |        1   |         14.17 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | URLLC          | Yes            |     2 |        1   |          4.12 |              3 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | URLLC   | URLLC          | Yes            |    14 |        1   |         14.17 |              1 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Failed   | eMBB    | eMBB           | Yes            |     6 |       20   |        139.46 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | URLLC   | URLLC          | Yes            |     3 |        1   |          4.76 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Success  | mMTC    | mMTC           | Yes            |     1 |        0.9 |          3.19 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 26/30 (86.7%)

Intent Understanding Evaluation:
Correctly identified intents: 28/30
Intent understanding rate: 93.3%

Workload Balancing Statistics:
Users with workload balancing: 0/30
Workload balancing rate: 0.0%

Slice Utilization Statistics:
Average eMBB utilization: 57.69%
Average URLLC utilization: 81.28%
Average mMTC utilization: 43.58%

Weighted Average Utilization: 62.05%

Transmission Rate Statistics:
Final eMBB total rate: 691.08 Mbps
Final URLLC total rate: 206.29 Mbps
Final mMTC total rate: 44.91 Mbps

Resource Utilization:
Average resource utilization: 98.00%

Results exported to F:\code\wirelessagent\run_results\batch_run\kb\glm-5\network_slicing_results_TJU_south_glm-5.csv

✓ TJU_south 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\kb\glm-5\network_slicing_results_TJU_south_glm-5.csv