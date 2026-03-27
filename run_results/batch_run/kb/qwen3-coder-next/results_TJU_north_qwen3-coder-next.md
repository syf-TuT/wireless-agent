F:\code\wirelessagent\.venv\Scripts\python.exe F:\code\wirelessagent\run_all_scenarios_kb.py 
============================================================
开始运行所有场景... (模型: qwen3-coder-next)
============================================================

============================================================
场景 1/5: TJU_north
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_north.csv
输出: F:\code\wirelessagent\run_results\batch_run\kb\qwen3-coder-next\network_slicing_results_TJU_north_qwen3-coder-next.csv
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
[RAG] Using knowledge base: F:\code\wirelessagent\with_knowledge_base\Intent_Understand.txt
[RAG] Initializing RAG system...
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 11377.45it/s]
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
[Token] Intent_Analysis: prompt~1368, completion~1531, total~2899
[Token] Slice_Type_Determination: prompt~3272, completion~122, total~3394
[Token] Bandwidth_Analysis: prompt~212, completion~3, total~215
[Token] Allocate_Resources: prompt~3502, completion~195, total~3697

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-16 10:43:28
Total Users: 1
Average Resource Utilization: 9.23%
eMBB Total Rate: 103.67 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → eMBB Slice
CQI: 8, Bandwidth: 12.0 MHz, Rate: 103.67 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |     8 |         12 |        103.67 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3906, completion~362, total~4268

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~730, total~2102
[Token] Slice_Type_Determination: prompt~2535, completion~112, total~2647
[Token] Bandwidth_Analysis: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~2763, completion~227, total~2990

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-16 10:43:42
Total Users: 2
Average Resource Utilization: 10.77%
eMBB Total Rate: 103.67 Mbps, URLLC Total Rate: 12.39 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          1  2.0/30 MHz        6.67%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → URLLC Slice
CQI: 5, Bandwidth: 2.0 MHz, Rate: 12.39 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |         12.39 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         12 |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3210, completion~290, total~3500

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 11
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to balance electrical load in real-time across microgrids", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1380, completion~1042, total~2422
[Token] Slice_Type_Determination: prompt~2841, completion~128, total~2969
[Token] Bandwidth_Analysis: prompt~220, completion~4, total~224
[Token] Allocate_Resources: prompt~3081, completion~170, total~3251

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-16 10:43:58
Total Users: 3
Average Resource Utilization: 11.54%
eMBB Total Rate: 103.67 Mbps, URLLC Total Rate: 23.72 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          2  3.0/30 MHz        10.00%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 3 → URLLC Slice
CQI: 11, Bandwidth: 1.0 MHz, Rate: 11.33 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          2 |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          1 |         11.33 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         12 |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3473, completion~336, total~3809

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of city-wide smart streetlights", "mmtc"),, ("i need to check the status of city-wide smart streetlights", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~754, total~2132
[Token] Slice_Type_Determination: prompt~2568, completion~107, total~2675
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2830, completion~356, total~3186

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-16 10:44:09
Total Users: 4
Average Resource Utilization: 13.0%
eMBB Total Rate: 103.67 Mbps, URLLC Total Rate: 23.72 Mbps, mMTC Total Rate: 16.41 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          2  3.0/30 MHz        10.00%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 4 → mMTC Slice
CQI: 8, Bandwidth: 1.9 MHz, Rate: 16.41 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3399, completion~319, total~3718

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I want to use remote surgery equipment"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to use remote surgery equipment", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1368, completion~790, total~2158
[Token] Slice_Type_Determination: prompt~2579, completion~145, total~2724
[Token] Bandwidth_Analysis: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~2832, completion~184, total~3016

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-16 10:44:48
Total Users: 5
Average Resource Utilization: 14.54%
eMBB Total Rate: 103.67 Mbps, URLLC Total Rate: 52.06 Mbps, mMTC Total Rate: 16.41 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          3  5.0/30 MHz        16.67%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 5 → URLLC Slice
CQI: 14, Bandwidth: 2.0 MHz, Rate: 28.34 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3241, completion~327, total~3568

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart parking sensor needs to report if the spot is free", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~892, total~2270
[Token] Slice_Type_Determination: prompt~2705, completion~122, total~2827
[Token] Bandwidth_Analysis: prompt~218, completion~2, total~220
[Token] Allocate_Resources: prompt~2981, completion~316, total~3297

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-16 10:45:00
Total Users: 6
Average Resource Utilization: 15.23%
eMBB Total Rate: 103.67 Mbps, URLLC Total Rate: 52.06 Mbps, mMTC Total Rate: 23.42 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          3  5.0/30 MHz        16.67%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 6 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3521, completion~306, total~3827

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~903, total~2275
[Token] Slice_Type_Determination: prompt~2694, completion~92, total~2786
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2902, completion~171, total~3073

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-16 10:45:28
Total Users: 7
Average Resource Utilization: 16.77%
eMBB Total Rate: 103.67 Mbps, URLLC Total Rate: 66.01 Mbps, mMTC Total Rate: 23.42 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          4  7.0/30 MHz        23.33%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 6, Bandwidth: 2.0 MHz, Rate: 13.95 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3301, completion~363, total~3664

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I need to transmit real-time patient vital signs during critical care"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to transmit real-time patient vital signs during critical care", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~829, total~2207
[Token] Slice_Type_Determination: prompt~2629, completion~118, total~2747
[Token] Bandwidth_Analysis: prompt~219, completion~2, total~221
[Token] Allocate_Resources: prompt~2863, completion~235, total~3098

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-16 10:45:39
Total Users: 8
Average Resource Utilization: 18.31%
eMBB Total Rate: 103.67 Mbps, URLLC Total Rate: 83.29 Mbps, mMTC Total Rate: 23.42 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          5  9.0/30 MHz        30.00%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 8, Bandwidth: 2.0 MHz, Rate: 17.28 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3323, completion~371, total~3694

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a fleet of delivery drones needs to send low-rate telemetry data", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~1067, total~2445
[Token] Slice_Type_Determination: prompt~2861, completion~125, total~2986
[Token] Bandwidth_Analysis: prompt~218, completion~2, total~220
[Token] Allocate_Resources: prompt~3135, completion~264, total~3399

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-16 10:45:54
Total Users: 9
Average Resource Utilization: 19.0%
eMBB Total Rate: 103.67 Mbps, URLLC Total Rate: 83.29 Mbps, mMTC Total Rate: 30.43 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           1  12.0/90 MHz                13.33%
URLLC          5  9.0/30 MHz                 30.00%
mMTC           3  3.6999999999999997/10 MHz  37.00%

New User Allocation:
User 9 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3632, completion~256, total~3888

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~825, total~2199
[Token] Slice_Type_Determination: prompt~2634, completion~133, total~2767
[Token] Bandwidth_Analysis: prompt~220, completion~2, total~222
[Token] Allocate_Resources: prompt~2917, completion~650, total~3567

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-16 10:46:51
Total Users: 10
Average Resource Utilization: 20.46%
eMBB Total Rate: 103.67 Mbps, URLLC Total Rate: 83.29 Mbps, mMTC Total Rate: 48.51 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          5  9.0/30 MHz        30.00%
mMTC           4  5.6/10 MHz        56.00%

New User Allocation:
User 10 → mMTC Slice
CQI: 9, Bandwidth: 1.9 MHz, Rate: 18.08 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3773, completion~321, total~4094

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to detect and isolate power grid faults instantly", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~952, total~2326
[Token] Slice_Type_Determination: prompt~2746, completion~96, total~2842
[Token] Bandwidth_Analysis: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~2958, completion~315, total~3273

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-16 10:47:03
Total Users: 11
Average Resource Utilization: 22.0%
eMBB Total Rate: 103.67 Mbps, URLLC Total Rate: 113.56 Mbps, mMTC Total Rate: 48.51 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          6  11.0/30 MHz       36.67%
mMTC           4  5.6/10 MHz        56.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 15, Bandwidth: 2.0 MHz, Rate: 30.27 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3500, completion~509, total~4009

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need to check the status of my smart home sensors"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of my smart home sensors", "mmtc"),, ("i need to check the status of my smart home sensors", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~1005, total~2381
[Token] Slice_Type_Determination: prompt~2807, completion~120, total~2927
[Token] Bandwidth_Analysis: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~3079, completion~592, total~3671

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-16 10:47:21
Total Users: 12
Average Resource Utilization: 22.69%
eMBB Total Rate: 103.67 Mbps, URLLC Total Rate: 113.56 Mbps, mMTC Total Rate: 55.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          6  11.0/30 MHz       36.67%
mMTC           5  6.5/10 MHz        65.00%

New User Allocation:
User 12 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3892, completion~423, total~4315

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need to check weather forecasts"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to check weather forecasts", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1366, completion~992, total~2358
[Token] Slice_Type_Determination: prompt~2782, completion~128, total~2910
[Token] Bandwidth_Analysis: prompt~213, completion~3, total~216
[Token] Allocate_Resources: prompt~3023, completion~193, total~3216

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-16 10:47:39
Total Users: 13
Average Resource Utilization: 31.92%
eMBB Total Rate: 207.34 Mbps, URLLC Total Rate: 113.56 Mbps, mMTC Total Rate: 55.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  24.0/90 MHz       26.67%
URLLC          6  11.0/30 MHz       36.67%
mMTC           5  6.5/10 MHz        65.00%

New User Allocation:
User 13 → eMBB Slice
CQI: 8, Bandwidth: 12.0 MHz, Rate: 103.67 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3430, completion~250, total~3680

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1061, total~2431
[Token] Slice_Type_Determination: prompt~2847, completion~114, total~2961
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3070, completion~172, total~3242

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-16 10:48:48
Total Users: 14
Average Resource Utilization: 33.46%
eMBB Total Rate: 207.34 Mbps, URLLC Total Rate: 132.59 Mbps, mMTC Total Rate: 55.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  24.0/90 MHz       26.67%
URLLC          7  13.0/30 MHz       43.33%
mMTC           5  6.5/10 MHz        65.00%

New User Allocation:
User 14 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 19.03 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3471, completion~379, total~3850

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use maps for basic navigation"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use maps for basic navigation", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1260, total~2630
[Token] Slice_Type_Determination: prompt~3035, completion~144, total~3179
[Token] Bandwidth_Analysis: prompt~215, completion~3, total~218
[Token] Allocate_Resources: prompt~3320, completion~287, total~3607

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-16 10:49:07
Total Users: 15
Average Resource Utilization: 48.85%
eMBB Total Rate: 397.64 Mbps, URLLC Total Rate: 132.59 Mbps, mMTC Total Rate: 55.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  44.0/90 MHz       48.89%
URLLC          7  13.0/30 MHz       43.33%
mMTC           5  6.5/10 MHz        65.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 9, Bandwidth: 20.0 MHz, Rate: 190.30 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3821, completion~219, total~4040

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need to stream 8K video content"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to stream 8k video content", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~831, total~2203
[Token] Slice_Type_Determination: prompt~2598, completion~143, total~2741
[Token] Bandwidth_Analysis: prompt~216, completion~3, total~219
[Token] Allocate_Resources: prompt~2847, completion~197, total~3044

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-16 10:49:48
Total Users: 16
Average Resource Utilization: 58.08%
eMBB Total Rate: 544.83 Mbps, URLLC Total Rate: 132.59 Mbps, mMTC Total Rate: 55.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  56.0/90 MHz       62.22%
URLLC          7  13.0/30 MHz       43.33%
mMTC           5  6.5/10 MHz        65.00%

New User Allocation:
User 16 → eMBB Slice
CQI: 12, Bandwidth: 12.0 MHz, Rate: 147.19 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       12   |        147.19 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3259, completion~433, total~3692

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need instant alerts for life-threatening patient conditions"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need instant alerts for life-threatening patient conditions", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1040, total~2412
[Token] Slice_Type_Determination: prompt~2830, completion~96, total~2926
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3041, completion~171, total~3212

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-16 10:50:14
Total Users: 17
Average Resource Utilization: 59.62%
eMBB Total Rate: 544.83 Mbps, URLLC Total Rate: 148.17 Mbps, mMTC Total Rate: 55.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  56.0/90 MHz       62.22%
URLLC          8  15.0/30 MHz       50.00%
mMTC           5  6.5/10 MHz        65.00%

New User Allocation:
User 17 → URLLC Slice
CQI: 7, Bandwidth: 2.0 MHz, Rate: 15.58 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |         15.58 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3437, completion~214, total~3651

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "My smart meter needs to report its reading"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart meter needs to report its reading", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~699, total~2069
[Token] Slice_Type_Determination: prompt~2496, completion~111, total~2607
[Token] Bandwidth_Analysis: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~2759, completion~165, total~2924

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-16 10:50:26
Total Users: 18
Average Resource Utilization: 60.31%
eMBB Total Rate: 544.83 Mbps, URLLC Total Rate: 148.17 Mbps, mMTC Total Rate: 63.30 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  56.0/90 MHz       62.22%
URLLC          8  15.0/30 MHz       50.00%
mMTC           6  7.4/10 MHz        74.00%

New User Allocation:
User 18 → mMTC Slice
CQI: 8, Bandwidth: 0.9 MHz, Rate: 7.78 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3156, completion~504, total~3660

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to control a robotic arm in real time"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control a robotic arm in real time", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~952, total~2326
[Token] Slice_Type_Determination: prompt~2753, completion~120, total~2873
[Token] Bandwidth_Analysis: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~2981, completion~239, total~3220

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-16 10:50:44
Total Users: 19
Average Resource Utilization: 61.85%
eMBB Total Rate: 544.83 Mbps, URLLC Total Rate: 163.75 Mbps, mMTC Total Rate: 63.30 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  56.0/90 MHz       62.22%
URLLC          9  17.0/30 MHz       56.67%
mMTC           6  7.4/10 MHz        74.00%

New User Allocation:
User 19 → URLLC Slice
CQI: 7, Bandwidth: 2.0 MHz, Rate: 15.58 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3446, completion~346, total~3792

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I want to watch 4K video"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to watch 4k video", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1251, total~2621
[Token] Slice_Type_Determination: prompt~2993, completion~123, total~3116
[Token] Bandwidth_Analysis: prompt~215, completion~3, total~218
[Token] Allocate_Resources: prompt~3225, completion~187, total~3412

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-16 10:50:59
Total Users: 20
Average Resource Utilization: 71.08%
eMBB Total Rate: 680.81 Mbps, URLLC Total Rate: 163.75 Mbps, mMTC Total Rate: 63.30 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  68.0/90 MHz       75.56%
URLLC          9  17.0/30 MHz       56.67%
mMTC           6  7.4/10 MHz        74.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 11, Bandwidth: 12.0 MHz, Rate: 135.98 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       12   |        135.98 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3630, completion~279, total~3909

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to stream music while browsing social media"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to stream music while browsing social media", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~936, total~2308
[Token] Slice_Type_Determination: prompt~2719, completion~119, total~2838
WARNING: LLM didn't provide explicit slice recommendation. Using knowledge base recommendation.
[Token] Bandwidth_Analysis: prompt~216, completion~3, total~219
[Token] Allocate_Resources: prompt~2952, completion~225, total~3177

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-16 10:51:10
Total Users: 21
Average Resource Utilization: 80.31%
eMBB Total Rate: 828.00 Mbps, URLLC Total Rate: 163.75 Mbps, mMTC Total Rate: 63.30 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  80.0/90 MHz       88.89%
URLLC          9  17.0/30 MHz       56.67%
mMTC           6  7.4/10 MHz        74.00%

New User Allocation:
User 21 → eMBB Slice
CQI: 12, Bandwidth: 12.0 MHz, Rate: 147.19 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       12   |        135.98 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       12   |        147.19 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3453, completion~365, total~3818

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "My asset tracking device needs to send location update"
CQI: 12
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my asset tracking device needs to send location update", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~651, total~2023
[Token] Slice_Type_Determination: prompt~2458, completion~126, total~2584
[Token] Bandwidth_Analysis: prompt~235, completion~2, total~237
[Token] Allocate_Resources: prompt~2733, completion~485, total~3218

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-16 10:51:23
Total Users: 22
Average Resource Utilization: 81.0%
eMBB Total Rate: 828.00 Mbps, URLLC Total Rate: 163.75 Mbps, mMTC Total Rate: 74.34 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  80.0/90 MHz       88.89%
URLLC          9  17.0/30 MHz       56.67%
mMTC           7  8.3/10 MHz        83.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 12, Bandwidth: 0.9 MHz, Rate: 11.04 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       12   |        135.98 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3422, completion~523, total~3945

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~750, total~2124
[Token] Slice_Type_Determination: prompt~2552, completion~109, total~2661
[Token] Bandwidth_Analysis: prompt~236, completion~2, total~238
[Token] Allocate_Resources: prompt~2811, completion~484, total~3295

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-16 10:51:48
Total Users: 23
Average Resource Utilization: 81.69%
eMBB Total Rate: 828.00 Mbps, URLLC Total Rate: 163.75 Mbps, mMTC Total Rate: 87.96 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  80.0/90 MHz               88.89%
URLLC          9  17.0/30 MHz               56.67%
mMTC           8  9.200000000000001/10 MHz  92.00%

New User Allocation:
User 23 → mMTC Slice
CQI: 15, Bandwidth: 0.9 MHz, Rate: 13.62 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       12   |        135.98 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3498, completion~359, total~3857

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to deploy early warning systems for natural disasters"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to deploy early warning systems for natural disasters", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~1282, total~2656
[Token] Slice_Type_Determination: prompt~3059, completion~102, total~3161
[Token] Bandwidth_Analysis: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~3273, completion~150, total~3423

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-16 10:52:30
Total Users: 24
Average Resource Utilization: 83.23%
eMBB Total Rate: 828.00 Mbps, URLLC Total Rate: 181.03 Mbps, mMTC Total Rate: 87.96 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  80.0/90 MHz               88.89%
URLLC         10  19.0/30 MHz               63.33%
mMTC           8  9.200000000000001/10 MHz  92.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 8, Bandwidth: 2.0 MHz, Rate: 17.28 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        2   |         17.28 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       12   |        135.98 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3655, completion~337, total~3992

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I want to update my social media status"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~913, total~2283
[Token] Slice_Type_Determination: prompt~2703, completion~138, total~2841
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3057, completion~199, total~3256

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-16 10:52:42
Total Users: 25
Average Resource Utilization: 90.92%
eMBB Total Rate: 885.89 Mbps, URLLC Total Rate: 181.03 Mbps, mMTC Total Rate: 87.96 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           7  90.0/90 MHz               100.00%
URLLC         10  19.0/30 MHz               63.33%
mMTC           8  9.200000000000001/10 MHz  92.00%

New User Allocation:
User 25 → eMBB Slice
CQI: 7, Bandwidth: 20.0 MHz, Rate: 155.80 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 10.0 MHz
  User 15: 20.0 → 11.0 MHz, Rate: 190.30 → 104.66 Mbps, User 16: 12.0 → 11.0 MHz, Rate: 147.19 → 134.92 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       11   |        134.92 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       12   |        135.98 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3580, completion~286, total~3866

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control precision cnc machines with zero tolerance for delay", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~607, total~1985
[Token] Slice_Type_Determination: prompt~2409, completion~109, total~2518
[Token] Bandwidth_Analysis: prompt~219, completion~2, total~221
[Token] Allocate_Resources: prompt~2630, completion~170, total~2800

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-16 10:53:01
Total Users: 26
Average Resource Utilization: 92.46%
eMBB Total Rate: 885.89 Mbps, URLLC Total Rate: 200.06 Mbps, mMTC Total Rate: 87.96 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           7  90.0/90 MHz               100.00%
URLLC         11  21.0/30 MHz               70.00%
mMTC           8  9.200000000000001/10 MHz  92.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 19.03 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        2   |         19.03 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       11   |        134.92 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       12   |        135.98 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3031, completion~335, total~3366

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need to deploy early warning systems for natural disasters"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to deploy early warning systems for natural disasters", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~1170, total~2544
[Token] Slice_Type_Determination: prompt~2967, completion~119, total~3086
LLM recommended mMTC but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~3154, completion~118, total~3272
[Token] Bandwidth_Analysis: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~3386, completion~165, total~3551

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-16 10:53:14
Total Users: 27
Average Resource Utilization: 94.0%
eMBB Total Rate: 885.89 Mbps, URLLC Total Rate: 219.09 Mbps, mMTC Total Rate: 87.96 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           7  90.0/90 MHz               100.00%
URLLC         12  23.0/30 MHz               76.67%
mMTC           8  9.200000000000001/10 MHz  92.00%

New User Allocation:
User 27 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 19.03 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        2   |         19.03 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       11   |        134.92 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       12   |        135.98 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.9 |         18.08 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        0.9 |         11.04 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3844, completion~444, total~4288

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of city-wide smart streetlights", "mmtc"),, ("i need to check the status of city-wide smart streetlights", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~993, total~2371
[Token] Slice_Type_Determination: prompt~2798, completion~136, total~2934
[Token] Bandwidth_Analysis: prompt~223, completion~4, total~227
[Token] Allocate_Resources: prompt~3142, completion~1214, total~4356

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-16 10:53:39
Total Users: 28
Average Resource Utilization: 94.62%
eMBB Total Rate: 885.89 Mbps, URLLC Total Rate: 219.09 Mbps, mMTC Total Rate: 94.79 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         12  23.0/30 MHz       76.67%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 8, Bandwidth: 0.9 MHz, Rate: 7.78 Mbps, Latency: 1000.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.10000000000000109 MHz
  User 10: 1.9 → 1.799999999999999 MHz, Rate: 18.08 → 17.13 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       11   |        134.92 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       12   |        135.98 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1.8 |         17.13 |            500 | ADJUSTED |
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
|         4 | mMTC    |     8 |        1.9 |         16.41 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4647, completion~547, total~5194

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need to check the status of my smart home sensors"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of my smart home sensors", "mmtc"),, ("i need to check the status of my smart home sensors", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~904, total~2280
[Token] Slice_Type_Determination: prompt~2709, completion~108, total~2817
[Token] Bandwidth_Analysis: prompt~217, completion~4, total~221
[Token] Allocate_Resources: prompt~3058, completion~478, total~3536

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-16 10:53:57
Total Users: 29
Average Resource Utilization: 94.62%
eMBB Total Rate: 885.89 Mbps, URLLC Total Rate: 219.09 Mbps, mMTC Total Rate: 94.09 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         12  23.0/30 MHz       76.67%
mMTC          10  10.0/10 MHz       100.00%

New User Allocation:
User 29 → mMTC Slice
CQI: 8, Bandwidth: 0.9 MHz, Rate: 7.78 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 0.9 MHz
  User 10: 1.799999999999999 → 1.0 MHz, Rate: 17.13 → 9.51 Mbps, User 4: 1.9 → 1.799999999999999 MHz, Rate: 16.41 → 15.55 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       11   |        134.92 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       12   |        135.98 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1   |          9.51 |            500 | ADJUSTED |
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
|         4 | mMTC    |     8 |        1.8 |         15.55 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3879, completion~415, total~4294

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to balance electrical load in real-time across microgrids", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1380, completion~1333, total~2713
[Token] Slice_Type_Determination: prompt~3111, completion~99, total~3210
[Token] Bandwidth_Analysis: prompt~220, completion~2, total~222
[Token] Allocate_Resources: prompt~3324, completion~183, total~3507

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-16 10:54:15
Total Users: 30
Average Resource Utilization: 96.15%
eMBB Total Rate: 885.89 Mbps, URLLC Total Rate: 234.67 Mbps, mMTC Total Rate: 94.09 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         13  25.0/30 MHz       83.33%
mMTC          10  10.0/10 MHz       100.00%

New User Allocation:
User 30 → URLLC Slice
CQI: 7, Bandwidth: 2.0 MHz, Rate: 15.58 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        2   |         30.27 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        2   |         15.58 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        1   |         11.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | URLLC   |     7 |        2   |         15.58 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        2   |         28.34 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        2   |         17.28 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       11   |        134.92 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       12   |        135.98 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        1   |          9.51 |            500 |          |
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
|         4 | mMTC    |     8 |        1.8 |         15.55 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3734, completion~385, total~4119

Detailed Slice Utilization Values:
eMBB utils: [13.33, 13.33, 13.33, 13.33, 13.33, 13.33, 13.33, 13.33, 13.33, 13.33, 13.33, 13.33, 26.67, 26.67, 48.89, 62.22, 62.22, 62.22, 62.22, 75.56, 88.89, 88.89, 88.89, 88.89, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 6.67, 10.0, 10.0, 16.67, 16.67, 23.33, 30.0, 30.0, 30.0, 36.67, 36.67, 36.67, 43.33, 43.33, 43.33, 50.0, 50.0, 56.67, 56.67, 56.67, 56.67, 56.67, 63.33, 63.33, 70.0, 76.67, 76.67, 76.67, 83.33]
mMTC utils: [0.0, 0.0, 0.0, 19.0, 19.0, 28.0, 28.0, 28.0, 37.0, 56.0, 56.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 74.0, 74.0, 74.0, 74.0, 83.0, 92.0, 92.0, 92.0, 92.0, 92.0, 100.0, 100.0, 100.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |     8 |       12   |        103.67 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | URLLC   | URLLC          | Yes            |     5 |        2   |         12.39 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | URLLC   | URLLC          | Yes            |    11 |        1   |         11.33 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | mMTC    | mMTC           | Yes            |     8 |        1.9 |         16.41 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | URLLC   | URLLC          | Yes            |    14 |        2   |         28.34 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |     6 |        2   |         13.95 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | URLLC          | Yes            |     8 |        2   |         17.28 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | mMTC    | mMTC           | Yes            |     9 |        1.9 |         18.08 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |    15 |        2   |         30.27 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | eMBB    | eMBB           | Yes            |     8 |       12   |        103.67 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | URLLC   | URLLC          | Yes            |     9 |        2   |         19.03 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |     9 |       20   |        190.3  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | eMBB    | eMBB           | Yes            |    12 |       12   |        147.19 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | URLLC   | URLLC          | Yes            |     7 |        2   |         15.58 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | mMTC    | mMTC           | Yes            |     8 |        0.9 |          7.78 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | URLLC   | URLLC          | Yes            |     7 |        2   |         15.58 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    11 |       12   |        135.98 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | eMBB    | eMBB           | Yes            |    12 |       12   |        147.19 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | mMTC    | mMTC           | Yes            |    12 |        0.9 |         11.04 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | mMTC    | mMTC           | Yes            |    15 |        0.9 |         13.62 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     8 |        2   |         17.28 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | URLLC          | Yes            |     9 |        2   |         19.03 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | URLLC   | URLLC          | Yes            |     9 |        2   |         19.03 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | mMTC    | mMTC           | Yes            |     8 |        0.9 |          7.78 |           1000 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | mMTC    | mMTC           | Yes            |     8 |        0.9 |          7.78 |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Success  | URLLC   | URLLC          | Yes            |     7 |        2   |         15.58 |              5 | No         |
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
Average eMBB utilization: 51.41%
Average URLLC utilization: 43.56%
Average mMTC utilization: 60.00%

Weighted Average Utilization: 50.26%

Transmission Rate Statistics:
Final eMBB total rate: 885.89 Mbps
Final URLLC total rate: 234.67 Mbps
Final mMTC total rate: 94.09 Mbps

Resource Utilization:
Average resource utilization: 96.15%

Results exported to F:\code\wirelessagent\run_results\batch_run\kb\qwen3-coder-next\network_slicing_results_TJU_north_qwen3-coder-next.csv

✓ TJU_north 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\kb\qwen3-coder-next\network_slicing_results_TJU_north_qwen3-coder-next.csv