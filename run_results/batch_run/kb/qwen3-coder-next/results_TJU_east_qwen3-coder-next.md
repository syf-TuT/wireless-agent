场景 3/5: TJU_east
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_east.csv
输出: F:\code\wirelessagent\run_results\batch_run\kb\qwen3-coder-next\network_slicing_results_TJU_east_qwen3-coder-next.csv
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
[RAG] Using knowledge base: F:\code\wirelessagent\with_knowledge_base\Intent_Understand.txt
[RAG] Initializing RAG system...
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 4302.84it/s]
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
[Token] Intent_Analysis: prompt~1366, completion~1128, total~2494
[Token] Slice_Type_Determination: prompt~2928, completion~133, total~3061
[Token] Bandwidth_Analysis: prompt~211, completion~3, total~214
[Token] Allocate_Resources: prompt~3174, completion~179, total~3353

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-16 11:05:21
Total Users: 1
Average Resource Utilization: 9.23%
eMBB Total Rate: 181.62 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  12.0/90 MHz       13.33%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → eMBB Slice
CQI: 15, Bandwidth: 12.0 MHz, Rate: 181.62 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         12 |        181.62 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3564, completion~323, total~3887

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need to stream 8K video content"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to stream 8k video content", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~997, total~2369
[Token] Slice_Type_Determination: prompt~2766, completion~138, total~2904
[Token] Bandwidth_Analysis: prompt~216, completion~3, total~219
[Token] Allocate_Resources: prompt~3043, completion~175, total~3218

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-16 11:05:38
Total Users: 2
Average Resource Utilization: 24.62%
eMBB Total Rate: 290.73 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  32.0/90 MHz       35.56%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         12 |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3424, completion~320, total~3744

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I want to listen to low-quality audio streaming"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to listen to low-quality audio streaming", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1074, total~2446
[Token] Slice_Type_Determination: prompt~2859, completion~113, total~2972
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~3036, completion~123, total~3159
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3299, completion~179, total~3478

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-16 11:05:55
Total Users: 3
Average Resource Utilization: 40.0%
eMBB Total Rate: 593.43 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  52.0/90 MHz       57.78%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 3 → eMBB Slice
CQI: 15, Bandwidth: 20.0 MHz, Rate: 302.70 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         12 |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3744, completion~302, total~4046

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I want to browse websites and check email"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~836, total~2206
[Token] Slice_Type_Determination: prompt~2626, completion~161, total~2787
[Token] Bandwidth_Analysis: prompt~215, completion~3, total~218
[Token] Allocate_Resources: prompt~2925, completion~221, total~3146

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-16 11:06:10
Total Users: 4
Average Resource Utilization: 55.38%
eMBB Total Rate: 783.73 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  72.0/90 MHz       80.00%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 4 → eMBB Slice
CQI: 9, Bandwidth: 20.0 MHz, Rate: 190.30 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         12 |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        190.3  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3347, completion~303, total~3650

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to monitor my home security cameras remotely", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1472, total~2844
[Token] Slice_Type_Determination: prompt~3233, completion~124, total~3357
LLM recommended eMBB but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~3422, completion~171, total~3593
[Token] Bandwidth_Analysis: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~3705, completion~219, total~3924

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-16 11:06:33
Total Users: 5
Average Resource Utilization: 56.92%
eMBB Total Rate: 783.73 Mbps, URLLC Total Rate: 22.66 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  72.0/90 MHz       80.00%
URLLC          1  2.0/30 MHz        6.67%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 5 → URLLC Slice
CQI: 11, Bandwidth: 2.0 MHz, Rate: 22.66 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |          2 |         22.66 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         12 |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4198, completion~307, total~4505

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I need to participate in a video conference meeting"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to participate in a video conference meeting", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1387, total~2759
[Token] Slice_Type_Determination: prompt~3146, completion~112, total~3258
[Token] Bandwidth_Analysis: prompt~216, completion~3, total~219
[Token] Allocate_Resources: prompt~3369, completion~134, total~3503

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-16 11:06:48
Total Users: 6
Average Resource Utilization: 66.15%
eMBB Total Rate: 930.92 Mbps, URLLC Total Rate: 22.66 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  84.0/90 MHz       93.33%
URLLC          1  2.0/30 MHz        6.67%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 6 → eMBB Slice
CQI: 12, Bandwidth: 12.0 MHz, Rate: 147.19 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |          2 |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         12 |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         12 |        147.19 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3720, completion~256, total~3976

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to monitor water level in a reservoir"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor water level in a reservoir", "mmtc"),, ("i need to monitor water level in a reservoir periodically", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~677, total~2049
[Token] Slice_Type_Determination: prompt~2496, completion~100, total~2596
[Token] Bandwidth_Analysis: prompt~213, completion~2, total~215
[Token] Allocate_Resources: prompt~2751, completion~418, total~3169

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-16 11:07:01
Total Users: 7
Average Resource Utilization: 67.62%
eMBB Total Rate: 930.92 Mbps, URLLC Total Rate: 22.66 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  84.0/90 MHz       93.33%
URLLC          1  2.0/30 MHz        6.67%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 7 → mMTC Slice
CQI: 6, Bandwidth: 1.9 MHz, Rate: 13.25 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       20   |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |       20   |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3382, completion~347, total~3729

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1013, total~2383
[Token] Slice_Type_Determination: prompt~2800, completion~134, total~2934
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3046, completion~184, total~3230

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-16 11:07:14
Total Users: 8
Average Resource Utilization: 69.15%
eMBB Total Rate: 930.92 Mbps, URLLC Total Rate: 41.69 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  84.0/90 MHz       93.33%
URLLC          2  4.0/30 MHz        13.33%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 19.03 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       20   |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |       20   |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3456, completion~305, total~3761

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to update my social media status"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1171, total~2541
[Token] Slice_Type_Determination: prompt~2957, completion~129, total~3086
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3304, completion~216, total~3520

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-16 11:07:33
Total Users: 9
Average Resource Utilization: 73.77%
eMBB Total Rate: 969.96 Mbps, URLLC Total Rate: 41.69 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          2  4.0/30 MHz        13.33%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 9 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 245.31 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 14.0 MHz
  User 3: 20.0 → 7.0 MHz, Rate: 302.70 → 105.95 Mbps, User 4: 20.0 → 19.0 MHz, Rate: 190.30 → 180.78 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |       12   |        181.62 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       20   |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       19   |        180.78 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |       20   |        245.31 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3850, completion~311, total~4161

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I want to update my social media status"
CQI: 13
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~976, total~2346
[Token] Slice_Type_Determination: prompt~2758, completion~143, total~2901
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3145, completion~187, total~3332

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-16 11:07:52
Total Users: 10
Average Resource Utilization: 73.77%
eMBB Total Rate: 985.56 Mbps, URLLC Total Rate: 41.69 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC          2  4.0/30 MHz        13.33%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 10 → eMBB Slice
CQI: 13, Bandwidth: 20.0 MHz, Rate: 264.25 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 3, Bandwidth freed: 20.0 MHz
  User 9: 20.0 → 9.0 MHz, Rate: 245.31 → 110.39 Mbps, User 1: 12.0 → 7.0 MHz, Rate: 181.62 → 105.95 Mbps, User 4: 19.0 → 15.0 MHz, Rate: 180.78 → 142.72 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |       20   |        264.25 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       20   |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       15   |        142.72 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3699, completion~283, total~3982

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need vehicle-to-vehicle collision avoidance systems"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need vehicle-to-vehicle collision avoidance systems", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1030, total~2402
[Token] Slice_Type_Determination: prompt~2796, completion~114, total~2910
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3015, completion~212, total~3227

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-16 11:08:04
Total Users: 11
Average Resource Utilization: 75.31%
eMBB Total Rate: 985.56 Mbps, URLLC Total Rate: 70.03 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC          3  6.0/30 MHz        20.00%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 14, Bandwidth: 2.0 MHz, Rate: 28.34 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       20   |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       15   |        142.72 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |       12   |        147.19 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3453, completion~393, total~3846

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need to sync my calendar and contacts"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to sync my calendar and contacts", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~821, total~2191
[Token] Slice_Type_Determination: prompt~2622, completion~121, total~2743
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~2804, completion~169, total~2973
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3257, completion~231, total~3488

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-16 11:08:29
Total Users: 12
Average Resource Utilization: 75.31%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 70.03 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          3  6.0/30 MHz        20.00%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 12 → eMBB Slice
CQI: 5, Bandwidth: 20.0 MHz, Rate: 123.87 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 4, Bandwidth freed: 20.0 MHz
  User 10: 20.0 → 8.0 MHz, Rate: 264.25 → 105.70 Mbps, User 6: 12.0 → 9.0 MHz, Rate: 147.19 → 110.39 Mbps, User 4: 15.0 → 11.0 MHz, Rate: 142.72 → 104.66 Mbps, User 2: 20.0 → 19.0 MHz, Rate: 109.11 → 103.65 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |        7   |        105.95 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |    13 |        8   |        105.7  |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | eMBB    |     5 |       20   |        123.87 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |       19   |        103.65 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |        7   |        105.95 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |       11   |        104.66 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |        9   |        110.39 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    12 |        9   |        110.39 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3951, completion~280, total~4231

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~922, total~2294
[Token] Slice_Type_Determination: prompt~2725, completion~134, total~2859
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2975, completion~176, total~3151

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-16 11:08:40
Total Users: 13
Average Resource Utilization: 76.85%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 80.94 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          4  8.0/30 MHz        26.67%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 4, Bandwidth: 2.0 MHz, Rate: 10.91 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        2   |         10.91 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
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
[Token] Network_Evaluation: prompt~3379, completion~347, total~3726

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I want to use holographic communication"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use holographic communication", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1368, completion~1397, total~2765
[Token] Slice_Type_Determination: prompt~3135, completion~158, total~3293
[Token] Bandwidth_Analysis: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~3479, completion~532, total~4011
[Token] Failure_Evaluation: prompt~4078, completion~398, total~4476

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
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1131, total~2501
[Token] Slice_Type_Determination: prompt~2916, completion~115, total~3031
[Token] Bandwidth_Analysis: prompt~215, completion~4, total~219
[Token] Allocate_Resources: prompt~3215, completion~206, total~3421
[Token] Failure_Evaluation: prompt~3496, completion~361, total~3857

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
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1368, completion~1028, total~2396
[Token] Slice_Type_Determination: prompt~2807, completion~94, total~2901
[Token] Bandwidth_Analysis: prompt~214, completion~4, total~218
[Token] Allocate_Resources: prompt~3016, completion~138, total~3154

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-16 11:09:56
Total Users: 14
Average Resource Utilization: 77.62%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 89.58 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          5  9.0/30 MHz        30.00%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 8.64 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        1   |          8.64 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
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
[Token] Network_Evaluation: prompt~3383, completion~201, total~3584

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to participate in a video conference meeting"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to participate in a video conference meeting", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~863, total~2235
[Token] Slice_Type_Determination: prompt~2650, completion~91, total~2741
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2927, completion~345, total~3272
[Token] Failure_Evaluation: prompt~3340, completion~406, total~3746

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
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~969, total~2347
[Token] Slice_Type_Determination: prompt~2762, completion~125, total~2887
[Token] Bandwidth_Analysis: prompt~219, completion~2, total~221
[Token] Allocate_Resources: prompt~3001, completion~201, total~3202

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-16 11:10:20
Total Users: 15
Average Resource Utilization: 79.15%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 100.49 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          6  11.0/30 MHz       36.67%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 4, Bandwidth: 2.0 MHz, Rate: 10.91 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        1   |          8.64 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        2   |         10.91 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
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
[Token] Network_Evaluation: prompt~3430, completion~356, total~3786

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~1111, total~2487
[Token] Slice_Type_Determination: prompt~2881, completion~116, total~2997
[Token] Bandwidth_Analysis: prompt~218, completion~2, total~220
[Token] Allocate_Resources: prompt~3180, completion~212, total~3392
[Token] Failure_Evaluation: prompt~3460, completion~346, total~3806

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
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~995, total~2367
[Token] Slice_Type_Determination: prompt~2783, completion~90, total~2873
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2988, completion~156, total~3144

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-16 11:10:57
Total Users: 16
Average Resource Utilization: 80.69%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 119.52 Mbps, mMTC Total Rate: 13.25 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          7  13.0/30 MHz       43.33%
mMTC           1  1.9/10 MHz        19.00%

New User Allocation:
User 20 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 19.03 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        1   |          8.64 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
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
[Token] Network_Evaluation: prompt~3372, completion~375, total~3747

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart agriculture sensor needs to report soil temperature", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~705, total~2077
[Token] Slice_Type_Determination: prompt~2514, completion~99, total~2613
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~2768, completion~463, total~3231

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-16 11:11:12
Total Users: 17
Average Resource Utilization: 81.38%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 119.52 Mbps, mMTC Total Rate: 20.26 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          7  13.0/30 MHz       43.33%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 21 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        1   |          8.64 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3428, completion~465, total~3893

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "I need to synchronize multiple robots on a factory floor"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to synchronize multiple robots on a factory floor", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~846, total~2220
[Token] Slice_Type_Determination: prompt~2638, completion~88, total~2726
[Token] Bandwidth_Analysis: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~2840, completion~159, total~2999

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-16 11:11:57
Total Users: 18
Average Resource Utilization: 82.92%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 138.55 Mbps, mMTC Total Rate: 20.26 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          8  15.0/30 MHz       50.00%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 22 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 19.03 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        1   |          8.64 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3226, completion~472, total~3698

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to control a robotic arm in real time"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control a robotic arm in real time", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~1059, total~2433
[Token] Slice_Type_Determination: prompt~2844, completion~95, total~2939
[Token] Bandwidth_Analysis: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~3054, completion~198, total~3252

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-16 11:12:16
Total Users: 19
Average Resource Utilization: 84.46%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 149.46 Mbps, mMTC Total Rate: 20.26 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          9  17.0/30 MHz       56.67%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 23 → URLLC Slice
CQI: 4, Bandwidth: 2.0 MHz, Rate: 10.91 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        1   |          8.64 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3478, completion~366, total~3844

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to detect and isolate power grid faults instantly", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~1005, total~2379
[Token] Slice_Type_Determination: prompt~2801, completion~103, total~2904
[Token] Bandwidth_Analysis: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~3020, completion~214, total~3234

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-16 11:12:33
Total Users: 20
Average Resource Utilization: 86.0%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 161.85 Mbps, mMTC Total Rate: 20.26 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         10  19.0/30 MHz       63.33%
mMTC           2  2.8/10 MHz        28.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 5, Bandwidth: 2.0 MHz, Rate: 12.39 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        1   |          8.64 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        2   |         12.39 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3460, completion~320, total~3780

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 2
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~690, total~2064
[Token] Slice_Type_Determination: prompt~2497, completion~107, total~2604
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2754, completion~530, total~3284

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-16 11:12:49
Total Users: 21
Average Resource Utilization: 87.46%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 161.85 Mbps, mMTC Total Rate: 28.10 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           8  90.0/90 MHz               100.00%
URLLC         10  19.0/30 MHz               63.33%
mMTC           3  4.699999999999999/10 MHz  47.00%

New User Allocation:
User 25 → mMTC Slice
CQI: 2, Bandwidth: 1.9 MHz, Rate: 7.84 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        1   |          8.64 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3504, completion~396, total~3900

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 1
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a fleet of delivery drones needs to send low-rate telemetry data", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~992, total~2370
[Token] Slice_Type_Determination: prompt~2793, completion~104, total~2897
[Token] Bandwidth_Analysis: prompt~222, completion~2, total~224
[Token] Allocate_Resources: prompt~3052, completion~154, total~3206

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-16 11:13:06
Total Users: 22
Average Resource Utilization: 88.15%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 161.85 Mbps, mMTC Total Rate: 31.29 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         10  19.0/30 MHz       63.33%
mMTC           4  5.6/10 MHz        56.00%

New User Allocation:
User 26 → mMTC Slice
CQI: 1, Bandwidth: 0.9 MHz, Rate: 3.19 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        1   |          8.64 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        0.9 |          3.19 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3439, completion~256, total~3695

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "My smart meter needs to report its reading"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart meter needs to report its reading", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~712, total~2082
[Token] Slice_Type_Determination: prompt~2504, completion~102, total~2606
[Token] Bandwidth_Analysis: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~2756, completion~348, total~3104

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-16 11:13:17
Total Users: 23
Average Resource Utilization: 88.85%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 161.85 Mbps, mMTC Total Rate: 38.30 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         10  19.0/30 MHz       63.33%
mMTC           5  6.5/10 MHz        65.00%

New User Allocation:
User 27 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        1   |          8.64 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        0.9 |          3.19 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3323, completion~416, total~3739

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("industrial equipment monitoring sensor data", "mmtc"))
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1364, completion~1215, total~2579
[Token] Slice_Type_Determination: prompt~2986, completion~148, total~3134
[Token] Bandwidth_Analysis: prompt~211, completion~2, total~213
[Token] Allocate_Resources: prompt~3282, completion~483, total~3765

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-16 11:14:05
Total Users: 24
Average Resource Utilization: 89.54%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 161.85 Mbps, mMTC Total Rate: 42.59 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         10  19.0/30 MHz       63.33%
mMTC           6  7.4/10 MHz        74.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 3, Bandwidth: 0.9 MHz, Rate: 4.29 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        1   |          8.64 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        0.9 |          3.19 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     3 |        0.9 |          4.29 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3985, completion~399, total~4384

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart agriculture sensor needs to report soil temperature", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~884, total~2256
[Token] Slice_Type_Determination: prompt~2684, completion~100, total~2784
[Token] Bandwidth_Analysis: prompt~235, completion~2, total~237
[Token] Allocate_Resources: prompt~2938, completion~503, total~3441

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-16 11:14:25
Total Users: 25
Average Resource Utilization: 90.23%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 161.85 Mbps, mMTC Total Rate: 51.15 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         10  19.0/30 MHz       63.33%
mMTC           7  8.3/10 MHz        83.00%

New User Allocation:
User 29 → mMTC Slice
CQI: 9, Bandwidth: 0.9 MHz, Rate: 8.56 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        1   |          8.64 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        0.9 |          3.19 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | mMTC    |     9 |        0.9 |          8.56 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3667, completion~256, total~3923

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I want to use remote surgery equipment"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to use remote surgery equipment", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1368, completion~837, total~2205
[Token] Slice_Type_Determination: prompt~2614, completion~117, total~2731
[Token] Bandwidth_Analysis: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~2841, completion~206, total~3047

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-16 11:14:58
Total Users: 26
Average Resource Utilization: 91.77%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 179.13 Mbps, mMTC Total Rate: 51.15 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         11  21.0/30 MHz       70.00%
mMTC           7  8.3/10 MHz        83.00%

New User Allocation:
User 30 → URLLC Slice
CQI: 8, Bandwidth: 2.0 MHz, Rate: 17.28 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        2   |         28.34 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        1   |          8.64 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        2   |         19.03 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        2   |         19.03 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        2   |         10.91 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        2   |         12.39 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | URLLC   |     8 |        2   |         17.28 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        2   |         22.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     9 |        2   |         19.03 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        0.9 |          3.19 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.9 |         13.25 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3272, completion~260, total~3532

Detailed Slice Utilization Values:
eMBB utils: [13.33, 35.56, 57.78, 80.0, 80.0, 93.33, 93.33, 93.33, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 0.0, 0.0, 0.0, 6.67, 6.67, 6.67, 13.33, 13.33, 13.33, 20.0, 20.0, 26.67, 30.0, 36.67, 43.33, 43.33, 50.0, 56.67, 63.33, 63.33, 63.33, 63.33, 63.33, 63.33, 70.0]
mMTC utils: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 19.0, 28.0, 28.0, 28.0, 28.0, 47.0, 56.0, 65.0, 74.0, 83.0, 83.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |    15 |       12   |        181.62 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | eMBB    | eMBB           | Yes            |    15 |       20   |        302.7  |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | eMBB    | eMBB           | Yes            |     9 |       20   |        190.3  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | URLLC   | eMBB           | No             |    11 |        2   |         22.66 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | eMBB    | eMBB           | Yes            |    12 |       12   |        147.19 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | mMTC    | mMTC           | Yes            |     6 |        1.9 |         13.25 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | URLLC          | Yes            |     9 |        2   |         19.03 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | eMBB    | eMBB           | Yes            |    13 |       20   |        264.25 |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |    14 |        2   |         28.34 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | eMBB    | eMBB           | Yes            |     5 |       20   |        123.87 |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | URLLC          | Yes            |     4 |        2   |         10.91 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Failed   | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Failed   | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |     8 |        1   |          8.64 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Failed   | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     4 |        2   |         10.91 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Failed   | eMBB    | eMBB           | Yes            |    11 |       20   |        226.64 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | URLLC   | URLLC          | Yes            |     9 |        2   |         19.03 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | URLLC   | URLLC          | Yes            |     9 |        2   |         19.03 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | URLLC   | URLLC          | Yes            |     4 |        2   |         10.91 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     5 |        2   |         12.39 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | mMTC    | mMTC           | Yes            |     2 |        1.9 |          7.84 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | mMTC    | mMTC           | Yes            |     1 |        0.9 |          3.19 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | mMTC    | mMTC           | Yes            |     3 |        0.9 |          4.29 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | mMTC    | mMTC           | Yes            |     9 |        0.9 |          8.56 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Success  | URLLC   | URLLC          | Yes            |     8 |        2   |         17.28 |              1 | No         |
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
Average eMBB utilization: 90.26%
Average URLLC utilization: 32.18%
Average mMTC utilization: 27.31%

Weighted Average Utilization: 72.01%

Transmission Rate Statistics:
Final eMBB total rate: 870.56 Mbps
Final URLLC total rate: 179.13 Mbps
Final mMTC total rate: 51.15 Mbps

Resource Utilization:
Average resource utilization: 91.77%

Results exported to F:\code\wirelessagent\run_results\batch_run\kb\qwen3-coder-next\network_slicing_results_TJU_east_qwen3-coder-next.csv

✓ TJU_east 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\kb\qwen3-coder-next\network_slicing_results_TJU_east_qwen3-coder-next.csv