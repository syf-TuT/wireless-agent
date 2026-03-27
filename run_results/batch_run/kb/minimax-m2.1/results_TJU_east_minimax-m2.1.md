============================================================
场景 3/5: TJU_east
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_east.csv
输出: F:\code\wirelessagent\run_results\batch_run\kb\minimax-m2.1\network_slicing_results_TJU_east_minimax-m2.1.csv
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
Loading weights: 100%|██████████| 103/103 [00:00<?, ?it/s]
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
[Token] Intent_Analysis: prompt~1366, completion~929, total~2295
[Token] Slice_Type_Determination: prompt~2762, completion~356, total~3118
[Token] Bandwidth_Analysis: prompt~211, completion~530, total~741
[Token] Allocate_Resources: prompt~3263, completion~781, total~4044

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-19 12:44:17
Total Users: 1
Average Resource Utilization: 15.38%
eMBB Total Rate: 302.70 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → eMBB Slice
CQI: 15, Bandwidth: 20.0 MHz, Rate: 302.70 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         20 |         302.7 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4290, completion~610, total~4900

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need to stream 8K video content"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to stream 8k video content", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~758, total~2130
[Token] Slice_Type_Determination: prompt~2591, completion~296, total~2887
[Token] Bandwidth_Analysis: prompt~216, completion~409, total~625
[Token] Allocate_Resources: prompt~3035, completion~870, total~3905

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-19 12:45:36
Total Users: 2
Average Resource Utilization: 30.77%
eMBB Total Rate: 411.81 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         20 |        302.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4146, completion~732, total~4878

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I want to listen to low-quality audio streaming"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to listen to low-quality audio streaming", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~958, total~2330
[Token] Slice_Type_Determination: prompt~2796, completion~307, total~3103
[Token] Bandwidth_Analysis: prompt~216, completion~328, total~544
[Token] Allocate_Resources: prompt~3252, completion~782, total~4034

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-19 12:47:10
Total Users: 3
Average Resource Utilization: 46.15%
eMBB Total Rate: 714.51 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 3 → eMBB Slice
CQI: 15, Bandwidth: 20.0 MHz, Rate: 302.70 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         20 |        302.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4276, completion~846, total~5122

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I want to browse websites and check email"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1089, total~2459
[Token] Slice_Type_Determination: prompt~2915, completion~282, total~3197
[Token] Bandwidth_Analysis: prompt~215, completion~434, total~649
[Token] Allocate_Resources: prompt~3345, completion~838, total~4183

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-19 12:48:32
Total Users: 4
Average Resource Utilization: 61.54%
eMBB Total Rate: 904.81 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 4 → eMBB Slice
CQI: 9, Bandwidth: 20.0 MHz, Rate: 190.30 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |    15 |         20 |        302.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        190.3  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4424, completion~608, total~5032

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to monitor my home security cameras remotely", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~936, total~2308
[Token] Slice_Type_Determination: prompt~2779, completion~409, total~3188
LLM recommended eMBB but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~3267, completion~551, total~3818
[Token] Bandwidth_Analysis: prompt~214, completion~330, total~544
[Token] Allocate_Resources: prompt~3958, completion~730, total~4688

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-19 12:50:11
Total Users: 5
Average Resource Utilization: 65.38%
eMBB Total Rate: 904.81 Mbps, URLLC Total Rate: 56.66 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          1  5.0/30 MHz        16.67%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 5 → URLLC Slice
CQI: 11, Bandwidth: 5.0 MHz, Rate: 56.66 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |          5 |         56.66 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         20 |        302.7  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5001, completion~904, total~5905

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I need to participate in a video conference meeting"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to participate in a video conference meeting", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~866, total~2238
[Token] Slice_Type_Determination: prompt~2715, completion~258, total~2973
[Token] Bandwidth_Analysis: prompt~216, completion~438, total~654
[Token] Allocate_Resources: prompt~3172, completion~835, total~4007

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-19 12:51:17
Total Users: 6
Average Resource Utilization: 73.08%
eMBB Total Rate: 998.77 Mbps, URLLC Total Rate: 56.66 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          1  5.0/30 MHz        16.67%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 6 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 245.31 Mbps, Latency: 50.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 10.0 MHz
  User 1: 20.0 → 10.0 MHz, Rate: 302.70 → 151.35 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |          5 |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |    15 |         10 |        151.35 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     4 |         20 |        109.11 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | eMBB    |    15 |         20 |        302.7  |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | eMBB    |     9 |         20 |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | eMBB    |    12 |         20 |        245.31 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4325, completion~892, total~5217

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to monitor water level in a reservoir"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor water level in a reservoir", "mmtc"),, ("i need to monitor water level in a reservoir periodically", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1123, total~2495
[Token] Slice_Type_Determination: prompt~2975, completion~522, total~3497
[Token] Bandwidth_Analysis: prompt~213, completion~1009, total~1222
[Token] Allocate_Resources: prompt~3657, completion~891, total~4548

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-19 12:52:50
Total Users: 7
Average Resource Utilization: 73.77%
eMBB Total Rate: 998.77 Mbps, URLLC Total Rate: 56.66 Mbps, mMTC Total Rate: 6.28 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          1  5.0/30 MHz        16.67%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 7 → mMTC Slice
CQI: 6, Bandwidth: 0.9 MHz, Rate: 6.28 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4787, completion~1136, total~5923

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~712, total~2082
[Token] Slice_Type_Determination: prompt~2552, completion~257, total~2809
[Token] Bandwidth_Analysis: prompt~215, completion~1533, total~1748
[Token] Allocate_Resources: prompt~2939, completion~735, total~3674

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-19 12:54:34
Total Users: 8
Average Resource Utilization: 77.62%
eMBB Total Rate: 998.77 Mbps, URLLC Total Rate: 104.23 Mbps, mMTC Total Rate: 6.28 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  90.0/90 MHz       100.00%
URLLC          2  10.0/30 MHz       33.33%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 9, Bandwidth: 5.0 MHz, Rate: 47.57 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3942, completion~767, total~4709

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to update my social media status"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1139, total~2509
[Token] Slice_Type_Determination: prompt~2966, completion~298, total~3264
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~3334, completion~463, total~3797
[Token] Bandwidth_Analysis: prompt~215, completion~498, total~713
[Token] Allocate_Resources: prompt~4025, completion~870, total~4895

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-19 12:56:01
Total Users: 9
Average Resource Utilization: 77.62%
eMBB Total Rate: 961.47 Mbps, URLLC Total Rate: 104.23 Mbps, mMTC Total Rate: 6.28 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          2  10.0/30 MHz       33.33%
mMTC           1  0.9/10 MHz        9.00%

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
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5316, completion~868, total~6184

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I want to update my social media status"
CQI: 13
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1010, total~2380
[Token] Slice_Type_Determination: prompt~2845, completion~656, total~3501
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~3573, completion~400, total~3973
[Token] Bandwidth_Analysis: prompt~215, completion~574, total~789
[Token] Allocate_Resources: prompt~4197, completion~935, total~5132

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-19 12:57:46
Total Users: 10
Average Resource Utilization: 77.62%
eMBB Total Rate: 1005.16 Mbps, URLLC Total Rate: 104.23 Mbps, mMTC Total Rate: 6.28 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC          2  10.0/30 MHz       33.33%
mMTC           1  0.9/10 MHz        9.00%

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
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5560, completion~755, total~6315

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need vehicle-to-vehicle collision avoidance systems"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need vehicle-to-vehicle collision avoidance systems", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1021, total~2393
[Token] Slice_Type_Determination: prompt~2881, completion~264, total~3145
[Token] Bandwidth_Analysis: prompt~216, completion~499, total~715
[Token] Allocate_Resources: prompt~3275, completion~805, total~4080

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-19 12:59:11
Total Users: 11
Average Resource Utilization: 81.46%
eMBB Total Rate: 1005.16 Mbps, URLLC Total Rate: 175.08 Mbps, mMTC Total Rate: 6.28 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC          3  15.0/30 MHz       50.00%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 14, Bandwidth: 5.0 MHz, Rate: 70.85 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        5   |         70.85 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4344, completion~931, total~5275

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need to sync my calendar and contacts"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to sync my calendar and contacts", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~834, total~2204
[Token] Slice_Type_Determination: prompt~2660, completion~364, total~3024
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~3089, completion~333, total~3422
[Token] Bandwidth_Analysis: prompt~215, completion~501, total~716
[Token] Allocate_Resources: prompt~3712, completion~929, total~4641

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-19 13:01:15
Total Users: 12
Average Resource Utilization: 81.46%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 175.08 Mbps, mMTC Total Rate: 6.28 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          3  15.0/30 MHz       50.00%
mMTC           1  0.9/10 MHz        9.00%

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
|        11 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5148, completion~876, total~6024

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1393, total~2765
[Token] Slice_Type_Determination: prompt~3272, completion~474, total~3746
[Token] Bandwidth_Analysis: prompt~216, completion~317, total~533
[Token] Allocate_Resources: prompt~3878, completion~795, total~4673

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-19 13:02:44
Total Users: 13
Average Resource Utilization: 82.23%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 180.54 Mbps, mMTC Total Rate: 6.28 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          4  16.0/30 MHz       53.33%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        1   |          5.46 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4942, completion~1037, total~5979

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I want to use holographic communication"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use holographic communication", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1368, completion~984, total~2352
[Token] Slice_Type_Determination: prompt~2817, completion~296, total~3113
[Token] Bandwidth_Analysis: prompt~214, completion~480, total~694
[Token] Allocate_Resources: prompt~3313, completion~797, total~4110
[Token] Failure_Evaluation: prompt~4204, completion~351, total~4555

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
[Token] Intent_Analysis: prompt~1370, completion~1066, total~2436
[Token] Slice_Type_Determination: prompt~2917, completion~456, total~3373
[Token] Bandwidth_Analysis: prompt~215, completion~524, total~739
[Token] Allocate_Resources: prompt~3558, completion~855, total~4413
[Token] Failure_Evaluation: prompt~4519, completion~550, total~5069

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
[Token] Intent_Analysis: prompt~1368, completion~725, total~2093
[Token] Slice_Type_Determination: prompt~2559, completion~225, total~2784
[Token] Bandwidth_Analysis: prompt~214, completion~457, total~671
[Token] Allocate_Resources: prompt~2906, completion~793, total~3699

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-19 13:07:59
Total Users: 14
Average Resource Utilization: 86.08%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 223.73 Mbps, mMTC Total Rate: 6.28 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          5  21.0/30 MHz       70.00%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 8, Bandwidth: 5.0 MHz, Rate: 43.19 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3967, completion~810, total~4777

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to participate in a video conference meeting"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to participate in a video conference meeting", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1123, total~2495
[Token] Slice_Type_Determination: prompt~2975, completion~295, total~3270
[Token] Bandwidth_Analysis: prompt~216, completion~434, total~650
[Token] Allocate_Resources: prompt~3464, completion~831, total~4295
[Token] Failure_Evaluation: prompt~4396, completion~618, total~5014

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
[Token] Intent_Analysis: prompt~1378, completion~768, total~2146
[Token] Slice_Type_Determination: prompt~2618, completion~262, total~2880
[Token] Bandwidth_Analysis: prompt~219, completion~485, total~704
[Token] Allocate_Resources: prompt~3004, completion~809, total~3813

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-19 13:10:48
Total Users: 15
Average Resource Utilization: 86.85%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 229.19 Mbps, mMTC Total Rate: 6.28 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          6  22.0/30 MHz       73.33%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4078, completion~726, total~4804

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~968, total~2344
[Token] Slice_Type_Determination: prompt~2813, completion~308, total~3121
[Token] Bandwidth_Analysis: prompt~218, completion~323, total~541
[Token] Allocate_Resources: prompt~3316, completion~772, total~4088
[Token] Failure_Evaluation: prompt~4193, completion~508, total~4701

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
[Token] Intent_Analysis: prompt~1372, completion~796, total~2168
[Token] Slice_Type_Determination: prompt~2643, completion~279, total~2922
[Token] Bandwidth_Analysis: prompt~216, completion~444, total~660
[Token] Allocate_Resources: prompt~3048, completion~905, total~3953

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-19 13:13:50
Total Users: 16
Average Resource Utilization: 87.62%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 238.70 Mbps, mMTC Total Rate: 6.28 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          7  23.0/30 MHz       76.67%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 20 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4216, completion~813, total~5029

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart agriculture sensor needs to report soil temperature", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~929, total~2301
[Token] Slice_Type_Determination: prompt~2754, completion~272, total~3026
[Token] Bandwidth_Analysis: prompt~215, completion~464, total~679
[Token] Allocate_Resources: prompt~3174, completion~888, total~4062

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-19 13:15:11
Total Users: 17
Average Resource Utilization: 88.31%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 238.70 Mbps, mMTC Total Rate: 13.29 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          7  23.0/30 MHz       76.67%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 21 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4306, completion~895, total~5201

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "I need to synchronize multiple robots on a factory floor"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to synchronize multiple robots on a factory floor", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~678, total~2052
[Token] Slice_Type_Determination: prompt~2515, completion~351, total~2866
[Token] Bandwidth_Analysis: prompt~217, completion~294, total~511
[Token] Allocate_Resources: prompt~2990, completion~862, total~3852

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-19 13:16:33
Total Users: 18
Average Resource Utilization: 92.15%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 286.27 Mbps, mMTC Total Rate: 13.29 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          8  28.0/30 MHz       93.33%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 22 → URLLC Slice
CQI: 9, Bandwidth: 5.0 MHz, Rate: 47.57 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4129, completion~961, total~5090

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to control a robotic arm in real time"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control a robotic arm in real time", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~860, total~2234
[Token] Slice_Type_Determination: prompt~2707, completion~242, total~2949
[Token] Bandwidth_Analysis: prompt~217, completion~289, total~506
[Token] Allocate_Resources: prompt~3072, completion~787, total~3859

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-19 13:18:04
Total Users: 19
Average Resource Utilization: 92.92%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 291.73 Mbps, mMTC Total Rate: 13.29 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC          9  29.0/30 MHz       96.67%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 23 → URLLC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 5.46 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4121, completion~1028, total~5149

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to detect and isolate power grid faults instantly", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~792, total~2166
[Token] Slice_Type_Determination: prompt~2635, completion~333, total~2968
[Token] Bandwidth_Analysis: prompt~217, completion~328, total~545
[Token] Allocate_Resources: prompt~3094, completion~832, total~3926

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-19 13:19:30
Total Users: 20
Average Resource Utilization: 93.69%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 297.92 Mbps, mMTC Total Rate: 13.29 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         10  30.0/30 MHz       100.00%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 5, Bandwidth: 1.0 MHz, Rate: 6.19 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4183, completion~928, total~5111

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 2
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~945, total~2319
[Token] Slice_Type_Determination: prompt~2782, completion~272, total~3054
[Token] Bandwidth_Analysis: prompt~216, completion~1128, total~1344
[Token] Allocate_Resources: prompt~3209, completion~916, total~4125

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-19 13:21:29
Total Users: 21
Average Resource Utilization: 95.15%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 297.92 Mbps, mMTC Total Rate: 21.13 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  90.0/90 MHz       100.00%
URLLC         10  30.0/30 MHz       100.00%
mMTC           3  3.7/10 MHz        37.00%

New User Allocation:
User 25 → mMTC Slice
CQI: 2, Bandwidth: 1.9 MHz, Rate: 7.84 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4371, completion~869, total~5240

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 1
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a fleet of delivery drones needs to send low-rate telemetry data", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~878, total~2256
[Token] Slice_Type_Determination: prompt~2713, completion~253, total~2966
[Token] Bandwidth_Analysis: prompt~218, completion~669, total~887
[Token] Allocate_Resources: prompt~3121, completion~923, total~4044

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-19 13:22:59
Total Users: 22
Average Resource Utilization: 95.85%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 297.92 Mbps, mMTC Total Rate: 24.32 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           8  90.0/90 MHz                100.00%
URLLC         10  30.0/30 MHz                100.00%
mMTC           4  4.6000000000000005/10 MHz  46.00%

New User Allocation:
User 26 → mMTC Slice
CQI: 1, Bandwidth: 0.9 MHz, Rate: 3.19 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        0.9 |          3.19 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4310, completion~943, total~5253

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "My smart meter needs to report its reading"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart meter needs to report its reading", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~943, total~2313
[Token] Slice_Type_Determination: prompt~2763, completion~243, total~3006
[Token] Bandwidth_Analysis: prompt~219, completion~446, total~665
[Token] Allocate_Resources: prompt~3160, completion~969, total~4129

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-19 13:23:59
Total Users: 23
Average Resource Utilization: 96.54%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 297.92 Mbps, mMTC Total Rate: 31.33 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           8  90.0/90 MHz               100.00%
URLLC         10  30.0/30 MHz               100.00%
mMTC           5  5.500000000000001/10 MHz  55.00%

New User Allocation:
User 27 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|        21 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     2 |        1.9 |          7.84 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | mMTC    |     1 |        0.9 |          3.19 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4388, completion~862, total~5250

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("industrial equipment monitoring sensor data", "mmtc"))
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1364, completion~898, total~2262
[Token] Slice_Type_Determination: prompt~2730, completion~270, total~3000
LLM recommended URLLC but knowledge base recommended mMTC, using knowledge base recommendation
[Token] Intent_Override: prompt~3072, completion~439, total~3511
[Token] Bandwidth_Analysis: prompt~215, completion~468, total~683
[Token] Allocate_Resources: prompt~3670, completion~876, total~4546

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-19 13:25:26
Total Users: 24
Average Resource Utilization: 97.23%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 297.92 Mbps, mMTC Total Rate: 35.62 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           8  90.0/90 MHz               100.00%
URLLC         10  30.0/30 MHz               100.00%
mMTC           6  6.400000000000001/10 MHz  64.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 3, Bandwidth: 0.9 MHz, Rate: 4.29 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4851, completion~853, total~5704

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart agriculture sensor needs to report soil temperature", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~802, total~2174
[Token] Slice_Type_Determination: prompt~2634, completion~252, total~2886
[Token] Bandwidth_Analysis: prompt~220, completion~622, total~842
[Token] Allocate_Resources: prompt~3044, completion~901, total~3945

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-19 13:26:54
Total Users: 25
Average Resource Utilization: 97.92%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 297.92 Mbps, mMTC Total Rate: 44.18 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           8  90.0/90 MHz               100.00%
URLLC         10  30.0/30 MHz               100.00%
mMTC           7  7.300000000000002/10 MHz  73.00%

New User Allocation:
User 29 → mMTC Slice
CQI: 9, Bandwidth: 0.9 MHz, Rate: 8.56 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        5   |         70.85 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4208, completion~836, total~5044

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I want to use remote surgery equipment"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to use remote surgery equipment", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1368, completion~929, total~2297
[Token] Slice_Type_Determination: prompt~2776, completion~173, total~2949
[Token] Bandwidth_Analysis: prompt~214, completion~400, total~614
[Token] Allocate_Resources: prompt~3123, completion~826, total~3949

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-19 13:28:31
Total Users: 26
Average Resource Utilization: 97.92%
eMBB Total Rate: 870.56 Mbps, URLLC Total Rate: 292.39 Mbps, mMTC Total Rate: 44.18 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           8  90.0/90 MHz               100.00%
URLLC         11  30.0/30 MHz               100.00%
mMTC           7  7.300000000000002/10 MHz  73.00%

New User Allocation:
User 30 → URLLC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 8.64 Mbps, Latency: 1.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 11: 5.0 → 4.0 MHz, Rate: 70.85 → 56.68 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    14 |        4   |         56.68 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     4 |        1   |          5.46 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     8 |        5   |         43.19 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     9 |        5   |         47.57 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | URLLC   |     4 |        1   |          5.46 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        1   |          6.19 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | URLLC   |     8 |        1   |          8.64 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    11 |        5   |         56.66 |              5 |          |
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
|         7 | mMTC    |     6 |        0.9 |          6.28 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4296, completion~761, total~5057

Detailed Slice Utilization Values:
eMBB utils: [22.22, 44.44, 66.67, 88.89, 88.89, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 0.0, 0.0, 0.0, 16.67, 16.67, 16.67, 33.33, 33.33, 33.33, 50.0, 50.0, 53.33, 70.0, 73.33, 76.67, 76.67, 93.33, 96.67, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
mMTC utils: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 18.0, 18.0, 18.0, 18.0, 37.0, 46.0, 55.0, 64.0, 73.0, 73.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |    15 |       20   |        302.7  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | eMBB    | eMBB           | Yes            |    15 |       20   |        302.7  |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | eMBB    | eMBB           | Yes            |     9 |       20   |        190.3  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | URLLC   | eMBB           | No             |    11 |        5   |         56.66 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             50 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | mMTC    | mMTC           | Yes            |     6 |        0.9 |          6.28 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | URLLC          | Yes            |     9 |        5   |         47.57 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | eMBB    | eMBB           | Yes            |    13 |       20   |        264.25 |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |    14 |        5   |         70.85 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | eMBB    | eMBB           | Yes            |     5 |       20   |        123.87 |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          5.46 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Failed   | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Failed   | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |     8 |        5   |         43.19 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Failed   | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          5.46 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Failed   | eMBB    | eMBB           | Yes            |    11 |       20   |        226.64 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | URLLC   | URLLC          | Yes            |     9 |        1   |          9.51 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | URLLC   | URLLC          | Yes            |     9 |        5   |         47.57 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | URLLC   | URLLC          | Yes            |     4 |        1   |          5.46 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     5 |        1   |          6.19 |              5 | No         |
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
Average eMBB utilization: 92.73%
Average URLLC utilization: 57.31%
Average mMTC utilization: 19.62%

Weighted Average Utilization: 78.93%

Transmission Rate Statistics:
Final eMBB total rate: 870.56 Mbps
Final URLLC total rate: 292.39 Mbps
Final mMTC total rate: 44.18 Mbps

Resource Utilization:
Average resource utilization: 97.92%

Results exported to F:\code\wirelessagent\run_results\batch_run\kb\minimax-m2.1\network_slicing_results_TJU_east_minimax-m2.1.csv

✓ TJU_east 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\kb\minimax-m2.1\network_slicing_results_TJU_east_minimax-m2.1.csv