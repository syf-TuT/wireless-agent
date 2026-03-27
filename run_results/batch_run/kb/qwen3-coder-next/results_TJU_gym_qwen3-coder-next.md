场景 5/5: TJU_gym
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_gym.csv
输出: F:\code\wirelessagent\run_results\batch_run\kb\qwen3-coder-next\network_slicing_results_TJU_gym_qwen3-coder-next.csv
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
[RAG] Using knowledge base: F:\code\wirelessagent\with_knowledge_base\Intent_Understand.txt
[RAG] Initializing RAG system...
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 5149.27it/s]
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
[Token] Intent_Analysis: prompt~1374, completion~636, total~2010
[Token] Slice_Type_Determination: prompt~2454, completion~144, total~2598
[Token] Bandwidth_Analysis: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~2752, completion~242, total~2994

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-16 11:26:07
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
[Token] Network_Evaluation: prompt~3214, completion~460, total~3674

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I want to browse websites and check email"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~839, total~2209
[Token] Slice_Type_Determination: prompt~2625, completion~128, total~2753
[Token] Bandwidth_Analysis: prompt~213, completion~3, total~216
[Token] Allocate_Resources: prompt~2863, completion~190, total~3053

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-16 11:26:28
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
[Token] Network_Evaluation: prompt~3265, completion~375, total~3640

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to monitor water level in a reservoir"
CQI: 13
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor water level in a reservoir", "mmtc"),, ("i need to monitor water level in a reservoir periodically", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~917, total~2289
[Token] Slice_Type_Determination: prompt~2710, completion~84, total~2794
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~2949, completion~823, total~3772

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-16 11:26:51
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
[Token] Network_Evaluation: prompt~3951, completion~280, total~4231

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my wearable device needs to upload health data periodically", "mmtc"),, ("my wearable device needs to upload health data periodically", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~785, total~2157
[Token] Slice_Type_Determination: prompt~2580, completion~157, total~2737
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~2888, completion~268, total~3156

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-16 11:27:05
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
[Token] Network_Evaluation: prompt~3385, completion~388, total~3773

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~1235, total~2611
[Token] Slice_Type_Determination: prompt~3005, completion~111, total~3116
[Token] Bandwidth_Analysis: prompt~218, completion~3, total~221
[Token] Allocate_Resources: prompt~3230, completion~188, total~3418

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-16 11:27:22
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
[Token] Network_Evaluation: prompt~3632, completion~331, total~3963

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I need to stream 8K video content"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to stream 8k video content", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~809, total~2181
[Token] Slice_Type_Determination: prompt~2571, completion~135, total~2706
[Token] Bandwidth_Analysis: prompt~216, completion~3, total~219
[Token] Allocate_Resources: prompt~2839, completion~149, total~2988

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-16 11:28:00
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
[Token] Network_Evaluation: prompt~3203, completion~319, total~3522

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to track the location of a shipping container"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to track the location of a shipping container", "mmtc"),, ("i need to track the location of a shipping container", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~1043, total~2417
[Token] Slice_Type_Determination: prompt~2836, completion~138, total~2974
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3125, completion~456, total~3581

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-16 11:28:26
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
[Token] Network_Evaluation: prompt~3805, completion~339, total~4144

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my wearable device needs to upload health data periodically", "mmtc"),, ("my wearable device needs to upload health data periodically", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~809, total~2181
[Token] Slice_Type_Determination: prompt~2615, completion~123, total~2738
[Token] Bandwidth_Analysis: prompt~220, completion~2, total~222
[Token] Allocate_Resources: prompt~2886, completion~449, total~3335

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-16 11:28:47
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
[Token] Network_Evaluation: prompt~3546, completion~645, total~4191

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I need reliable connectivity for implanted medical devices"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable connectivity for implanted medical devices", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1179, total~2549
[Token] Slice_Type_Determination: prompt~2962, completion~119, total~3081
[Token] Bandwidth_Analysis: prompt~213, completion~2, total~215
[Token] Allocate_Resources: prompt~3192, completion~174, total~3366

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-16 11:29:34
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
[Token] Network_Evaluation: prompt~3593, completion~342, total~3935

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to check weather forecasts"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to check weather forecasts", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1366, completion~811, total~2177
[Token] Slice_Type_Determination: prompt~2610, completion~133, total~2743
WARNING: LLM didn't provide explicit slice recommendation. Using knowledge base recommendation.
[Token] Bandwidth_Analysis: prompt~213, completion~2, total~215
[Token] Allocate_Resources: prompt~2879, completion~173, total~3052

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-16 11:29:47
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
[Token] Network_Evaluation: prompt~3326, completion~278, total~3604

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of city-wide smart streetlights", "mmtc"),, ("i need to check the status of city-wide smart streetlights", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~1015, total~2393
[Token] Slice_Type_Determination: prompt~2825, completion~98, total~2923
[Token] Bandwidth_Analysis: prompt~218, completion~2, total~220
[Token] Allocate_Resources: prompt~3078, completion~369, total~3447

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-16 11:30:02
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
[Token] Network_Evaluation: prompt~3665, completion~503, total~4168

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need microsecond-level latency for high-frequency tradin"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need microsecond-level latency for high-frequency trading", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~868, total~2244
[Token] Slice_Type_Determination: prompt~2672, completion~173, total~2845
[Token] Bandwidth_Analysis: prompt~218, completion~2, total~220
[Token] Allocate_Resources: prompt~2952, completion~192, total~3144

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-16 11:30:16
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
[Token] Network_Evaluation: prompt~3373, completion~233, total~3606

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I want to use holographic communication"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use holographic communication", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1368, completion~1225, total~2593
[Token] Slice_Type_Determination: prompt~2973, completion~111, total~3084
[Token] Bandwidth_Analysis: prompt~214, completion~3, total~217
[Token] Allocate_Resources: prompt~3195, completion~199, total~3394

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-16 11:30:47
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
[Token] Network_Evaluation: prompt~3613, completion~405, total~4018

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 12
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control critical infrastructure with zero downtime", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1092, total~2464
[Token] Slice_Type_Determination: prompt~2873, completion~103, total~2976
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3090, completion~295, total~3385

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-16 11:31:02
Total Users: 14
Average Resource Utilization: 71.85%
eMBB Total Rate: 601.57 Mbps, URLLC Total Rate: 86.44 Mbps, mMTC Total Rate: 80.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  76.0/90 MHz       84.44%
URLLC          3  9.0/30 MHz        30.00%
mMTC           6  8.4/10 MHz        84.00%

New User Allocation:
User 14 → URLLC Slice
CQI: 12, Bandwidth: 5.0 MHz, Rate: 61.33 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 | NEW      |
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
[Token] Network_Evaluation: prompt~3608, completion~439, total~4047

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need real-time fraud detection for financial transactions"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need real-time fraud detection for financial transactions", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~994, total~2366
[Token] Slice_Type_Determination: prompt~2780, completion~137, total~2917
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3024, completion~159, total~3183

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-16 11:31:25
Total Users: 15
Average Resource Utilization: 73.38%
eMBB Total Rate: 601.57 Mbps, URLLC Total Rate: 102.02 Mbps, mMTC Total Rate: 80.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  76.0/90 MHz       84.44%
URLLC          4  11.0/30 MHz       36.67%
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
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
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
[Token] Network_Evaluation: prompt~3411, completion~329, total~3740

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need instant facial recognition for public security threats"
CQI: 10
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need instant facial recognition for public security threats", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1085, total~2457
[Token] Slice_Type_Determination: prompt~2879, completion~106, total~2985
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3100, completion~261, total~3361

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-16 11:31:49
Total Users: 16
Average Resource Utilization: 74.92%
eMBB Total Rate: 601.57 Mbps, URLLC Total Rate: 122.85 Mbps, mMTC Total Rate: 80.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  76.0/90 MHz       84.44%
URLLC          5  13.0/30 MHz       43.33%
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
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
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
[Token] Network_Evaluation: prompt~3577, completion~324, total~3901

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "My smart trash can needs to signal that it's full"
CQI: 13
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart trash can needs to signal that it's full", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~776, total~2152
[Token] Slice_Type_Determination: prompt~2586, completion~142, total~2728
[Token] Bandwidth_Analysis: prompt~237, completion~2, total~239
[Token] Allocate_Resources: prompt~2879, completion~728, total~3607

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-16 11:32:53
Total Users: 17
Average Resource Utilization: 75.62%
eMBB Total Rate: 601.57 Mbps, URLLC Total Rate: 122.85 Mbps, mMTC Total Rate: 92.27 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  76.0/90 MHz       84.44%
URLLC          5  13.0/30 MHz       43.33%
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
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
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
[Token] Network_Evaluation: prompt~3799, completion~386, total~4185

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1200, total~2570
[Token] Slice_Type_Determination: prompt~2984, completion~115, total~3099
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3214, completion~202, total~3416

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-16 11:33:14
Total Users: 18
Average Resource Utilization: 77.92%
eMBB Total Rate: 601.57 Mbps, URLLC Total Rate: 137.14 Mbps, mMTC Total Rate: 92.27 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  76.0/90 MHz       84.44%
URLLC          6  16.0/30 MHz       53.33%
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
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
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
[Token] Network_Evaluation: prompt~3641, completion~412, total~4053

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to monitor water level in a reservoir"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to monitor water level in a reservoir", "mmtc"),, ("i need to monitor water level in a reservoir periodically", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~980, total~2352
[Token] Slice_Type_Determination: prompt~2772, completion~134, total~2906
[Token] Bandwidth_Analysis: prompt~220, completion~4, total~224
[Token] Allocate_Resources: prompt~3116, completion~334, total~3450

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-16 11:33:29
Total Users: 19
Average Resource Utilization: 78.46%
eMBB Total Rate: 601.57 Mbps, URLLC Total Rate: 137.14 Mbps, mMTC Total Rate: 93.53 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  76.0/90 MHz       84.44%
URLLC          6  16.0/30 MHz       53.33%
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
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
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
[Token] Network_Evaluation: prompt~3763, completion~324, total~4087

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I need to make a high-quality voice call"
CQI: 10
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to make a high-quality voice call", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1697, total~3069
[Token] Slice_Type_Determination: prompt~3435, completion~142, total~3577
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3796, completion~357, total~4153

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-16 11:34:12
Total Users: 20
Average Resource Utilization: 89.23%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 137.14 Mbps, mMTC Total Rate: 93.53 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          6  16.0/30 MHz       53.33%
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
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
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
[Token] Network_Evaluation: prompt~4480, completion~388, total~4868

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to use holographic communication"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use holographic communication", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1368, completion~1499, total~2867
[Token] Slice_Type_Determination: prompt~3231, completion~173, total~3404
[Token] Bandwidth_Analysis: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~3579, completion~215, total~3794
[Token] Failure_Evaluation: prompt~3861, completion~363, total~4224

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
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~1020, total~2396
[Token] Slice_Type_Determination: prompt~2821, completion~116, total~2937
[Token] Bandwidth_Analysis: prompt~217, completion~4, total~221
[Token] Allocate_Resources: prompt~3176, completion~794, total~3970

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-16 11:35:02
Total Users: 21
Average Resource Utilization: 89.23%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 137.14 Mbps, mMTC Total Rate: 90.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          6  16.0/30 MHz       53.33%
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
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
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
[Token] Network_Evaluation: prompt~4297, completion~398, total~4695

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 1
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to detect and isolate power grid faults instantly", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~1013, total~2387
[Token] Slice_Type_Determination: prompt~2797, completion~112, total~2909
[Token] Bandwidth_Analysis: prompt~217, completion~2, total~219
[Token] Allocate_Resources: prompt~3024, completion~197, total~3221

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-16 11:35:20
Total Users: 22
Average Resource Utilization: 90.77%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 144.22 Mbps, mMTC Total Rate: 90.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          7  18.0/30 MHz       60.00%
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
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
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
[Token] Network_Evaluation: prompt~3450, completion~310, total~3760

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I want to use remote surgery equipment"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to use remote surgery equipment", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1368, completion~1099, total~2467
[Token] Slice_Type_Determination: prompt~2868, completion~97, total~2965
[Token] Bandwidth_Analysis: prompt~214, completion~2, total~216
[Token] Allocate_Resources: prompt~3080, completion~177, total~3257

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-16 11:35:35
Total Users: 23
Average Resource Utilization: 94.62%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 175.19 Mbps, mMTC Total Rate: 90.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          8  23.0/30 MHz       76.67%
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
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
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
[Token] Network_Evaluation: prompt~3485, completion~328, total~3813

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I want to stream music while browsing social media"
CQI: 2
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to stream music while browsing social media", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~838, total~2210
[Token] Slice_Type_Determination: prompt~2622, completion~175, total~2797
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~2982, completion~504, total~3486
[Token] Failure_Evaluation: prompt~3552, completion~414, total~3966

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
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1380, completion~1213, total~2593
[Token] Slice_Type_Determination: prompt~3007, completion~98, total~3105
[Token] Bandwidth_Analysis: prompt~220, completion~2, total~222
[Token] Allocate_Resources: prompt~3221, completion~149, total~3370

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-16 11:36:18
Total Users: 24
Average Resource Utilization: 96.15%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 189.14 Mbps, mMTC Total Rate: 90.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          9  25.0/30 MHz       83.33%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 6, Bandwidth: 2.0 MHz, Rate: 13.95 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
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
|        26 | URLLC   |     6 |        2   |         13.95 |              5 | NEW      |
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
[Token] Network_Evaluation: prompt~3598, completion~310, total~3908

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need to transmit real-time patient vital signs during critical care"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to transmit real-time patient vital signs during critical care", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~639, total~2017
[Token] Slice_Type_Determination: prompt~2451, completion~108, total~2559
[Token] Bandwidth_Analysis: prompt~219, completion~2, total~221
[Token] Allocate_Resources: prompt~2675, completion~240, total~2915

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-16 11:36:42
Total Users: 25
Average Resource Utilization: 97.69%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 198.67 Mbps, mMTC Total Rate: 90.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         10  27.0/30 MHz       90.00%
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
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
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
|        26 | URLLC   |     6 |        2   |         13.95 |              5 |          |
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
[Token] Network_Evaluation: prompt~3139, completion~413, total~3552

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1035, total~2405
[Token] Slice_Type_Determination: prompt~2827, completion~121, total~2948
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3057, completion~158, total~3215

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-16 11:37:01
Total Users: 26
Average Resource Utilization: 99.23%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 209.58 Mbps, mMTC Total Rate: 90.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         11  29.0/30 MHz       96.67%
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
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
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
|        26 | URLLC   |     6 |        2   |         13.95 |              5 |          |
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
[Token] Network_Evaluation: prompt~3443, completion~277, total~3720

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need to participate in an online multiplayer game"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to participate in an online multiplayer game", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~1203, total~2575
[Token] Slice_Type_Determination: prompt~2977, completion~103, total~3080
[Token] Bandwidth_Analysis: prompt~216, completion~2, total~218
[Token] Allocate_Resources: prompt~3193, completion~163, total~3356

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-16 11:37:16
Total Users: 27
Average Resource Utilization: 100.0%
eMBB Total Rate: 757.28 Mbps, URLLC Total Rate: 224.72 Mbps, mMTC Total Rate: 90.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         12  30.0/30 MHz       100.00%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 29 → URLLC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 15.14 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        12 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         61.33 |              3 |          |
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
|        26 | URLLC   |     6 |        2   |         13.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        2   |          9.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | URLLC   |     4 |        2   |         10.91 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | URLLC   |    15 |        1   |         15.14 |              5 | NEW      |
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
[Token] Network_Evaluation: prompt~3583, completion~404, total~3987

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I want to browse websites and check email"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~985, total~2355
[Token] Slice_Type_Determination: prompt~2762, completion~156, total~2918
[Token] Bandwidth_Analysis: prompt~215, completion~2, total~217
[Token] Allocate_Resources: prompt~3094, completion~432, total~3526
[Token] Failure_Evaluation: prompt~3597, completion~419, total~4016

----------------------------------------
ALLOCATION FAILED FOR USER 30
----------------------------------------
Request: I want to browse websites and check email
Slice type: eMBB
Reason: Insufficient capacity even after attempted adjustments

Detailed Slice Utilization Values:
eMBB utils: [0.0, 13.33, 13.33, 13.33, 26.67, 48.89, 48.89, 48.89, 48.89, 71.11, 71.11, 71.11, 84.44, 84.44, 84.44, 84.44, 84.44, 84.44, 84.44, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.67, 6.67, 6.67, 13.33, 13.33, 30.0, 36.67, 43.33, 43.33, 53.33, 53.33, 53.33, 53.33, 60.0, 76.67, 83.33, 90.0, 96.67, 100.0]
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
|        14 | Success  | URLLC   | URLLC          | Yes            |    12 |        5   |         61.33 |              3 | No         |
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
|        26 | Success  | URLLC   | URLLC          | Yes            |     6 |        2   |         13.95 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | URLLC   | URLLC          | Yes            |     3 |        2   |          9.53 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | URLLC   | URLLC          | Yes            |     4 |        2   |         10.91 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | URLLC   | URLLC          | Yes            |    15 |        1   |         15.14 |              5 | No         |
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
Average URLLC utilization: 34.07%
Average mMTC utilization: 77.33%

Weighted Average Utilization: 61.67%

Transmission Rate Statistics:
Final eMBB total rate: 757.28 Mbps
Final URLLC total rate: 224.72 Mbps
Final mMTC total rate: 90.50 Mbps

Resource Utilization:
Average resource utilization: 100.00%

Results exported to F:\code\wirelessagent\run_results\batch_run\kb\qwen3-coder-next\network_slicing_results_TJU_gym_qwen3-coder-next.csv

✓ TJU_gym 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\kb\qwen3-coder-next\network_slicing_results_TJU_gym_qwen3-coder-next.csv