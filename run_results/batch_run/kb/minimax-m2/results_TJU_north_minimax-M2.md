F:\code\wirelessagent\.venv\Scripts\python.exe F:\code\wirelessagent\with_knowledge_base\WA_DS_V3_KB.py 
============================================================
Initializing WirelessAgent with RAG optimization...
============================================================
[RAG] Using knowledge base: F:\code\wirelessagent\with_knowledge_base\Intent_Understand.txt
[RAG] Initializing RAG system...
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 8481.83it/s]
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
[SUCCESS] RAG system initialized - Semantic search enabled
============================================================
Starting network slicing processing...
============================================================
Starting network slice management system with CSV-based user testing...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I want to use augmented reality navigation"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use augmented reality navigation", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1368, completion~963, total~2331
[Token] Slice_Type_Determination: prompt~2808, completion~302, total~3110
[Token] Bandwidth_Analysis: prompt~212, completion~239, total~451
[Token] Allocate_Resources: prompt~3259, completion~968, total~4227

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-15 14:06:30
Total Users: 1
Average Resource Utilization: 15.38%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → eMBB Slice
CQI: 8, Bandwidth: 20.0 MHz, Rate: 172.78 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |     8 |         20 |        172.78 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4475, completion~589, total~5064

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~633, total~2005
[Token] Slice_Type_Determination: prompt~2487, completion~292, total~2779
[Token] Bandwidth_Analysis: prompt~214, completion~781, total~995
[Token] Allocate_Resources: prompt~2909, completion~747, total~3656

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-15 14:07:59
Total Users: 2
Average Resource Utilization: 19.23%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 30.97 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          1  5.0/30 MHz        16.67%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → URLLC Slice
CQI: 5, Bandwidth: 5.0 MHz, Rate: 30.97 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          5 |         30.97 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         20 |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3924, completion~779, total~4703

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 11
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to balance electrical load in real-time across microgrids", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1380, completion~782, total~2162
[Token] Slice_Type_Determination: prompt~2630, completion~397, total~3027
[Token] Bandwidth_Analysis: prompt~220, completion~474, total~694
[Token] Allocate_Resources: prompt~3154, completion~618, total~3772

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-15 14:09:22
Total Users: 3
Average Resource Utilization: 23.08%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 87.63 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          2  10.0/30 MHz       33.33%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 3 → URLLC Slice
CQI: 11, Bandwidth: 5.0 MHz, Rate: 56.66 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |          5 |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |          5 |         56.66 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |         20 |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4025, completion~680, total~4705

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of city-wide smart streetlights", "mmtc"),, ("i need to check the status of city-wide smart streetlights", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~898, total~2276
[Token] Slice_Type_Determination: prompt~2726, completion~307, total~3033
[Token] Bandwidth_Analysis: prompt~216, completion~563, total~779
[Token] Allocate_Resources: prompt~3193, completion~905, total~4098

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-15 14:11:16
Total Users: 4
Average Resource Utilization: 25.31%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 87.63 Mbps, mMTC Total Rate: 25.05 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          2  10.0/30 MHz       33.33%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 4 → mMTC Slice
CQI: 8, Bandwidth: 2.9 MHz, Rate: 25.05 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4358, completion~771, total~5129

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I want to use remote surgery equipment"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to use remote surgery equipment", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1368, completion~737, total~2105
[Token] Slice_Type_Determination: prompt~2582, completion~280, total~2862
[Token] Bandwidth_Analysis: prompt~214, completion~286, total~500
[Token] Allocate_Resources: prompt~2990, completion~787, total~3777

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-15 14:12:23
Total Users: 5
Average Resource Utilization: 26.08%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 101.80 Mbps, mMTC Total Rate: 25.05 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          3  11.0/30 MHz       36.67%
mMTC           1  2.9/10 MHz        29.00%

New User Allocation:
User 5 → URLLC Slice
CQI: 14, Bandwidth: 1.0 MHz, Rate: 14.17 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4041, completion~991, total~5032

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart parking sensor needs to report if the spot is free", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~747, total~2125
[Token] Slice_Type_Determination: prompt~2579, completion~276, total~2855
[Token] Bandwidth_Analysis: prompt~218, completion~424, total~642
[Token] Allocate_Resources: prompt~3015, completion~845, total~3860

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-15 14:14:00
Total Users: 6
Average Resource Utilization: 26.77%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 101.80 Mbps, mMTC Total Rate: 32.06 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          3  11.0/30 MHz       36.67%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 6 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4093, completion~886, total~4979

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~644, total~2016
[Token] Slice_Type_Determination: prompt~2499, completion~327, total~2826
[Token] Bandwidth_Analysis: prompt~216, completion~4313, total~4529
[Token] Allocate_Resources: prompt~2958, completion~781, total~3739

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-15 14:16:29
Total Users: 7
Average Resource Utilization: 27.54%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 108.77 Mbps, mMTC Total Rate: 32.06 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          4  12.0/30 MHz       40.00%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 6.97 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4007, completion~961, total~4968

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I need to transmit real-time patient vital signs during critical care"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to transmit real-time patient vital signs during critical care", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~938, total~2316
[Token] Slice_Type_Determination: prompt~2782, completion~326, total~3108
[Token] Bandwidth_Analysis: prompt~219, completion~474, total~693
[Token] Allocate_Resources: prompt~3234, completion~801, total~4035

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-15 14:17:42
Total Users: 8
Average Resource Utilization: 28.31%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 117.41 Mbps, mMTC Total Rate: 32.06 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          5  13.0/30 MHz       43.33%
mMTC           2  3.8/10 MHz        38.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 8.64 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4301, completion~1028, total~5329

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a fleet of delivery drones needs to send low-rate telemetry data", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~929, total~2307
[Token] Slice_Type_Determination: prompt~2765, completion~269, total~3034
[Token] Bandwidth_Analysis: prompt~218, completion~447, total~665
[Token] Allocate_Resources: prompt~3191, completion~1039, total~4230

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-15 14:19:12
Total Users: 9
Average Resource Utilization: 29.0%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 117.41 Mbps, mMTC Total Rate: 39.07 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          5  13.0/30 MHz       43.33%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 9 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4477, completion~701, total~5178

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~1031, total~2405
[Token] Slice_Type_Determination: prompt~2866, completion~415, total~3281
[Token] Bandwidth_Analysis: prompt~216, completion~377, total~593
[Token] Allocate_Resources: prompt~3440, completion~918, total~4358

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-15 14:20:28
Total Users: 10
Average Resource Utilization: 29.69%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 117.41 Mbps, mMTC Total Rate: 47.63 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           1  20.0/90 MHz                22.22%
URLLC          5  13.0/30 MHz                43.33%
mMTC           4  5.6000000000000005/10 MHz  56.00%

New User Allocation:
User 10 → mMTC Slice
CQI: 9, Bandwidth: 0.9 MHz, Rate: 8.56 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4605, completion~881, total~5486

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to detect and isolate power grid faults instantly", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~1010, total~2384
[Token] Slice_Type_Determination: prompt~2869, completion~227, total~3096
[Token] Bandwidth_Analysis: prompt~217, completion~487, total~704
[Token] Allocate_Resources: prompt~3224, completion~795, total~4019

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-15 14:21:40
Total Users: 11
Average Resource Utilization: 33.54%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 193.09 Mbps, mMTC Total Rate: 47.63 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           1  20.0/90 MHz                22.22%
URLLC          6  18.0/30 MHz                60.00%
mMTC           4  5.6000000000000005/10 MHz  56.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 15, Bandwidth: 5.0 MHz, Rate: 75.68 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4291, completion~804, total~5095

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need to check the status of my smart home sensors"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of my smart home sensors", "mmtc"),, ("i need to check the status of my smart home sensors", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~1038, total~2414
[Token] Slice_Type_Determination: prompt~2865, completion~591, total~3456
[Token] Bandwidth_Analysis: prompt~222, completion~499, total~721
[Token] Allocate_Resources: prompt~3615, completion~939, total~4554

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-15 14:23:39
Total Users: 12
Average Resource Utilization: 34.23%
eMBB Total Rate: 172.78 Mbps, URLLC Total Rate: 193.09 Mbps, mMTC Total Rate: 54.64 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           1  20.0/90 MHz               22.22%
URLLC          6  18.0/30 MHz               60.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 12 → mMTC Slice
CQI: 7, Bandwidth: 0.9 MHz, Rate: 7.01 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4801, completion~836, total~5637

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need to check weather forecasts"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to check weather forecasts", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1366, completion~1002, total~2368
[Token] Slice_Type_Determination: prompt~2835, completion~447, total~3282
[Token] Bandwidth_Analysis: prompt~213, completion~386, total~599
[Token] Allocate_Resources: prompt~3429, completion~766, total~4195

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-15 14:25:07
Total Users: 13
Average Resource Utilization: 49.62%
eMBB Total Rate: 345.56 Mbps, URLLC Total Rate: 193.09 Mbps, mMTC Total Rate: 54.64 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           2  40.0/90 MHz               44.44%
URLLC          6  18.0/30 MHz               60.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 13 → eMBB Slice
CQI: 8, Bandwidth: 20.0 MHz, Rate: 172.78 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4442, completion~794, total~5236

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need reliable communication for firefighters inside buildings", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~719, total~2089
[Token] Slice_Type_Determination: prompt~2560, completion~316, total~2876
[Token] Bandwidth_Analysis: prompt~215, completion~719, total~934
[Token] Allocate_Resources: prompt~3004, completion~862, total~3866

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-15 14:26:36
Total Users: 14
Average Resource Utilization: 53.46%
eMBB Total Rate: 345.56 Mbps, URLLC Total Rate: 240.66 Mbps, mMTC Total Rate: 54.64 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           2  40.0/90 MHz               44.44%
URLLC          7  23.0/30 MHz               76.67%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 14 → URLLC Slice
CQI: 9, Bandwidth: 5.0 MHz, Rate: 47.57 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4133, completion~912, total~5045

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use maps for basic navigation"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use maps for basic navigation", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~831, total~2201
[Token] Slice_Type_Determination: prompt~2666, completion~469, total~3135
[Token] Bandwidth_Analysis: prompt~215, completion~532, total~747
[Token] Allocate_Resources: prompt~3281, completion~932, total~4213

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-15 14:28:09
Total Users: 15
Average Resource Utilization: 68.85%
eMBB Total Rate: 535.86 Mbps, URLLC Total Rate: 240.66 Mbps, mMTC Total Rate: 54.64 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           3  60.0/90 MHz               66.67%
URLLC          7  23.0/30 MHz               76.67%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 9, Bandwidth: 20.0 MHz, Rate: 190.30 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4465, completion~1092, total~5557

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need to stream 8K video content"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to stream 8k video content", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~715, total~2087
[Token] Slice_Type_Determination: prompt~2551, completion~226, total~2777
[Token] Bandwidth_Analysis: prompt~216, completion~367, total~583
[Token] Allocate_Resources: prompt~2924, completion~912, total~3836

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-15 14:29:24
Total Users: 16
Average Resource Utilization: 84.23%
eMBB Total Rate: 781.17 Mbps, URLLC Total Rate: 240.66 Mbps, mMTC Total Rate: 54.64 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           4  80.0/90 MHz               88.89%
URLLC          7  23.0/30 MHz               76.67%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 16 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 245.31 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       20   |        245.31 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4092, completion~892, total~4984

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need instant alerts for life-threatening patient conditions"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need instant alerts for life-threatening patient conditions", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~774, total~2146
[Token] Slice_Type_Determination: prompt~2619, completion~380, total~2999
[Token] Bandwidth_Analysis: prompt~216, completion~610, total~826
[Token] Allocate_Resources: prompt~3118, completion~799, total~3917

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-15 14:30:49
Total Users: 17
Average Resource Utilization: 88.08%
eMBB Total Rate: 781.17 Mbps, URLLC Total Rate: 279.61 Mbps, mMTC Total Rate: 54.64 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           4  80.0/90 MHz               88.89%
URLLC          8  28.0/30 MHz               93.33%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 17 → URLLC Slice
CQI: 7, Bandwidth: 5.0 MHz, Rate: 38.95 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4193, completion~786, total~4979

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "My smart meter needs to report its reading"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart meter needs to report its reading", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~942, total~2312
[Token] Slice_Type_Determination: prompt~2759, completion~273, total~3032
[Token] Bandwidth_Analysis: prompt~218, completion~414, total~632
[Token] Allocate_Resources: prompt~3190, completion~925, total~4115

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-15 14:32:26
Total Users: 18
Average Resource Utilization: 88.77%
eMBB Total Rate: 781.17 Mbps, URLLC Total Rate: 279.61 Mbps, mMTC Total Rate: 62.42 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           4  80.0/90 MHz               88.89%
URLLC          8  28.0/30 MHz               93.33%
mMTC           6  7.400000000000001/10 MHz  74.00%

New User Allocation:
User 18 → mMTC Slice
CQI: 8, Bandwidth: 0.9 MHz, Rate: 7.78 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4371, completion~745, total~5116

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to control a robotic arm in real time"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control a robotic arm in real time", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~943, total~2317
[Token] Slice_Type_Determination: prompt~2792, completion~353, total~3145
[Token] Bandwidth_Analysis: prompt~217, completion~348, total~565
[Token] Allocate_Resources: prompt~3273, completion~862, total~4135

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-15 14:33:34
Total Users: 19
Average Resource Utilization: 90.31%
eMBB Total Rate: 781.17 Mbps, URLLC Total Rate: 295.19 Mbps, mMTC Total Rate: 62.42 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           4  80.0/90 MHz               88.89%
URLLC          9  30.0/30 MHz               100.00%
mMTC           6  7.400000000000001/10 MHz  74.00%

New User Allocation:
User 19 → URLLC Slice
CQI: 7, Bandwidth: 2.0 MHz, Rate: 15.58 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4402, completion~1076, total~5478

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I want to watch 4K video"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to watch 4k video", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~833, total~2203
[Token] Slice_Type_Determination: prompt~2669, completion~205, total~2874
[Token] Bandwidth_Analysis: prompt~215, completion~577, total~792
[Token] Allocate_Resources: prompt~2995, completion~681, total~3676

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-15 14:34:58
Total Users: 20
Average Resource Utilization: 98.0%
eMBB Total Rate: 894.49 Mbps, URLLC Total Rate: 295.19 Mbps, mMTC Total Rate: 62.42 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC          9  30.0/30 MHz               100.00%
mMTC           6  7.400000000000001/10 MHz  74.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 11, Bandwidth: 10.0 MHz, Rate: 113.32 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       20   |        190.3  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~3934, completion~878, total~4812

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to stream music while browsing social media"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to stream music while browsing social media", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~941, total~2313
[Token] Slice_Type_Determination: prompt~2784, completion~238, total~3022
[Token] Bandwidth_Analysis: prompt~216, completion~564, total~780
[Token] Allocate_Resources: prompt~3246, completion~767, total~4013

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-15 14:36:41
Total Users: 21
Average Resource Utilization: 98.0%
eMBB Total Rate: 919.24 Mbps, URLLC Total Rate: 295.19 Mbps, mMTC Total Rate: 62.42 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           6  90.0/90 MHz               100.00%
URLLC          9  30.0/30 MHz               100.00%
mMTC           6  7.400000000000001/10 MHz  74.00%

New User Allocation:
User 21 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 245.31 Mbps, Latency: 50.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 20.0 MHz
  User 16: 20.0 → 9.0 MHz, Rate: 245.31 → 110.39 Mbps, User 15: 20.0 → 11.0 MHz, Rate: 190.30 → 104.66 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |        9   |        110.39 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4386, completion~819, total~5205

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "My asset tracking device needs to send location update"
CQI: 12
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my asset tracking device needs to send location update", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1372, completion~817, total~2189
[Token] Slice_Type_Determination: prompt~2642, completion~284, total~2926
[Token] Bandwidth_Analysis: prompt~235, completion~587, total~822
[Token] Allocate_Resources: prompt~3091, completion~786, total~3877

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-15 14:38:01
Total Users: 22
Average Resource Utilization: 99.92%
eMBB Total Rate: 919.24 Mbps, URLLC Total Rate: 295.19 Mbps, mMTC Total Rate: 93.08 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           7  9.9/10 MHz        99.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 12, Bandwidth: 2.4999999999999987 MHz, Rate: 30.66 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        2.5 |         30.66 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4123, completion~600, total~4723

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a network of environmental sensors needs to report air quality", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~1050, total~2424
[Token] Slice_Type_Determination: prompt~2892, completion~233, total~3125
[Token] Bandwidth_Analysis: prompt~221, completion~586, total~807
[Token] Allocate_Resources: prompt~3342, completion~894, total~4236

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-15 14:39:26
Total Users: 23
Average Resource Utilization: 100.0%
eMBB Total Rate: 919.24 Mbps, URLLC Total Rate: 295.19 Mbps, mMTC Total Rate: 96.89 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC          9  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 23 → mMTC Slice
CQI: 15, Bandwidth: 0.9 MHz, Rate: 13.62 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.8000000000000004 MHz
  User 22: 2.4999999999999987 → 1.6999999999999984 MHz, Rate: 30.66 → 20.85 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        5   |         75.68 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        1.7 |         20.85 |            500 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4584, completion~973, total~5557

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to deploy early warning systems for natural disasters"
CQI: 8
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to deploy early warning systems for natural disasters", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~892, total~2266
[Token] Slice_Type_Determination: prompt~2738, completion~295, total~3033
LLM recommended mMTC but knowledge base recommended URLLC, using knowledge base recommendation
[Token] Intent_Override: prompt~3100, completion~442, total~3542
[Token] Bandwidth_Analysis: prompt~217, completion~377, total~594
[Token] Allocate_Resources: prompt~3723, completion~903, total~4626

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-15 14:41:04
Total Users: 24
Average Resource Utilization: 100.0%
eMBB Total Rate: 919.24 Mbps, URLLC Total Rate: 288.69 Mbps, mMTC Total Rate: 96.89 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  90.0/90 MHz       100.00%
URLLC         10  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 8, Bandwidth: 1.0 MHz, Rate: 8.64 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 11: 5.0 → 4.0 MHz, Rate: 75.68 → 60.54 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        4   |         60.54 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       20   |        172.78 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        245.31 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        1.7 |         20.85 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5022, completion~746, total~5768

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I want to update my social media status"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1370, completion~1074, total~2444
[Token] Slice_Type_Determination: prompt~2900, completion~329, total~3229
LLM recommended mMTC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~3300, completion~307, total~3607
[Token] Bandwidth_Analysis: prompt~215, completion~568, total~783
[Token] Allocate_Resources: prompt~3869, completion~817, total~4686

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-15 14:43:19
Total Users: 25
Average Resource Utilization: 100.0%
eMBB Total Rate: 862.37 Mbps, URLLC Total Rate: 288.69 Mbps, mMTC Total Rate: 96.89 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         10  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 25 → eMBB Slice
CQI: 7, Bandwidth: 20.0 MHz, Rate: 155.80 Mbps, Latency: 40.0 ms

Dynamic Resource Adjustments:
Users adjusted: 3, Bandwidth freed: 20.0 MHz
  User 21: 20.0 → 9.0 MHz, Rate: 245.31 → 110.39 Mbps, User 1: 20.0 → 12.0 MHz, Rate: 172.78 → 103.67 Mbps, User 13: 20.0 → 19.0 MHz, Rate: 172.78 → 164.14 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        4   |         60.54 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       19   |        164.14 |             40 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        1.7 |         20.85 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5153, completion~671, total~5824

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control precision cnc machines with zero tolerance for delay", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~865, total~2243
[Token] Slice_Type_Determination: prompt~2720, completion~282, total~3002
[Token] Bandwidth_Analysis: prompt~219, completion~439, total~658
[Token] Allocate_Resources: prompt~3175, completion~804, total~3979

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-15 14:44:37
Total Users: 26
Average Resource Utilization: 100.0%
eMBB Total Rate: 862.37 Mbps, URLLC Total Rate: 283.07 Mbps, mMTC Total Rate: 96.89 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         11  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 3.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 11: 4.0 → 3.0 MHz, Rate: 60.54 → 45.41 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        3   |         45.41 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        5   |         56.66 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        1.7 |         20.85 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4310, completion~837, total~5147

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need to deploy early warning systems for natural disasters"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to deploy early warning systems for natural disasters", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1374, completion~945, total~2319
[Token] Slice_Type_Determination: prompt~2788, completion~378, total~3166
[Token] Bandwidth_Analysis: prompt~217, completion~467, total~684
[Token] Allocate_Resources: prompt~3343, completion~962, total~4305

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-15 14:46:23
Total Users: 27
Average Resource Utilization: 100.0%
eMBB Total Rate: 862.37 Mbps, URLLC Total Rate: 281.25 Mbps, mMTC Total Rate: 96.89 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         12  30.0/30 MHz       100.00%
mMTC           8  10.0/10 MHz       100.00%

New User Allocation:
User 27 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 3: 5.0 → 4.0 MHz, Rate: 56.66 → 45.33 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        1   |          9.51 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        1.7 |         20.85 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2.9 |         25.05 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4642, completion~798, total~5440

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of city-wide smart streetlights", "mmtc"),, ("i need to check the status of city-wide smart streetlights", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1378, completion~1035, total~2413
[Token] Slice_Type_Determination: prompt~2876, completion~300, total~3176
[Token] Bandwidth_Analysis: prompt~218, completion~473, total~691
[Token] Allocate_Resources: prompt~3381, completion~899, total~4280

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-03-15 14:47:56
Total Users: 28
Average Resource Utilization: 100.0%
eMBB Total Rate: 862.37 Mbps, URLLC Total Rate: 281.25 Mbps, mMTC Total Rate: 96.90 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         12  30.0/30 MHz       100.00%
mMTC           9  10.0/10 MHz       100.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 8, Bandwidth: 0.9 MHz, Rate: 7.78 Mbps, Latency: 1000.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 0.9 MHz
  User 4: 2.9 → 2.0 MHz, Rate: 25.05 → 17.28 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        1.7 |         20.85 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     8 |        0.9 |          7.78 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     8 |        2   |         17.28 |           1000 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | mMTC    |     7 |        0.9 |          7.01 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4604, completion~974, total~5578

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need to check the status of my smart home sensors"
CQI: 8
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of my smart home sensors", "mmtc"),, ("i need to check the status of my smart home sensors", "mmtc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1376, completion~927, total~2303
[Token] Slice_Type_Determination: prompt~2759, completion~223, total~2982
[Token] Bandwidth_Analysis: prompt~217, completion~555, total~772
[Token] Allocate_Resources: prompt~3222, completion~855, total~4077

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-15 14:49:23
Total Users: 29
Average Resource Utilization: 100.0%
eMBB Total Rate: 862.37 Mbps, URLLC Total Rate: 281.25 Mbps, mMTC Total Rate: 94.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         12  30.0/30 MHz       100.00%
mMTC          10  10.0/10 MHz       100.00%

New User Allocation:
User 29 → mMTC Slice
CQI: 8, Bandwidth: 0.9 MHz, Rate: 7.78 Mbps, Latency: 500.0 ms

Dynamic Resource Adjustments:
Users adjusted: 2, Bandwidth freed: 0.9 MHz
  User 22: 1.6999999999999984 → 1.0 MHz, Rate: 20.85 → 12.27 Mbps, User 4: 2.0 → 1.7999999999999985 MHz, Rate: 17.28 → 15.55 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        5   |         47.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        1   |         12.27 |            500 | ADJUSTED |
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
[Token] Network_Evaluation: prompt~4463, completion~679, total~5142

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to balance electrical load in real-time across microgrids", "urllc"),)
[RAG] Using RAG-enhanced knowledge base query
[Token] Intent_Analysis: prompt~1380, completion~939, total~2319
[Token] Slice_Type_Determination: prompt~2792, completion~276, total~3068
[Token] Bandwidth_Analysis: prompt~220, completion~479, total~699
[Token] Allocate_Resources: prompt~3242, completion~729, total~3971

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-15 14:50:41
Total Users: 30
Average Resource Utilization: 100.0%
eMBB Total Rate: 862.37 Mbps, URLLC Total Rate: 279.53 Mbps, mMTC Total Rate: 94.37 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  90.0/90 MHz       100.00%
URLLC         13  30.0/30 MHz       100.00%
mMTC          10  10.0/10 MHz       100.00%

New User Allocation:
User 30 → URLLC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 7.79 Mbps, Latency: 5.0 ms

Dynamic Resource Adjustments:
Users adjusted: 1, Bandwidth freed: 1.0 MHz
  User 14: 5.0 → 4.0 MHz, Rate: 47.57 → 38.06 Mbps

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        11 | URLLC   |    15 |        3   |         45.41 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |     9 |        4   |         38.06 |              5 | ADJUSTED |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |     7 |        5   |         38.95 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     7 |        2   |         15.58 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     5 |        5   |         30.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     9 |        1   |          9.51 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |    11 |        4   |         45.33 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | URLLC   |     7 |        1   |          7.79 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |        1   |          6.97 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     8 |        1   |          8.64 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     8 |       12   |        103.67 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     8 |       19   |        164.14 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     9 |       11   |        104.66 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |       10   |        113.32 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |        9   |        110.39 |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     7 |       20   |        155.8  |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | mMTC    |     9 |        0.9 |          8.56 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | mMTC    |     7 |        0.9 |          7.01 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     8 |        0.9 |          7.78 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |    12 |        1   |         12.27 |            500 |          |
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
[Token] Network_Evaluation: prompt~4301, completion~573, total~4874

Detailed Slice Utilization Values:
eMBB utils: [22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 22.22, 44.44, 44.44, 66.67, 88.89, 88.89, 88.89, 88.89, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [0.0, 16.67, 33.33, 33.33, 36.67, 36.67, 40.0, 43.33, 43.33, 43.33, 60.0, 60.0, 60.0, 76.67, 76.67, 76.67, 93.33, 93.33, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
mMTC utils: [0.0, 0.0, 0.0, 29.0, 29.0, 38.0, 38.0, 38.0, 47.0, 56.0, 56.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 74.0, 74.0, 74.0, 74.0, 99.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |     8 |       20   |        172.78 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | URLLC   | URLLC          | Yes            |     5 |        5   |         30.97 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | URLLC   | URLLC          | Yes            |    11 |        5   |         56.66 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | mMTC    | mMTC           | Yes            |     8 |        2.9 |         25.05 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | URLLC   | URLLC          | Yes            |    14 |        1   |         14.17 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |     6 |        1   |          6.97 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | URLLC          | Yes            |     8 |        1   |          8.64 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | mMTC    | mMTC           | Yes            |     9 |        0.9 |          8.56 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |    15 |        5   |         75.68 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | mMTC    | mMTC           | Yes            |     7 |        0.9 |          7.01 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | eMBB    | eMBB           | Yes            |     8 |       20   |        172.78 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | URLLC   | URLLC          | Yes            |     9 |        5   |         47.57 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |     9 |       20   |        190.3  |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | URLLC   | URLLC          | Yes            |     7 |        5   |         38.95 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | mMTC    | mMTC           | Yes            |     8 |        0.9 |          7.78 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | URLLC   | URLLC          | Yes            |     7 |        2   |         15.58 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    11 |       10   |        113.32 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             50 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | mMTC    | mMTC           | Yes            |    12 |        2.5 |         30.66 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | mMTC    | mMTC           | Yes            |    15 |        0.9 |         13.62 |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     8 |        1   |          8.64 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | eMBB    | eMBB           | Yes            |     7 |       20   |        155.8  |             40 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | URLLC          | Yes            |     9 |        1   |          9.51 |              3 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | URLLC   | URLLC          | Yes            |     9 |        1   |          9.51 |              5 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | mMTC    | mMTC           | Yes            |     8 |        0.9 |          7.78 |           1000 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | mMTC    | mMTC           | Yes            |     8 |        0.9 |          7.78 |            500 | Yes        |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Success  | URLLC   | URLLC          | Yes            |     7 |        1   |          7.79 |              5 | Yes        |
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
Average eMBB utilization: 62.59%
Average URLLC utilization: 70.78%
Average mMTC utilization: 63.87%

Weighted Average Utilization: 64.58%

Transmission Rate Statistics:
Final eMBB total rate: 862.37 Mbps
Final URLLC total rate: 279.53 Mbps
Final mMTC total rate: 94.37 Mbps

Resource Utilization:
Average resource utilization: 100.00%

Results exported to F:\code\wirelessagent\run_results\with_kb\network_slicing_results_TJU_north_minimax-M2.csv

进程已结束，退出代码为 0
