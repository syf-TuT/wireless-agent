Active code page: 65001                                                                                                                                                                                  
(.venv) PS F:\code\wirelessagent> python with_knowledge_base/WA_DS_V3_FKB.py                                                                                                                             
Starting network slice management system with CSV-based user testing...                                                                                                                                  
                                                                                                                                                                                                         
Testing 30 users from ray tracing results CSV                                                                                                                                                            
                                                                                                                                                                                                         
--------------------------------------------------------------------------------------------------------------------------------------------                                                             
PROCESSING USER 1 (1/30)                                                                                                                                                                                 
Request: "I need immediate machine shutdown capability for safety incidents"                                                                                                                             
CQI: 2                                                                                                                                                                                                   
Ground Truth Slice: URLLC                                                                                                                                                                                
--------------------------------------------------------------------------------------------------------------------------------------------                                                             
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need immediate machine shutdown capability for safety incidents", "urllc"),)                                                          
[Token] Intent_Analysis: prompt~1372, completion~983, total~2355                                                                                                                                         
[Token] Slice_Type_Determination: prompt~2843, completion~274, total~3117                                                                                                                                
[Token] Beamforming_Bandwidth: prompt~214, completion~1174, total~1388                                                                                                                                   
[Token] Allocate_Resources: prompt~3251, completion~890, total~4141                                                                                                                                      
                                                                                                                                                                                                         
----------------------------------------                                                                                                                                                                 
ALLOCATION RESULT FOR USER 1                                                                                                                                                                             
----------------------------------------                                                                                                                                                                 
Network Status @ 2026-03-15 11:04:50                                                                                                                                                                     
Total Users: 1                                                                                                                                                                                           
Average Resource Utilization: 0.77%                                                                                                                                                                      
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 4.12 Mbps, mMTC Total Rate: 0.00 Mbps                                                                                                                      
                                                                                                                                                                                                         
Slice      Users  Resource Usage    Utilization                                                                                                                                                          
-------  -------  ----------------  -------------                                                                                                                                                        
eMBB           0  0/90 MHz          0%                                                                                                                                                                   
URLLC          1  1.0/30 MHz        3.33%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → URLLC Slice
CQI: 2, Bandwidth: 1.0 MHz, Rate: 4.12 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          1 |          4.12 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4425, completion~786, total~5211

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need my autonomous vehicle to communicate in real time"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need my autonomous vehicle to communicate in real time", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~795, total~2169
[Token] Slice_Type_Determination: prompt~2650, completion~342, total~2992
[Token] Beamforming_Bandwidth: prompt~217, completion~470, total~687
[Token] Allocate_Resources: prompt~3123, completion~808, total~3931

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-15 11:05:37
Total Users: 2
Average Resource Utilization: 1.54%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 8.88 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          2  2.0/30 MHz        6.67%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → URLLC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 4.76 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |          1 |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |          1 |          4.76 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4188, completion~767, total~4955

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart agriculture sensor needs to report soil temperature", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~849, total~2221
[Token] Slice_Type_Determination: prompt~2687, completion~423, total~3110
[Token] Beamforming_Bandwidth: prompt~213, completion~824, total~1037
[Token] Allocate_Resources: prompt~3259, completion~819, total~4078

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-15 11:06:55
Total Users: 3
Average Resource Utilization: 2.23%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 8.88 Mbps, mMTC Total Rate: 13.62 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          2  2.0/30 MHz        6.67%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 3 → mMTC Slice
CQI: 15, Bandwidth: 0.9 MHz, Rate: 13.62 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4329, completion~807, total~5136

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to participate in an online multiplayer game"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to participate in an online multiplayer game", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~1075, total~2447
[Token] Slice_Type_Determination: prompt~2912, completion~315, total~3227
[Token] Beamforming_Bandwidth: prompt~216, completion~448, total~664
[Token] Allocate_Resources: prompt~3352, completion~814, total~4166

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-15 11:07:55
Total Users: 4
Average Resource Utilization: 3.0%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 24.02 Mbps, mMTC Total Rate: 13.62 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          3  3.0/30 MHz        10.00%
mMTC           1  0.9/10 MHz        9.00%

New User Allocation:
User 4 → URLLC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 15.14 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4434, completion~732, total~5166

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("a fleet of delivery drones needs to send low-rate telemetry data", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~1064, total~2442
[Token] Slice_Type_Determination: prompt~2903, completion~283, total~3186
[Token] Beamforming_Bandwidth: prompt~218, completion~1020, total~1238
[Token] Allocate_Resources: prompt~3340, completion~1184, total~4524

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-15 11:08:56
Total Users: 5
Average Resource Utilization: 3.69%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 24.02 Mbps, mMTC Total Rate: 17.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          3  3.0/30 MHz        10.00%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 5 → mMTC Slice
CQI: 3, Bandwidth: 0.9 MHz, Rate: 4.29 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4782, completion~1137, total~5919

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I want to play competitive mobile games with ultra-low latency"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to play competitive mobile games with ultra-low latency", "urllc"),)
[Token] Intent_Analysis: prompt~1376, completion~973, total~2349
[Token] Slice_Type_Determination: prompt~2836, completion~263, total~3099
[Token] Beamforming_Bandwidth: prompt~218, completion~386, total~604
[Token] Allocate_Resources: prompt~3226, completion~840, total~4066

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-15 11:09:54
Total Users: 6
Average Resource Utilization: 6.77%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 45.84 Mbps, mMTC Total Rate: 17.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          4  7.0/30 MHz        23.33%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 6 → URLLC Slice
CQI: 4, Bandwidth: 4.0 MHz, Rate: 21.82 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4326, completion~583, total~4909

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to participate in an online multiplayer game"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to participate in an online multiplayer game", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~1057, total~2429
[Token] Slice_Type_Determination: prompt~2909, completion~679, total~3588
[Token] Beamforming_Bandwidth: prompt~216, completion~438, total~654
[Token] Allocate_Resources: prompt~3729, completion~820, total~4549

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-15 11:12:02
Total Users: 7
Average Resource Utilization: 7.54%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 60.01 Mbps, mMTC Total Rate: 17.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          5  8.0/30 MHz        26.67%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 14, Bandwidth: 1.0 MHz, Rate: 14.17 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4812, completion~777, total~5589

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I want to browse websites and check email"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to browse websites and check email", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~799, total~2169
[Token] Slice_Type_Determination: prompt~2632, completion~328, total~2960
[Token] Beamforming_Bandwidth: prompt~213, completion~1395, total~1608
[Token] Allocate_Resources: prompt~3106, completion~942, total~4048

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-15 11:13:04
Total Users: 8
Average Resource Utilization: 22.92%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 60.01 Mbps, mMTC Total Rate: 17.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          5  8.0/30 MHz        26.67%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 8 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4307, completion~1006, total~5313

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to monitor my home security cameras remotely", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~987, total~2359
[Token] Slice_Type_Determination: prompt~2842, completion~439, total~3281
[Token] Beamforming_Bandwidth: prompt~216, completion~443, total~659
[Token] Allocate_Resources: prompt~3410, completion~796, total~4206

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-15 11:15:02
Total Users: 9
Average Resource Utilization: 26.77%
eMBB Total Rate: 109.11 Mbps, URLLC Total Rate: 94.88 Mbps, mMTC Total Rate: 17.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  20.0/90 MHz       22.22%
URLLC          6  13.0/30 MHz       43.33%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 9 → URLLC Slice
CQI: 6, Bandwidth: 5.0 MHz, Rate: 34.87 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4472, completion~961, total~5433

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to make a high-quality voice call"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to make a high-quality voice call", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~1008, total~2380
[Token] Slice_Type_Determination: prompt~2849, completion~565, total~3414
LLM recommended URLLC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~3491, completion~432, total~3923
[Token] Beamforming_Bandwidth: prompt~216, completion~492, total~708
[Token] Allocate_Resources: prompt~4072, completion~758, total~4830

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-15 11:16:10
Total Users: 10
Average Resource Utilization: 42.15%
eMBB Total Rate: 218.22 Mbps, URLLC Total Rate: 94.88 Mbps, mMTC Total Rate: 17.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  40.0/90 MHz       44.44%
URLLC          6  13.0/30 MHz       43.33%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 10 → eMBB Slice
CQI: 4, Bandwidth: 20.0 MHz, Rate: 109.11 Mbps, Latency: 30.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~5141, completion~610, total~5751

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to send text messages and use messaging apps"
CQI: 3
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to send text messages and use messaging apps", "embb"),)
[Token] Intent_Analysis: prompt~1374, completion~1074, total~2448
[Token] Slice_Type_Determination: prompt~2909, completion~665, total~3574
[Token] Beamforming_Bandwidth: prompt~217, completion~399, total~616
[Token] Allocate_Resources: prompt~3719, completion~872, total~4591

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-15 11:17:10
Total Users: 11
Average Resource Utilization: 57.54%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 94.88 Mbps, mMTC Total Rate: 17.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          6  13.0/30 MHz       43.33%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 11 → eMBB Slice
CQI: 3, Bandwidth: 20.0 MHz, Rate: 95.29 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
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
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4849, completion~1010, total~5859

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I want to play competitive mobile games with ultra-low latency"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i want to play competitive mobile games with ultra-low latency", "urllc"),)
[Token] Intent_Analysis: prompt~1376, completion~1031, total~2407
[Token] Slice_Type_Determination: prompt~2896, completion~333, total~3229
[Token] Beamforming_Bandwidth: prompt~218, completion~247, total~465
[Token] Allocate_Resources: prompt~3357, completion~744, total~4101

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-15 11:18:05
Total Users: 12
Average Resource Utilization: 59.85%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 109.17 Mbps, mMTC Total Rate: 17.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          7  16.0/30 MHz       53.33%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 3, Bandwidth: 3.0 MHz, Rate: 14.29 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
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
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4368, completion~713, total~5081

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need real-time traffic updates for navigation"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need real-time traffic updates for navigation", "urllc"),)
[Token] Intent_Analysis: prompt~1370, completion~1028, total~2398
[Token] Slice_Type_Determination: prompt~2875, completion~364, total~3239
[Token] Beamforming_Bandwidth: prompt~215, completion~365, total~580
[Token] Allocate_Resources: prompt~3364, completion~838, total~4202

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-15 11:18:58
Total Users: 13
Average Resource Utilization: 60.62%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 118.68 Mbps, mMTC Total Rate: 17.91 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          8  17.0/30 MHz       56.67%
mMTC           2  1.8/10 MHz        18.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
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
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4472, completion~808, total~5280

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my wearable device needs to upload health data periodically", "mmtc"),, ("my wearable device needs to upload health data periodically", "mmtc"),)
[Token] Intent_Analysis: prompt~1372, completion~1143, total~2515
[Token] Slice_Type_Determination: prompt~2976, completion~252, total~3228
[Token] Beamforming_Bandwidth: prompt~215, completion~792, total~1007
[Token] Allocate_Resources: prompt~3386, completion~1279, total~4665

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-15 11:20:10
Total Users: 14
Average Resource Utilization: 62.85%
eMBB Total Rate: 313.51 Mbps, URLLC Total Rate: 118.68 Mbps, mMTC Total Rate: 38.13 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  60.0/90 MHz       66.67%
URLLC          8  17.0/30 MHz       56.67%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 14 → mMTC Slice
CQI: 6, Bandwidth: 2.9 MHz, Rate: 20.22 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        2.9 |         20.22 |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4926, completion~767, total~5693

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 13
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i need to use cloud-based ai services for image processing", "embb"),)
[Token] Intent_Analysis: prompt~1376, completion~855, total~2231
[Token] Slice_Type_Determination: prompt~2692, completion~273, total~2965
[Token] Beamforming_Bandwidth: prompt~218, completion~465, total~683
[Token] Allocate_Resources: prompt~3109, completion~773, total~3882

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-15 11:20:59
Total Users: 15
Average Resource Utilization: 78.23%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 118.68 Mbps, mMTC Total Rate: 38.13 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          8  17.0/30 MHz       56.67%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 13, Bandwidth: 20.0 MHz, Rate: 264.25 Mbps, Latency: 40.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
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
|        14 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4130, completion~714, total~4844

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need instant facial recognition for public security threats"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need instant facial recognition for public security threats", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~862, total~2234
[Token] Slice_Type_Determination: prompt~2713, completion~370, total~3083
[Token] Beamforming_Bandwidth: prompt~216, completion~879, total~1095
[Token] Allocate_Resources: prompt~3214, completion~821, total~4035

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-15 11:22:35
Total Users: 16
Average Resource Utilization: 79.0%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 128.19 Mbps, mMTC Total Rate: 38.13 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC          9  18.0/30 MHz       60.00%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 9.51 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4294, completion~697, total~4991

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to monitor and control critical manufacturing processes in real-time"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to monitor and control critical manufacturing processes in real-time", "urllc"),)
[Token] Intent_Analysis: prompt~1378, completion~816, total~2194
[Token] Slice_Type_Determination: prompt~2664, completion~365, total~3029
[Token] Beamforming_Bandwidth: prompt~219, completion~506, total~725
[Token] Allocate_Resources: prompt~3154, completion~885, total~4039

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-15 11:23:29
Total Users: 17
Average Resource Utilization: 79.77%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 143.33 Mbps, mMTC Total Rate: 38.13 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  80.0/90 MHz       88.89%
URLLC         10  19.0/30 MHz       63.33%
mMTC           3  4.7/10 MHz        47.00%

New User Allocation:
User 17 → URLLC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 15.14 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4304, completion~908, total~5212

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need to check the status of my smart home sensors"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("i need to check the status of my smart home sensors", "mmtc"),, ("i need to check the status of my smart home sensors", "mmtc"),)
[Token] Intent_Analysis: prompt~1376, completion~1059, total~2435
[Token] Slice_Type_Determination: prompt~2891, completion~405, total~3296
[Token] Beamforming_Bandwidth: prompt~217, completion~495, total~712
[Token] Allocate_Resources: prompt~3451, completion~943, total~4394

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-15 11:24:29
Total Users: 18
Average Resource Utilization: 80.46%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 143.33 Mbps, mMTC Total Rate: 42.42 Mbps

Slice      Users  Resource Usage             Utilization
-------  -------  -------------------------  -------------
eMBB           4  80.0/90 MHz                88.89%
URLLC         10  19.0/30 MHz                63.33%
mMTC           4  5.6000000000000005/10 MHz  56.00%

New User Allocation:
User 18 → mMTC Slice
CQI: 3, Bandwidth: 0.9 MHz, Rate: 4.29 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4644, completion~653, total~5297

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 2
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("industrial equipment monitoring sensor data", "mmtc"))
[Token] Intent_Analysis: prompt~1364, completion~1033, total~2397
[Token] Slice_Type_Determination: prompt~2857, completion~322, total~3179
[Token] Beamforming_Bandwidth: prompt~216, completion~392, total~608
[Token] Allocate_Resources: prompt~3337, completion~910, total~4247

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-15 11:25:48
Total Users: 19
Average Resource Utilization: 81.15%
eMBB Total Rate: 577.76 Mbps, URLLC Total Rate: 143.33 Mbps, mMTC Total Rate: 46.13 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           4  80.0/90 MHz               88.89%
URLLC         10  19.0/30 MHz               63.33%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 19 → mMTC Slice
CQI: 2, Bandwidth: 0.9 MHz, Rate: 3.71 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        10 | eMBB    |     4 |       20   |        109.11 |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | eMBB    |     3 |       20   |         95.29 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |       20   |        264.25 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | eMBB    |     4 |       20   |        109.11 |             40 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4492, completion~890, total~5382

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I want to download a big game file"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to download a big game file", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~946, total~2316
[Token] Slice_Type_Determination: prompt~2782, completion~243, total~3025
[Token] Beamforming_Bandwidth: prompt~215, completion~304, total~519
[Token] Allocate_Resources: prompt~3146, completion~720, total~3866

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-15 11:26:31
Total Users: 20
Average Resource Utilization: 88.85%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 143.33 Mbps, mMTC Total Rate: 46.13 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         10  19.0/30 MHz               63.33%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 11, Bandwidth: 10.0 MHz, Rate: 113.32 Mbps, Latency: 80.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
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
|        14 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4123, completion~966, total~5089

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to update my social media status"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to update my social media status", "embb"),)
[Token] Intent_Analysis: prompt~1370, completion~949, total~2319
[Token] Slice_Type_Determination: prompt~2785, completion~637, total~3422
[Token] Beamforming_Bandwidth: prompt~215, completion~573, total~788
[Token] Allocate_Resources: prompt~3617, completion~847, total~4464
[Token] Failure_Evaluation: prompt~4570, completion~527, total~5097

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
[Token] Intent_Analysis: prompt~1372, completion~944, total~2316
[Token] Slice_Type_Determination: prompt~2796, completion~301, total~3097
[Token] Beamforming_Bandwidth: prompt~216, completion~351, total~567
[Token] Allocate_Resources: prompt~3227, completion~819, total~4046

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-15 11:28:58
Total Users: 21
Average Resource Utilization: 90.38%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 151.58 Mbps, mMTC Total Rate: 46.13 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         11  21.0/30 MHz               70.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 22 → URLLC Slice
CQI: 2, Bandwidth: 2.0 MHz, Rate: 8.25 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        2   |          8.25 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
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
|        14 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4315, completion~800, total~5115

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I want to stream a webinar with interactive features"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to stream a webinar with interactive features", "embb"),)
[Token] Intent_Analysis: prompt~1372, completion~1034, total~2406
[Token] Slice_Type_Determination: prompt~2881, completion~320, total~3201
[Token] Beamforming_Bandwidth: prompt~216, completion~472, total~688
[Token] Allocate_Resources: prompt~3397, completion~878, total~4275
[Token] Failure_Evaluation: prompt~4374, completion~639, total~5013

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
[Token] Intent_Analysis: prompt~1376, completion~1041, total~2417
[Token] Slice_Type_Determination: prompt~2881, completion~284, total~3165
[Token] Beamforming_Bandwidth: prompt~218, completion~482, total~700
[Token] Allocate_Resources: prompt~3358, completion~850, total~4208
[Token] Failure_Evaluation: prompt~4306, completion~563, total~4869

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
[Token] Intent_Analysis: prompt~1372, completion~780, total~2152
[Token] Slice_Type_Determination: prompt~2617, completion~298, total~2915
[Token] Beamforming_Bandwidth: prompt~216, completion~502, total~718
[Token] Allocate_Resources: prompt~3035, completion~796, total~3831

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-15 11:32:06
Total Users: 22
Average Resource Utilization: 91.15%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 165.75 Mbps, mMTC Total Rate: 46.13 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         12  22.0/30 MHz               73.33%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 25 → URLLC Slice
CQI: 14, Bandwidth: 1.0 MHz, Rate: 14.17 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        2   |          8.25 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
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
|        14 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4097, completion~986, total~5083

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to control a robotic arm in real time"
CQI: 2
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need to control a robotic arm in real time", "urllc"),)
[Token] Intent_Analysis: prompt~1374, completion~812, total~2186
[Token] Slice_Type_Determination: prompt~2659, completion~199, total~2858
[Token] Beamforming_Bandwidth: prompt~217, completion~365, total~582
[Token] Allocate_Resources: prompt~2981, completion~874, total~3855

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-15 11:33:28
Total Users: 23
Average Resource Utilization: 92.69%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 174.00 Mbps, mMTC Total Rate: 46.13 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         13  24.0/30 MHz               80.00%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 2, Bandwidth: 2.0 MHz, Rate: 8.25 Mbps, Latency: 3.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        2   |          8.25 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     2 |        2   |          8.25 |              3 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
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
|        14 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4125, completion~770, total~4895

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need emergency response coordination during a disaste"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: URLLC (Knowledge base match: ("i need emergency response coordination during a disaster", "urllc"),)
[Token] Intent_Analysis: prompt~1372, completion~763, total~2135
[Token] Slice_Type_Determination: prompt~2610, completion~380, total~2990
[Token] Beamforming_Bandwidth: prompt~216, completion~461, total~677
[Token] Allocate_Resources: prompt~3115, completion~931, total~4046

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-15 11:34:17
Total Users: 24
Average Resource Utilization: 93.46%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 188.17 Mbps, mMTC Total Rate: 46.13 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         14  25.0/30 MHz               83.33%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 27 → URLLC Slice
CQI: 14, Bandwidth: 1.0 MHz, Rate: 14.17 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        2   |          8.25 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     2 |        2   |          8.25 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |    14 |        1   |         14.17 |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
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
|        14 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4322, completion~846, total~5168

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I want to use augmented reality navigation"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: eMBB (Knowledge base match: ("i want to use augmented reality navigation", "embb"),)
[Token] Intent_Analysis: prompt~1368, completion~999, total~2367
[Token] Slice_Type_Determination: prompt~2847, completion~284, total~3131
LLM recommended URLLC but knowledge base recommended eMBB, using knowledge base recommendation
[Token] Intent_Override: prompt~3206, completion~371, total~3577
[Token] Beamforming_Bandwidth: prompt~214, completion~475, total~689
[Token] Allocate_Resources: prompt~3777, completion~796, total~4573
[Token] Failure_Evaluation: prompt~4674, completion~537, total~5211

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
[Token] Intent_Analysis: prompt~1372, completion~826, total~2198
[Token] Slice_Type_Determination: prompt~2691, completion~230, total~2921
[Token] Beamforming_Bandwidth: prompt~216, completion~1087, total~1303
[Token] Allocate_Resources: prompt~3052, completion~840, total~3892

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-15 11:36:41
Total Users: 25
Average Resource Utilization: 95.77%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 202.46 Mbps, mMTC Total Rate: 46.13 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         15  28.0/30 MHz               93.33%
mMTC           5  6.500000000000001/10 MHz  65.00%

New User Allocation:
User 29 → URLLC Slice
CQI: 3, Bandwidth: 3.0 MHz, Rate: 14.29 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        2   |          8.25 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     2 |        2   |          8.25 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | URLLC   |     3 |        3   |         14.29 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
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
|        14 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4163, completion~693, total~4856

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 1
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
Knowledge Base recommended slice: mMTC (Knowledge base match: ("my smart parking sensor needs to report if the spot is free", "mmtc"),)
[Token] Intent_Analysis: prompt~1378, completion~1009, total~2387
[Token] Slice_Type_Determination: prompt~2844, completion~305, total~3149
[Token] Beamforming_Bandwidth: prompt~222, completion~502, total~724
[Token] Allocate_Resources: prompt~3306, completion~1088, total~4394

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-15 11:37:29
Total Users: 26
Average Resource Utilization: 96.46%
eMBB Total Rate: 691.08 Mbps, URLLC Total Rate: 202.46 Mbps, mMTC Total Rate: 49.32 Mbps

Slice      Users  Resource Usage            Utilization
-------  -------  ------------------------  -------------
eMBB           5  90.0/90 MHz               100.00%
URLLC         15  28.0/30 MHz               93.33%
mMTC           6  7.400000000000001/10 MHz  74.00%

New User Allocation:
User 30 → mMTC Slice
CQI: 1, Bandwidth: 0.9 MHz, Rate: 3.19 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | URLLC   |     2 |        1   |          4.12 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |        1   |          9.51 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | URLLC   |    15 |        1   |         15.14 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | URLLC   |     3 |        1   |          4.76 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |     2 |        2   |          8.25 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     2 |        2   |          8.25 |              3 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |    14 |        1   |         14.17 |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | URLLC   |     3 |        3   |         14.29 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |        1   |         15.14 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |        4   |         21.82 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |        1   |         14.17 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | URLLC   |     6 |        5   |         34.87 |              5 |          |
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
|        14 | mMTC    |     6 |        2.9 |         20.22 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |        0.9 |          4.29 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |        0.9 |          3.71 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |        0.9 |         13.62 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | mMTC    |     1 |        0.9 |          3.19 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | mMTC    |     3 |        0.9 |          4.29 |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
[Token] Network_Evaluation: prompt~4657, completion~843, total~5500

Detailed Slice Utilization Values:
eMBB utils: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.22, 22.22, 44.44, 66.67, 66.67, 66.67, 66.67, 88.89, 88.89, 88.89, 88.89, 88.89, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
URLLC utils: [3.33, 6.67, 6.67, 10.0, 10.0, 23.33, 26.67, 26.67, 43.33, 43.33, 43.33, 53.33, 56.67, 56.67, 56.67, 60.0, 63.33, 63.33, 63.33, 63.33, 70.0, 73.33, 80.0, 83.33, 93.33, 93.33]
mMTC utils: [0.0, 0.0, 9.0, 9.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 18.0, 47.0, 47.0, 47.0, 47.0, 56.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 65.0, 74.0]

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | URLLC   | URLLC          | Yes            |     2 |        1   |          4.12 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | URLLC   | URLLC          | Yes            |     3 |        1   |          4.76 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | mMTC    | mMTC           | Yes            |    15 |        0.9 |         13.62 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | URLLC   | URLLC          | Yes            |    15 |        1   |         15.14 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | mMTC    | mMTC           | Yes            |     3 |        0.9 |          4.29 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | URLLC   | URLLC          | Yes            |     4 |        4   |         21.82 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |    14 |        1   |         14.17 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | URLLC   | eMBB           | No             |     6 |        5   |         34.87 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             30 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | eMBB    | eMBB           | Yes            |     3 |       20   |         95.29 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     3 |        3   |         14.29 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | eMBB           | No             |     9 |        1   |          9.51 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | mMTC    | mMTC           | Yes            |     6 |        2.9 |         20.22 |           1000 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |    13 |       20   |        264.25 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |     9 |        1   |          9.51 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | URLLC   | URLLC          | Yes            |    15 |        1   |         15.14 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | mMTC    | mMTC           | Yes            |     3 |        0.9 |          4.29 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | mMTC    | mMTC           | Yes            |     2 |        0.9 |          3.71 |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    11 |       10   |        113.32 |             80 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Failed   | eMBB    | eMBB           | Yes            |     6 |       20   |        139.46 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | URLLC   | URLLC          | Yes            |     2 |        2   |          8.25 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Failed   | eMBB    | eMBB           | Yes            |    12 |       20   |        245.31 |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Failed   | eMBB    | eMBB           | Yes            |     4 |       20   |        109.11 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | URLLC   | URLLC          | Yes            |    14 |        1   |         14.17 |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | URLLC          | Yes            |     2 |        2   |          8.25 |              3 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | URLLC   | URLLC          | Yes            |    14 |        1   |         14.17 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Failed   | eMBB    | eMBB           | Yes            |     6 |       20   |        139.46 |             40 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | URLLC   | URLLC          | Yes            |     3 |        3   |         14.29 |              5 | No         |
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
Average URLLC utilization: 48.97%
Average mMTC utilization: 36.65%

Weighted Average Utilization: 54.06%

Transmission Rate Statistics:
Final eMBB total rate: 691.08 Mbps
Final URLLC total rate: 202.46 Mbps
Final mMTC total rate: 49.32 Mbps

Resource Utilization:
Average resource utilization: 96.46%

Results exported to F:\code\wirelessagent\run_results\with_fkb\network_slicing_results_TJU_south_minimax-M2.5.csv
(.venv) PS F:\code\wirelessagent> 