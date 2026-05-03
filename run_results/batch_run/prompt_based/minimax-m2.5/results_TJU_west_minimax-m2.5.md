============================================================
场景 4/5: TJU_west
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_west.csv
输出: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_west_minimax-m2.5.csv
------------------------------------------------------------
Starting prompt-based network slice management system...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I need to download large files"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~344, completion~1259, total~1603

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 15.0, rate: 4.0

Intent Analysis: Large file download - bandwidth-intensive, high-throughput application
Recommended Slice: eMBB - Large file downloads require high data rates (100-400 Mbps) which eMBB slice provides. URLLC is overkill with strict latency requirements. mMTC supports only up to 1 Mbps, insufficient for large files.
Bandwidth Allocation: 15.0 MHz
Data Rate: 4.0 Mbps
Latency: 50.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-04-05 23:55:25
Total Users: 1
Average Resource Utilization: 11.54%
eMBB Total Rate: 4.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 1 → eMBB Slice
CQI: 4, Bandwidth: 15.0 MHz, Rate: 4.00 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |     4 |         15 |             4 |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~350, completion~3011, total~3361

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.877

Intent Analysis: Environmental sensor network reporting air quality
Recommended Slice: mMTC - Air‑quality sensor data are low‑rate and can tolerate higher latency, making the massive Machine‑Type Communications (mMTC) slice the most suitable. The allocated 1 MHz bandwidth yields a data rate of ≈0.88 Mbps, which falls within the mMTC rate range (0.1‑1 Mbps) and satisfies its latency window (100‑1000 ms).
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.877 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-04-05 23:56:11
Total Users: 2
Average Resource Utilization: 12.31%
eMBB Total Rate: 4.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.88 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          0  0/30 MHz          0%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 2 → mMTC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 0.88 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~1318, total~1672

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 7.0

Intent Analysis: N/A
Recommended Slice: URLLC - CNC machine control with 'zero tolerance for delay' requires ultra-reliable low-latency communication (URLLC). The application demands strict latency constraints (1-10ms) that cannot be met by eMBB or mMTC slices.
Bandwidth Allocation: 3.0 MHz
Data Rate: 7.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-04-05 23:56:33
Total Users: 3
Average Resource Utilization: 14.62%
eMBB Total Rate: 4.00 Mbps, URLLC Total Rate: 7.00 Mbps, mMTC Total Rate: 0.88 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          1  3.0/30 MHz        10.00%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 3 → URLLC Slice
CQI: 7, Bandwidth: 3.0 MHz, Rate: 7.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |          3 |          7    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to send text messages and use messaging apps"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~1172, total~1526

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: Text messaging and messaging apps - low-bandwidth, reliability-focused communication
Recommended Slice: mMTC - Massive Machine-Type Communications (mMTC) is optimized for low-bandwidth IoT and messaging applications with high device density. Text messages and messaging apps generate small, intermittent data packets that align perfectly with mMTC characteristics. The slice's 1-3 MHz bandwidth allocation and 0.1-1 Mbps rate range are sufficient for this use case while maintaining cost efficiency.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 150.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-04-05 23:56:50
Total Users: 4
Average Resource Utilization: 15.38%
eMBB Total Rate: 4.00 Mbps, URLLC Total Rate: 7.00 Mbps, mMTC Total Rate: 1.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          1  3.0/30 MHz        10.00%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 4 → mMTC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 150.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I need to use maps for basic navigation"
CQI: 3
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~2462, total~2814

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 8.0, rate: 3.0

Intent Analysis: N/A
Recommended Slice: eMBB - Maps and navigation are typical mobile broadband applications requiring consistent connectivity and moderate data rates. The eMBB slice provides appropriate bandwidth range and latency characteristics for this use case.
Bandwidth Allocation: 8.0 MHz
Data Rate: 3.0 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-04-05 23:57:40
Total Users: 5
Average Resource Utilization: 21.54%
eMBB Total Rate: 7.00 Mbps, URLLC Total Rate: 7.00 Mbps, mMTC Total Rate: 1.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  23.0/90 MHz       25.56%
URLLC          1  3.0/30 MHz        10.00%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 5 → eMBB Slice
CQI: 3, Bandwidth: 8.0 MHz, Rate: 3.00 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~1751, total~2105

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.0

Intent Analysis: N/A
Recommended Slice: mMTC - The request consists of periodic, small‑payload sensor data, which is characteristic of massive Machine‑Type Communications (mMTC). mMTC is designed for low‑rate, low‑bandwidth IoT traffic and can tolerate latencies in the hundreds of milliseconds, matching the 100‑1000 ms window. The user’s CQI of 6 indicates a moderate radio channel, suitable for a modest spectral efficiency (≈1.5 bits/Hz). Allocating the minimum allowed bandwidth of 1 MHz yields a raw capacity of ~1.5 Mbps; however, the slice’s maximum user rate is 1 Mbps, so the effective allocated rate is capped at 1 Mbps to stay within slice constraints. This allocation leaves ample room for additional mMTC devices while keeping the slice utilization at a safe level.
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.0 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-04-05 23:58:13
Total Users: 6
Average Resource Utilization: 22.31%
eMBB Total Rate: 7.00 Mbps, URLLC Total Rate: 7.00 Mbps, mMTC Total Rate: 2.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  23.0/90 MHz       25.56%
URLLC          1  3.0/30 MHz        10.00%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 6 → mMTC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 1.00 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need reliable connectivity for implanted medical devices"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~2422, total~2774

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 3.53

Intent Analysis: N/A
Recommended Slice: URLLC - Implanted medical devices require ultra‑reliable, low‑latency communication to ensure timely reporting of vital signs and immediate response to alerts.; URLLC slice provides the necessary latency range (1‑10 ms) and reliability mechanisms (e.g., HARQ, priority handling).; CQI 6 corresponds to a moderate SNR (~2 dB) and a spectral efficiency of ~1.18 bits/Hz, which is suitable for a modest data‑rate telemetry flow.; Allocating 3 MHz stays within the URLLC bandwidth limits (1‑5 MHz) and yields a realistic data rate of ~3.5 Mbps, well above the minimum (1 Mbps) while preserving reliability.; The allocation increases URLLC utilization from 10 % (3/30 MHz) to 20 % (6/30 MHz), leaving ample headroom (24 MHz free) for future users.
Bandwidth Allocation: 3.0 MHz
Data Rate: 3.53 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-04-05 23:58:54
Total Users: 7
Average Resource Utilization: 24.62%
eMBB Total Rate: 7.00 Mbps, URLLC Total Rate: 10.53 Mbps, mMTC Total Rate: 2.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  23.0/90 MHz       25.56%
URLLC          2  6.0/30 MHz        20.00%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 6, Bandwidth: 3.0 MHz, Rate: 3.53 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I want to remotely access my work computer"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~2377, total~2729

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 6.0

Intent Analysis: User wants to remotely access a work computer. Remote desktop/terminal sessions typically require reliable connectivity with moderate bandwidth and low latency, making a low‑latency, moderate‑rate slice appropriate.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 3.0 MHz
Data Rate: 6.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-04-05 23:59:35
Total Users: 8
Average Resource Utilization: 26.92%
eMBB Total Rate: 7.00 Mbps, URLLC Total Rate: 16.53 Mbps, mMTC Total Rate: 2.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  23.0/90 MHz       25.56%
URLLC          3  9.0/30 MHz        30.00%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 4, Bandwidth: 3.0 MHz, Rate: 6.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to use holographic communication"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~1383, total~1734

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 200.0

Intent Analysis: N/A
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 10.0 MHz
Data Rate: 200.0 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-04-05 23:59:56
Total Users: 9
Average Resource Utilization: 34.62%
eMBB Total Rate: 207.00 Mbps, URLLC Total Rate: 16.53 Mbps, mMTC Total Rate: 2.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  33.0/90 MHz       36.67%
URLLC          3  9.0/30 MHz        30.00%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 9 → eMBB Slice
CQI: 15, Bandwidth: 10.0 MHz, Rate: 200.00 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to control precision CNC machines with zero tolerance for delay"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~356, completion~1830, total~2186

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 26.5

Intent Analysis: Ultra‑reliable, extremely low‑latency communication for closed‑loop CNC machine control.
Recommended Slice: URLLC - Zero‑tolerance delay and high channel quality dictate URLLC. 5 MHz provides enough bandwidth for reliable low‑latency operation while keeping overall slice utilization balanced.
Bandwidth Allocation: 5.0 MHz
Data Rate: 26.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-04-06 00:00:27
Total Users: 10
Average Resource Utilization: 38.46%
eMBB Total Rate: 207.00 Mbps, URLLC Total Rate: 43.03 Mbps, mMTC Total Rate: 2.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  33.0/90 MHz       36.67%
URLLC          4  14.0/30 MHz       46.67%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 10 → URLLC Slice
CQI: 15, Bandwidth: 5.0 MHz, Rate: 26.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need emergency response coordination during a disaste"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~1520, total~1873

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 10.0

Intent Analysis: The user requests emergency response coordination during a disaster. This scenario demands ultra‑reliable, low‑latency communication to support real‑time command and control, data exchange, and situational awareness.
Recommended Slice: URLLC - URLLC provides the required 1‑10 ms latency and supports bandwidth of 1‑5 MHz, matching the emergency nature of the request. The user’s CQI is low (3), indicating a weak channel; allocating the maximum 5 MHz bandwidth improves robustness and yields an achievable data rate of roughly 10 Mbps, sufficient for critical coordination traffic. The URLLC slice currently uses 14 MHz of its 30 MHz capacity, leaving 16 MHz free, so the 5 MHz allocation is feasible and preserves balance across slices.
Bandwidth Allocation: 5.0 MHz
Data Rate: 10.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-04-06 00:00:56
Total Users: 11
Average Resource Utilization: 42.31%
eMBB Total Rate: 207.00 Mbps, URLLC Total Rate: 53.03 Mbps, mMTC Total Rate: 2.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  33.0/90 MHz       36.67%
URLLC          5  19.0/30 MHz       63.33%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 3, Bandwidth: 5.0 MHz, Rate: 10.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I want to use remote surgery equipment"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~351, completion~130, total~481
Error parsing LLM response: No JSON found in response

----------------------------------------
ALLOCATION FAILED FOR USER 12
----------------------------------------
Request: I want to use remote surgery equipment
Slice type: Failed
Reason: Unknown error

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need to participate in an online multiplayer game"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2499, total~2852

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 35.0

Intent Analysis: User requires low latency and moderate bandwidth for real‑time gaming; CQI indicates decent channel quality.
Recommended Slice: eMBB - URLLC offers latency of 1‑10 ms (suitable for gaming) with bandwidth of 1‑5 MHz and data rates up to 100 Mbps, meeting the user's needs.
Bandwidth Allocation: 5.0 MHz
Data Rate: 35.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-04-06 00:01:40
Total Users: 12
Average Resource Utilization: 46.15%
eMBB Total Rate: 242.00 Mbps, URLLC Total Rate: 53.03 Mbps, mMTC Total Rate: 2.38 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  38.0/90 MHz       42.22%
URLLC          5  19.0/30 MHz       63.33%
mMTC           3  3.0/10 MHz        30.00%

New User Allocation:
User 13 → eMBB Slice
CQI: 7, Bandwidth: 5.0 MHz, Rate: 35.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need to track the location of a shipping container"
CQI: 14
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~354, completion~2351, total~2705

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: N/A
Recommended Slice: mMTC - IoT-style periodic location updates with small data packets; Moderate latency tolerance (tracking updates typically seconds apart); Low-to-medium reliability requirements; Energy-efficient communication suitable for tracking devices; Designed for massive machine-type communications
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 150.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-04-06 00:02:18
Total Users: 13
Average Resource Utilization: 46.92%
eMBB Total Rate: 242.00 Mbps, URLLC Total Rate: 53.03 Mbps, mMTC Total Rate: 2.88 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  38.0/90 MHz       42.22%
URLLC          5  19.0/30 MHz       63.33%
mMTC           4  4.0/10 MHz        40.00%

New User Allocation:
User 14 → mMTC Slice
CQI: 14, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 150.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 15
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~355, completion~1094, total~1449

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 186.0

Intent Analysis: N/A
Recommended Slice: eMBB - User intent requires high bandwidth for image processing, matching eMBB capabilities.; CQI = 15 indicates excellent channel conditions, allowing a moderate bandwidth allocation.; 10 MHz allocation satisfies the eMBB bandwidth range (6‑20 MHz) and yields an estimated rate of 186 Mbps, well within the allowed 100‑400 Mbps.; Latency tolerance of 10‑100 ms is acceptable for cloud‑based AI inference workloads.; The eMBB slice still has ample free resources (52 MHz before, 42 MHz after allocation), preserving headroom for existing and future users.; URLLC and mMTC slices are not under‑utilised and do not need rebalancing for this request.
Bandwidth Allocation: 10.0 MHz
Data Rate: 186.0 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-04-06 00:02:38
Total Users: 14
Average Resource Utilization: 54.62%
eMBB Total Rate: 428.00 Mbps, URLLC Total Rate: 53.03 Mbps, mMTC Total Rate: 2.88 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  48.0/90 MHz       53.33%
URLLC          5  19.0/30 MHz       63.33%
mMTC           4  4.0/10 MHz        40.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 15, Bandwidth: 10.0 MHz, Rate: 186.00 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |         10 |        186    |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need to monitor water level in a reservoir"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2367, total~2720

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 2.4

Intent Analysis: IoT‑style environmental monitoring (periodic, low‑volume data)
Recommended Slice: mMTC - Water‑level monitoring typically generates small packets at low transmission intervals.; It does not require the high throughput of eMBB nor the ultra‑low latency of URLLC.; mMTC is designed for massive machine‑type communications with bandwidth 1‑3 MHz and rate 0.1‑1 Mbps, matching the application profile.
Bandwidth Allocation: 1.0 MHz
Data Rate: 2.4 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-04-06 00:03:20
Total Users: 15
Average Resource Utilization: 55.38%
eMBB Total Rate: 428.00 Mbps, URLLC Total Rate: 53.03 Mbps, mMTC Total Rate: 5.28 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  48.0/90 MHz       53.33%
URLLC          5  19.0/30 MHz       63.33%
mMTC           5  5.0/10 MHz        50.00%

New User Allocation:
User 16 → mMTC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 2.40 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |         10 |        186    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |          1 |          2.4  |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to monitor soil moisture levels in a large farm"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~355, completion~1813, total~2168

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.7

Intent Analysis: Agricultural IoT application requiring periodic data collection from distributed sensors across large farm area
Recommended Slice: mMTC - Soil moisture monitoring is a classic mMTC use case; Requires low bandwidth (sensor data only); Can tolerate higher latency (100-1000ms); Designed for massive IoT device connectivity; Energy efficient for battery-powered sensors; Current mMTC utilization at 50% provides adequate capacity
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.7 Mbps
Latency: 100.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-04-06 00:03:54
Total Users: 16
Average Resource Utilization: 56.15%
eMBB Total Rate: 428.00 Mbps, URLLC Total Rate: 53.03 Mbps, mMTC Total Rate: 5.98 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  48.0/90 MHz       53.33%
URLLC          5  19.0/30 MHz       63.33%
mMTC           6  6.0/10 MHz        60.00%

New User Allocation:
User 17 → mMTC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 0.70 Mbps, Latency: 100.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |         10 |        186    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |          1 |          2.4  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |          1 |          0.7  |            100 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need instant alerts for life-threatening patient conditions"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2340, total~2693

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 2.35

Intent Analysis: The request is for “instant alerts for life‑threatening patient conditions”. This implies ultra‑reliable, low‑latency communication (URLLC) with a priority on rapid notification rather than high throughput.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 2.0 MHz
Data Rate: 2.35 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-04-06 00:04:35
Total Users: 17
Average Resource Utilization: 57.69%
eMBB Total Rate: 428.00 Mbps, URLLC Total Rate: 55.38 Mbps, mMTC Total Rate: 5.98 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  48.0/90 MHz       53.33%
URLLC          6  21.0/30 MHz       70.00%
mMTC           6  6.0/10 MHz        60.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 6, Bandwidth: 2.0 MHz, Rate: 2.35 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |          2 |          2.35 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |         10 |        186    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |          1 |          2.4  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |          1 |          0.7  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to download large files"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~350, completion~2100, total~2450

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 160.0

Intent Analysis: User 19 requests a large‑file download, which requires high throughput and moderate latency. This workload aligns best with the eMBB slice, which is designed for enhanced Mobile Broadband services.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 20.0 MHz
Data Rate: 160.0 Mbps
Latency: 30.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-04-06 00:05:08
Total Users: 18
Average Resource Utilization: 73.08%
eMBB Total Rate: 588.00 Mbps, URLLC Total Rate: 55.38 Mbps, mMTC Total Rate: 5.98 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  68.0/90 MHz       75.56%
URLLC          6  21.0/30 MHz       70.00%
mMTC           6  6.0/10 MHz        60.00%

New User Allocation:
User 19 → eMBB Slice
CQI: 8, Bandwidth: 20.0 MHz, Rate: 160.00 Mbps, Latency: 30.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |          2 |          2.35 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |         10 |        186    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |         20 |        160    |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |          1 |          2.4  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |          1 |          0.7  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I need to synchronize distributed financial ledgers instantly"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~1543, total~1896

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 100.0

Intent Analysis: N/A
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 100.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-04-06 00:05:33
Total Users: 19
Average Resource Utilization: 76.92%
eMBB Total Rate: 588.00 Mbps, URLLC Total Rate: 155.38 Mbps, mMTC Total Rate: 5.98 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  68.0/90 MHz       75.56%
URLLC          7  26.0/30 MHz       86.67%
mMTC           6  6.0/10 MHz        60.00%

New User Allocation:
User 20 → URLLC Slice
CQI: 9, Bandwidth: 5.0 MHz, Rate: 100.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |          2 |          2.35 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |          5 |        100    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |         10 |        186    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |         20 |        160    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |          1 |          2.4  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |          1 |          0.7  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~352, completion~2670, total~3022

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 6.0

Intent Analysis: Mission‑critical, low‑latency communication for firefighters operating inside buildings. Requires high reliability, rapid response and moderate data throughput (voice + low‑rate video/data).
Recommended Slice: URLLC - CQI 6 yields ~2 bits/Hz; 3 MHz → ~6 Mbps, satisfying the 1‑100 Mbps URLLC rate range while keeping latency ≤10 ms.
Bandwidth Allocation: 3.0 MHz
Data Rate: 6.0 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-04-06 00:06:20
Total Users: 20
Average Resource Utilization: 79.23%
eMBB Total Rate: 588.00 Mbps, URLLC Total Rate: 161.38 Mbps, mMTC Total Rate: 5.98 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  68.0/90 MHz       75.56%
URLLC          8  29.0/30 MHz       96.67%
mMTC           6  6.0/10 MHz        60.00%

New User Allocation:
User 21 → URLLC Slice
CQI: 6, Bandwidth: 3.0 MHz, Rate: 6.00 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |          2 |          2.35 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |          5 |        100    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |          3 |          6    |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |         10 |        186    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |         20 |        160    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |          1 |          2.4  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |          1 |          0.7  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "Smart city parking sensor reporting availability"
CQI: 7
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~350, completion~3116, total~3466

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.0

Intent Analysis: Periodic small data transmission for parking spot status
Recommended Slice: mMTC - Traffic is low‑rate and periodic, aligning with mMTC use case.; mMTC slice has 4 MHz of free bandwidth, providing ample headroom.; Allocating to mMTC avoids further congestion in the heavily loaded URLLC slice (96.67% utilized) and eMBB slice (75.56%).; CQI 7 yields moderate spectral efficiency; the required rate can be met within mMTC’s 0.1‑1 Mbps limit after throttling.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.0 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-04-06 00:07:07
Total Users: 21
Average Resource Utilization: 80.0%
eMBB Total Rate: 588.00 Mbps, URLLC Total Rate: 161.38 Mbps, mMTC Total Rate: 5.98 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  68.0/90 MHz       75.56%
URLLC          8  29.0/30 MHz       96.67%
mMTC           7  7.0/10 MHz        70.00%

New User Allocation:
User 22 → mMTC Slice
CQI: 7, Bandwidth: 1.0 MHz, Rate: 0.00 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |          2 |          2.35 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |          5 |        100    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |          3 |          6    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |         10 |        186    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |         20 |        160    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |          1 |          2.4  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |          1 |          0.7  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |          1 |          0    |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2179, total~2532

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 68.0, rate: 0.59

Intent Analysis: MISSION_CRITICAL_CONTROL
Recommended Slice: URLLC - User request explicitly requires 'zero downtime' for critical infrastructure; URLLC slice provides 1-10ms latency requirement (mission-critical); URLLC supports ultra-reliable communications needed for infrastructure control; mMTC latency (100-1000ms) is unsuitable for real-time control; eMBB latency (10-100ms) does not meet zero-downtime SLA
Bandwidth Allocation: 68.0 MHz
Data Rate: 0.59 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 68.0 MHz, Available: 1.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 23
----------------------------------------
Request: I need to control critical infrastructure with zero downtime
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 68.0 MHz, Available: 1.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to monitor IoT sensors in real-time"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~1207, total~1560

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 5.0

Intent Analysis: Real-time IoT sensor monitoring requiring continuous data updates with minimal delay
Recommended Slice: URLLC - Real-time monitoring of IoT sensors requires low latency (1-10ms) to ensure timely data updates. The 'real-time' requirement prioritizes latency over bandwidth, making URLLC the appropriate slice despite higher utilization (96.67%).
Bandwidth Allocation: 2.0 MHz
Data Rate: 5.0 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 2.0 MHz, Available: 1.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 24
----------------------------------------
Request: I need to monitor IoT sensors in real-time
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 2.0 MHz, Available: 1.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "My smart trash can needs to signal that it's full"
CQI: 5
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~355, completion~1286, total~1641

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.6

Intent Analysis: N/A
Recommended Slice: mMTC - Smart trash can status signaling is a classic IoT use case requiring minimal bandwidth for small data payloads. mMTC slice is optimized for massive machine-type communications with power efficiency and support for many low-complexity devices.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.6 Mbps
Latency: 1000.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-04-06 00:08:25
Total Users: 22
Average Resource Utilization: 80.77%
eMBB Total Rate: 588.00 Mbps, URLLC Total Rate: 161.38 Mbps, mMTC Total Rate: 6.58 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  68.0/90 MHz       75.56%
URLLC          8  29.0/30 MHz       96.67%
mMTC           8  8.0/10 MHz        80.00%

New User Allocation:
User 25 → mMTC Slice
CQI: 5, Bandwidth: 1.0 MHz, Rate: 0.60 Mbps, Latency: 1000.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |          2 |          2.35 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |          5 |        100    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |          3 |          6    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |         10 |        186    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |         20 |        160    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |          1 |          2.4  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |          1 |          0.7  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |          1 |          0    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |          1 |          0.6  |           1000 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 7
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~2210, total~2563

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 6.0, rate: 15.0

Intent Analysis: The user wants to remotely monitor home security cameras. This is a video‑streaming service that requires a moderate‑to‑high data rate (single‑camera 1080p ≈ 2‑4 Mbps, typical home setup with 4‑8 cameras ≈ 10‑30 Mbps) and can tolerate a latency in the order of tens of milliseconds. The request does not involve ultra‑reliable low‑latency control or massive machine‑type connectivity, so it best fits the eMBB slice.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 6.0 MHz
Data Rate: 15.0 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-04-06 00:09:08
Total Users: 23
Average Resource Utilization: 85.38%
eMBB Total Rate: 603.00 Mbps, URLLC Total Rate: 161.38 Mbps, mMTC Total Rate: 6.58 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  74.0/90 MHz       82.22%
URLLC          8  29.0/30 MHz       96.67%
mMTC           8  8.0/10 MHz        80.00%

New User Allocation:
User 26 → eMBB Slice
CQI: 7, Bandwidth: 6.0 MHz, Rate: 15.00 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |          2 |          2.35 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |          5 |        100    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |          3 |          6    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |         10 |        186    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |         20 |        160    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     7 |          6 |         15    |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |          1 |          2.4  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |          1 |          0.7  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |          1 |          0    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |          1 |          0.6  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~349, completion~4037, total~4386

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: Industrial equipment monitoring sensor data – typical IoT/mMTC traffic that requires low to moderate bandwidth and can tolerate relatively high latency.
Recommended Slice: mMTC - Industrial sensor data usually consists of small, periodic packets that do not require the high data rates of eMBB.; Latency requirements are moderate (hundreds of milliseconds) and fit within the mMTC latency window (100‑1000 ms).; The mMTC slice still has spare capacity (≈20 % remaining), whereas URLLC is near‑full (≈97 % utilization) and eMBB is already heavily loaded.; Allocating the user to mMTC avoids over‑loading the ultra‑reliable slice and preserves eMBB resources for high‑speed services.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 150.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-04-06 00:10:13
Total Users: 24
Average Resource Utilization: 86.15%
eMBB Total Rate: 603.00 Mbps, URLLC Total Rate: 161.38 Mbps, mMTC Total Rate: 7.08 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  74.0/90 MHz       82.22%
URLLC          8  29.0/30 MHz       96.67%
mMTC           9  9.0/10 MHz        90.00%

New User Allocation:
User 27 → mMTC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 150.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |          2 |          2.35 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |          5 |        100    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |          3 |          6    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |         10 |        186    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |         20 |        160    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     7 |          6 |         15    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |          1 |          2.4  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |          1 |          0.7  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |          1 |          0    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |          1 |          0.6  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     4 |          1 |          0.5  |            150 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 9
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~356, completion~2381, total~2737

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.0

Intent Analysis: The user wants a low‑power IoT sensor to periodically report a binary occupancy status (spot free / occupied). The traffic is sporadic, very low‑volume, and can tolerate a latency on the order of seconds. This profile matches massive Machine‑Type Communications (mMTC) rather than high‑throughput eMBB or ultra‑reliable low‑latency URLLC services.
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.0 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 28
----------------------------------------
Network Status @ 2026-04-06 00:10:51
Total Users: 25
Average Resource Utilization: 86.92%
eMBB Total Rate: 603.00 Mbps, URLLC Total Rate: 161.38 Mbps, mMTC Total Rate: 8.08 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  74.0/90 MHz       82.22%
URLLC          8  29.0/30 MHz       96.67%
mMTC          10  10.0/10 MHz       100.00%

New User Allocation:
User 28 → mMTC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 1.00 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |          2 |          2.35 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |          5 |        100    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |          3 |          6    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |         10 |        186    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |         20 |        160    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     7 |          6 |         15    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |          1 |          2.4  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |          1 |          0.7  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |          1 |          0    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |          1 |          0.6  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     4 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     9 |          1 |          1    |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I want to stream music while browsing social media"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~353, completion~3033, total~3386

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 9.3

Intent Analysis: N/A
Recommended Slice: eMBB - The user wants to stream music and browse social media, which are typical eMBB (enhanced Mobile Broadband) services requiring moderate bandwidth and latency.; CQI = 5 indicates a lower channel quality, corresponding to QPSK modulation with a code rate ~0.46, yielding an approximate spectral efficiency of 0.93 bits/s/Hz.; Allocating 10 MHz of eMBB spectrum results in an achievable rate of ~9.3 Mbps (≈10 Mbps after rounding). This comfortably satisfies the ~1‑2 Mbps needed for music streaming and additional browsing traffic.; The eMBB slice currently uses 74 MHz of its 90 MHz total, leaving 16 MHz free; a 10 MHz grant leaves 6 MHz for other eMBB users, preserving workload balance.; The URLLC slice is 96.67 % utilized and the mMTC slice is fully occupied, so steering the user to eMBB avoids over‑stressing those latency‑critical and massive‑machine slices.; Latency for eMBB can be set to ~30 ms, well within the allowed 10‑100 ms window.
Bandwidth Allocation: 10.0 MHz
Data Rate: 9.3 Mbps
Latency: 30.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-04-06 00:11:45
Total Users: 26
Average Resource Utilization: 94.62%
eMBB Total Rate: 612.30 Mbps, URLLC Total Rate: 161.38 Mbps, mMTC Total Rate: 8.08 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  84.0/90 MHz       93.33%
URLLC          8  29.0/30 MHz       96.67%
mMTC          10  10.0/10 MHz       100.00%

New User Allocation:
User 29 → eMBB Slice
CQI: 5, Bandwidth: 10.0 MHz, Rate: 9.30 Mbps, Latency: 30.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |          2 |          2.35 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |          5 |        100    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |          3 |          6    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |         10 |        186    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |         20 |        160    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     7 |          6 |         15    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | eMBB    |     5 |         10 |          9.3  |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |          1 |          2.4  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |          1 |          0.7  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |          1 |          0    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |          1 |          0.6  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     4 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     9 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 11
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------
[Token] Prompt_Based: prompt~349, completion~3038, total~3387

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 4.5

Intent Analysis: The user needs to transmit low‑volume sensor data from industrial equipment. This traffic requires relatively low latency and moderate reliability, which aligns best with the URLLC slice characteristics rather than high‑throughput eMBB or high‑latency mMTC services.
Recommended Slice: URLLC - Only 1 MHz of bandwidth remains in the URLLC slice. Allocating the full remaining 1 MHz yields a data rate of ~4.5 Mbps, which satisfies the URLLC rate limits (1‑100 Mbps) and latency limits (1‑10 ms).
Bandwidth Allocation: 1.0 MHz
Data Rate: 4.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-04-06 00:12:28
Total Users: 27
Average Resource Utilization: 95.38%
eMBB Total Rate: 612.30 Mbps, URLLC Total Rate: 165.88 Mbps, mMTC Total Rate: 8.08 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  84.0/90 MHz       93.33%
URLLC          9  30.0/30 MHz       100.00%
mMTC          10  10.0/10 MHz       100.00%

New User Allocation:
User 30 → URLLC Slice
CQI: 11, Bandwidth: 1.0 MHz, Rate: 4.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |    15 |          5 |         26.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          5 |         10    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     6 |          2 |          2.35 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |     9 |          5 |        100    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | URLLC   |     6 |          3 |          6    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | URLLC   |     7 |          3 |          7    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | URLLC   |    11 |          1 |          4.5  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |     6 |          3 |          3.53 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          3 |          6    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | eMBB    |     4 |         15 |          4    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     7 |          5 |         35    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    15 |         10 |        186    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | eMBB    |     8 |         20 |        160    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | eMBB    |     7 |          6 |         15    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | eMBB    |     5 |         10 |          9.3  |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     3 |          8 |          3    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |    15 |         10 |        200    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |    14 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | mMTC    |     9 |          1 |          2.4  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |     4 |          1 |          0.7  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | mMTC    |     4 |          1 |          0.88 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | mMTC    |     7 |          1 |          0    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | mMTC    |     5 |          1 |          0.6  |           1000 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | mMTC    |     4 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        28 | mMTC    |     9 |          1 |          1    |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     7 |          1 |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | mMTC    |     6 |          1 |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI | BW (MHz)   | Rate (Mbps)   | Latency (ms)   | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | eMBB    | eMBB           | Yes            |     4 | 15.0       | 4.0           | 50.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | mMTC    | mMTC           | Yes            |     4 | 1.0        | 0.877         | 500.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | URLLC   | URLLC          | Yes            |     7 | 3.0        | 7.0           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | mMTC    | eMBB           | No             |     7 | 1.0        | 0.5           | 150.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | eMBB    | eMBB           | Yes            |     3 | 8.0        | 3.0           | 10.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | mMTC    | mMTC           | Yes            |     6 | 1.0        | 1.0           | 500.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |     6 | 3.0        | 3.53          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | eMBB           | No             |     4 | 3.0        | 6.0           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | eMBB    | eMBB           | Yes            |    15 | 10.0       | 200.0         | 20.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | URLLC   | URLLC          | Yes            |    15 | 5.0        | 26.5          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | URLLC          | Yes            |     3 | 5.0        | 10.0          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Failed   | Failed  | URLLC          |                |     7 | N/A        | N/A           | N/A            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | eMBB    | URLLC          | No             |     7 | 5.0        | 35.0          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | mMTC    | mMTC           | Yes            |    14 | 1.0        | 0.5           | 150.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |    15 | 10.0       | 186.0         | 10.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | mMTC    | mMTC           | Yes            |     9 | 1.0        | 2.4           | 500.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | mMTC    | mMTC           | Yes            |     4 | 1.0        | 0.7           | 100.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     6 | 2.0        | 2.35          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | eMBB    | eMBB           | Yes            |     8 | 20.0       | 160.0         | 30.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | URLLC   | URLLC          | Yes            |     9 | 5.0        | 100.0         | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | URLLC   | URLLC          | Yes            |     6 | 3.0        | 6.0           | 10.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | mMTC    | mMTC           | Yes            |     7 | 1.0        | 0.0           | 200.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Failed   | URLLC   | URLLC          |                |     4 | 68.0       | 0.59          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Failed   | URLLC   | URLLC          |                |     4 | 2.0        | 5.0           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | mMTC    | mMTC           | Yes            |     5 | 1.0        | 0.6           | 1000.0         | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | eMBB    | eMBB           | Yes            |     7 | 6.0        | 15.0          | 10.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | mMTC    | mMTC           | Yes            |     4 | 1.0        | 0.5           | 150.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Success  | mMTC    | mMTC           | Yes            |     9 | 1.0        | 1.0           | 200.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | eMBB    | eMBB           | Yes            |     5 | 10.0       | 9.3           | 30.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Success  | URLLC   | mMTC           | No             |    11 | 1.0        | 4.5           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 27/30 (90.0%)

Intent Understanding Evaluation:
Correctly identified intents: 23/27
Intent understanding rate: 85.2%

Workload Balancing Statistics:
Users with workload balancing: 27/30
Workload balancing rate: 90.0%

Slice Utilization Statistics:
Average eMBB utilization: 51.40%
Average URLLC utilization: 57.41%
Average mMTC utilization: 48.89%

Results exported to F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_west_minimax-m2.5.csv

[OK] TJU_west 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_west_minimax-m2.5.csv