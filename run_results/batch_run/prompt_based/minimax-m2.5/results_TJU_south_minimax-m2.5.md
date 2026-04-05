============================================================
场景 2/5: TJU_south
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_south.csv
输出: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_south_minimax-m2.5.csv
------------------------------------------------------------
Starting prompt-based network slice management system...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 2
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------
Error parsing LLM response: Expecting ',' delimiter: line 30 column 78 (char 1923)

----------------------------------------
ALLOCATION FAILED FOR USER 1
----------------------------------------
Request: I need immediate machine shutdown capability for safety incidents
Slice type: Failed
Reason: Unknown error

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I need my autonomous vehicle to communicate in real time"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 1.9

Intent Analysis: Real‑time, low‑latency communication for autonomous vehicle control and sensor data streaming.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 1.9 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-30 11:13:44
Total Users: 1
Average Resource Utilization: 3.85%
eMBB Total Rate: 1.90 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  5.0/90 MHz        5.56%
URLLC          0  0/30 MHz          0%
mMTC           0  0/10 MHz          0%

New User Allocation:
User 2 → eMBB Slice
CQI: 3, Bandwidth: 5.0 MHz, Rate: 1.90 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |     3 |          5 |           1.9 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "My smart agriculture sensor needs to report soil temperature"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: N/A
Recommended Slice: mMTC - Soil‑temperature reporting from an agricultural sensor generates a very small payload (a few bytes) at low transmission frequency. It does not require the high throughput or ultra‑low latency of eMBB or URLLC slices. The mMTC slice is designed for massive machine‑type communications with relaxed latency (100‑1000 ms) and modest bandwidth (1‑3 MHz) and data‑rate (0.1‑1 Mbps) requirements, which perfectly matches the sensor's needs. With a CQI of 15 the channel quality is excellent, guaranteeing reliable transmission even within the minimal 1 MHz allocation. Allocating the sensor to mMTC keeps the eMBB and URLLC slices undisturbed, preserving their resources for future high‑demand users.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-30 11:14:26
Total Users: 2
Average Resource Utilization: 4.62%
eMBB Total Rate: 1.90 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  5.0/90 MHz        5.56%
URLLC          0  0/30 MHz          0%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 3 → mMTC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |     3 |          5 |           1.9 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |           0.5 |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "I need to participate in an online multiplayer game"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 25.0

Intent Analysis: N/A
Recommended Slice: URLLC - Multiplayer gaming requires low latency and moderate bandwidth. URLLC offers latency of 1‑10 ms and a bandwidth range of 1‑5 MHz, which can support typical game traffic (≈5‑20 Mbps). With CQI = 15 (high channel quality), a spectral efficiency of ~5 bits/s/Hz yields ≈25 Mbps on a 5 MHz allocation, well within the URLLC rate limits (1‑100 Mbps). This choice meets the latency‑sensitivity of the game while providing sufficient capacity.
Bandwidth Allocation: 5.0 MHz
Data Rate: 25.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-30 11:15:15
Total Users: 3
Average Resource Utilization: 8.46%
eMBB Total Rate: 1.90 Mbps, URLLC Total Rate: 25.00 Mbps, mMTC Total Rate: 0.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  5.0/90 MHz        5.56%
URLLC          1  5.0/30 MHz        16.67%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 4 → URLLC Slice
CQI: 15, Bandwidth: 5.0 MHz, Rate: 25.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         4 | URLLC   |    15 |          5 |          25   |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |           1.9 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |           0.5 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "A fleet of delivery drones needs to send low-rate telemetry data"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 2.0

Intent Analysis: Fleet of delivery drones sending low‑rate telemetry data
Recommended Slice: URLLC - Low‑rate telemetry fits the URLLC profile (1‑100 Mbps, 1‑10 ms latency). With CQI 3 the estimated spectral efficiency (~2 bps/Hz) yields ~2 Mbps on 1 MHz, meeting the requirement while staying within URLLC limits. Allocation leaves ample remaining capacity.
Bandwidth Allocation: 1.0 MHz
Data Rate: 2.0 Mbps
Latency: 0.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-30 11:16:01
Total Users: 4
Average Resource Utilization: 9.23%
eMBB Total Rate: 1.90 Mbps, URLLC Total Rate: 27.00 Mbps, mMTC Total Rate: 0.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  5.0/90 MHz        5.56%
URLLC          2  6.0/30 MHz        20.00%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 5 → URLLC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 2.00 Mbps, Latency: 0.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         4 | URLLC   |    15 |          5 |          25   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |           2   |              0 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |           1.9 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |           0.5 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I want to play competitive mobile games with ultra-low latency"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 1.5

Intent Analysis: User requires ultra‑low latency for competitive mobile gaming, which aligns with the capabilities of the URLLC slice.
Recommended Slice: URLLC - The URLLC slice provides the required 1‑10 ms latency. Allocating 2 MHz stays within the URLLC bandwidth limits (1‑5 MHz) and yields a data rate of ~1.5 Mbps, satisfying the minimum 1 Mbps requirement. The CQI of 4 indicates modest channel quality, so a conservative 2 MHz allocation ensures reliability while not overloading the slice. The eMBB and mMTC slices remain unchanged, keeping overall network load balanced.
Bandwidth Allocation: 2.0 MHz
Data Rate: 1.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 6
----------------------------------------
Network Status @ 2026-03-30 11:17:09
Total Users: 5
Average Resource Utilization: 10.77%
eMBB Total Rate: 1.90 Mbps, URLLC Total Rate: 28.50 Mbps, mMTC Total Rate: 0.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  5.0/90 MHz        5.56%
URLLC          3  8.0/30 MHz        26.67%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 6 → URLLC Slice
CQI: 4, Bandwidth: 2.0 MHz, Rate: 1.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         4 | URLLC   |    15 |          5 |          25   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |           2   |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |           1.5 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |           1.9 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |           0.5 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to participate in an online multiplayer game"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 25.0

Intent Analysis: User intends to play an online multiplayer game, which requires low latency and moderate bandwidth.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 25.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-30 11:17:52
Total Users: 6
Average Resource Utilization: 14.62%
eMBB Total Rate: 1.90 Mbps, URLLC Total Rate: 53.50 Mbps, mMTC Total Rate: 0.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  5.0/90 MHz        5.56%
URLLC          4  13.0/30 MHz       43.33%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 7 → URLLC Slice
CQI: 14, Bandwidth: 5.0 MHz, Rate: 25.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         4 | URLLC   |    15 |          5 |          25   |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |           2   |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |           1.5 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |          25   |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |           1.9 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |           0.5 |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "I want to browse websites and check email"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 4.385

Intent Analysis: N/A
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 4.385 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-30 11:19:10
Total Users: 7
Average Resource Utilization: 18.46%
eMBB Total Rate: 1.90 Mbps, URLLC Total Rate: 57.88 Mbps, mMTC Total Rate: 0.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  5.0/90 MHz        5.56%
URLLC          5  18.0/30 MHz       60.00%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 8 → URLLC Slice
CQI: 4, Bandwidth: 5.0 MHz, Rate: 4.38 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I want to monitor my home security cameras remotely"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 5.0

Intent Analysis: Remote video streaming from home security cameras – requires moderate to high bandwidth and can tolerate a few tens of milliseconds of latency.
Recommended Slice: eMBB - eMBB is designed for enhanced mobile broadband services that need higher data rates (100‑400 Mbps) and latency in the 10‑100 ms range.; The requested video streaming fits the eMBB profile better than URLLC (ultra‑low latency) or mMTC (very low rate, high latency).; eMBB slice currently uses only 5 MHz of its 90 MHz capacity, leaving ample room for a new user.
Bandwidth Allocation: 20.0 MHz
Data Rate: 5.0 Mbps
Latency: 30.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-30 11:20:56
Total Users: 8
Average Resource Utilization: 33.85%
eMBB Total Rate: 6.90 Mbps, URLLC Total Rate: 57.88 Mbps, mMTC Total Rate: 0.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  25.0/90 MHz       27.78%
URLLC          5  18.0/30 MHz       60.00%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 9 → eMBB Slice
CQI: 6, Bandwidth: 20.0 MHz, Rate: 5.00 Mbps, Latency: 30.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to make a high-quality voice call"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 2.5

Intent Analysis: The user requests a high‑quality voice call. Voice traffic is real‑time, low‑to‑moderate data rate, and benefits from low latency. A slice that can guarantee sub‑10 ms latency is therefore the best fit.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 2.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-30 11:21:56
Total Users: 9
Average Resource Utilization: 37.69%
eMBB Total Rate: 6.90 Mbps, URLLC Total Rate: 60.38 Mbps, mMTC Total Rate: 0.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  25.0/90 MHz       27.78%
URLLC          6  23.0/30 MHz       76.67%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 10 → URLLC Slice
CQI: 4, Bandwidth: 5.0 MHz, Rate: 2.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     4 |          5 |          2.5  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to send text messages and use messaging apps"
CQI: 3
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: The user requires basic messaging functionality (text messages and messaging apps). This is a low-bandwidth, latency-sensitive application typical of real-time communication services.
Recommended Slice: URLLC - Messaging apps require low-latency communication for real-time text exchange. URLLC provides the appropriate balance of low latency (1-10ms) and moderate bandwidth. While eMBB offers higher rates, it has higher latency. mMTC has appropriate latency but is optimized for massive IoT connectivity rather than user-centric messaging.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-30 11:22:38
Total Users: 10
Average Resource Utilization: 38.46%
eMBB Total Rate: 6.90 Mbps, URLLC Total Rate: 60.88 Mbps, mMTC Total Rate: 0.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  25.0/90 MHz       27.78%
URLLC          7  24.0/30 MHz       80.00%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 11 → URLLC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     4 |          5 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          1 |          0.5  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I want to play competitive mobile games with ultra-low latency"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 3.0

Intent Analysis: User requires ultra‑low latency (1‑10 ms) and a moderate data rate for interactive gaming
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 3.0 MHz
Data Rate: 3.0 Mbps
Latency: 1.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-30 11:24:05
Total Users: 11
Average Resource Utilization: 40.77%
eMBB Total Rate: 6.90 Mbps, URLLC Total Rate: 63.88 Mbps, mMTC Total Rate: 0.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  25.0/90 MHz       27.78%
URLLC          8  27.0/30 MHz       90.00%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 3, Bandwidth: 3.0 MHz, Rate: 3.00 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     4 |          5 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          1 |          0.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          3 |          3    |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I need real-time traffic updates for navigation"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 5.0

Intent Analysis: The user requests real‑time traffic updates for navigation. This service requires low‑latency communication with moderate data volume, best served by the URLLC slice.
Recommended Slice: URLLC - CQI 9 indicates moderate channel quality; allocating 2 MHz provides sufficient reliability while staying within the URLLC slice capacity. The resulting data rate of 5 Mbps meets the traffic‑update requirement and respects the URLLC latency budget.
Bandwidth Allocation: 2.0 MHz
Data Rate: 5.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-30 11:25:03
Total Users: 12
Average Resource Utilization: 42.31%
eMBB Total Rate: 6.90 Mbps, URLLC Total Rate: 68.88 Mbps, mMTC Total Rate: 0.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  25.0/90 MHz       27.78%
URLLC          9  29.0/30 MHz       96.67%
mMTC           1  1.0/10 MHz        10.00%

New User Allocation:
User 13 → URLLC Slice
CQI: 9, Bandwidth: 2.0 MHz, Rate: 5.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     4 |          5 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          1 |          0.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          3 |          3    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |          5    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: Periodic health data upload from a wearable device
Recommended Slice: mMTC - The wearable device generates low‑volume, delay‑tolerant traffic, which aligns with the mMTC slice characteristics (bandwidth 1‑3 MHz, rate 0.1‑1 Mbps, latency 100‑1000 ms). The current mMTC slice is lightly loaded (10 % utilization), providing ample capacity. A CQI of 6 indicates moderate channel quality; a 0.5 Mbps rate is achievable under this condition.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-30 11:25:47
Total Users: 13
Average Resource Utilization: 43.08%
eMBB Total Rate: 6.90 Mbps, URLLC Total Rate: 68.88 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  25.0/90 MHz       27.78%
URLLC          9  29.0/30 MHz       96.67%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 14 → mMTC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     4 |          5 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          1 |          0.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          3 |          3    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |          0.5  |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 13
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 18.0, rate: 100.0

Intent Analysis: User request for cloud‑based AI image processing is bandwidth‑intensive and tolerates moderate latency, making the eMBB slice the most suitable choice.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 18.0 MHz
Data Rate: 100.0 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-30 11:26:23
Total Users: 14
Average Resource Utilization: 56.92%
eMBB Total Rate: 106.90 Mbps, URLLC Total Rate: 68.88 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  43.0/90 MHz       47.78%
URLLC          9  29.0/30 MHz       96.67%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 13, Bandwidth: 18.0 MHz, Rate: 100.00 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     4 |          5 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          1 |          0.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          3 |          3    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         18 |        100    |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need instant facial recognition for public security threats"
CQI: 9
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 4.5

Intent Analysis: The user needs very low‑latency, high‑reliability connectivity to run real‑time facial‑recognition on video streams for public‑security purposes.
Recommended Slice: URLLC - URLLC currently uses 29 MHz of its 30 MHz total, leaving only 1 MHz free. Allocating the full 1 MHz keeps the slice within its capacity while satisfying the latency and rate constraints.
Bandwidth Allocation: 1.0 MHz
Data Rate: 4.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-30 11:27:29
Total Users: 15
Average Resource Utilization: 57.69%
eMBB Total Rate: 106.90 Mbps, URLLC Total Rate: 73.38 Mbps, mMTC Total Rate: 1.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  43.0/90 MHz       47.78%
URLLC         10  30.0/30 MHz       100.00%
mMTC           2  2.0/10 MHz        20.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 9, Bandwidth: 1.0 MHz, Rate: 4.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     4 |          5 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          1 |          0.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          3 |          3    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          1 |          4.5  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         18 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "I need to monitor and control critical manufacturing processes in real-time"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 10.0

Intent Analysis: Real‑time monitoring and control of critical manufacturing processes demands ultra‑reliable low‑latency communication (URLLC). The request aligns best with the URLLC slice, which provides the required latency (1‑10 ms) and moderate data rates.
Recommended Slice: URLLC - The URLLC slice is currently fully utilized (30/30 MHz). To satisfy the new user without breaching the URLLC latency budget, 2 MHz will be dynamically re‑allocated from the eMBB slice, which has ample spare capacity.
Bandwidth Allocation: 2.0 MHz
Data Rate: 10.0 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 2.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 17
----------------------------------------
Request: I need to monitor and control critical manufacturing processes in real-time
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 2.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need to check the status of my smart home sensors"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 3.0

Intent Analysis: N/A
Recommended Slice: mMTC - Smart home sensors are IoT devices - mMTC is purpose-built for this traffic pattern; Status checks involve small, intermittent data transfers; mMTC slice has significant available capacity (80% free); Low latency is not critical for periodic sensor polling; Avoids overloading the fully-utilized URLLC slice
Bandwidth Allocation: 2.0 MHz
Data Rate: 3.0 Mbps
Latency: 100.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-30 11:29:04
Total Users: 16
Average Resource Utilization: 59.23%
eMBB Total Rate: 106.90 Mbps, URLLC Total Rate: 73.38 Mbps, mMTC Total Rate: 4.00 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  43.0/90 MHz       47.78%
URLLC         10  30.0/30 MHz       100.00%
mMTC           3  4.0/10 MHz        40.00%

New User Allocation:
User 18 → mMTC Slice
CQI: 3, Bandwidth: 2.0 MHz, Rate: 3.00 Mbps, Latency: 100.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     4 |          5 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          1 |          0.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          3 |          3    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          1 |          4.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         18 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |          2 |          3    |            100 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "Industrial equipment monitoring sensor data"
CQI: 2
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 0.5

Intent Analysis: N/A
Recommended Slice: mMTC - The request is for industrial equipment monitoring sensor data, which typically involves low‑volume, periodic transmissions that fit the mMTC profile (0.1‑1 Mbps, 100‑1000 ms latency). The user’s CQI of 2 indicates poor channel quality, so a modest bandwidth of 2 MHz with an assumed spectral efficiency of ≈0.25 bits/s/Hz yields an estimated data rate of ~0.5 Mbps, comfortably within the mMTC rate window. The mMTC slice currently uses 4 MHz out of a 10 MHz total (40 % utilization), leaving ample headroom; adding 2 MHz raises utilization to 60 % while still preserving capacity for future devices. The eMBB slice is moderately loaded (47.78 %) and the URLLC slice is already at full capacity (100 %), making mMTC the most appropriate and balanced choice for this low‑rate sensor traffic.
Bandwidth Allocation: 2.0 MHz
Data Rate: 0.5 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-30 11:30:05
Total Users: 17
Average Resource Utilization: 60.77%
eMBB Total Rate: 106.90 Mbps, URLLC Total Rate: 73.38 Mbps, mMTC Total Rate: 4.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  43.0/90 MHz       47.78%
URLLC         10  30.0/30 MHz       100.00%
mMTC           4  6.0/10 MHz        60.00%

New User Allocation:
User 19 → mMTC Slice
CQI: 2, Bandwidth: 2.0 MHz, Rate: 0.50 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     4 |          5 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          1 |          0.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          3 |          3    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          1 |          4.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         18 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |          2 |          3    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |          2 |          0.5  |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I want to download a big game file"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 66.44

Intent Analysis: high‑bandwidth bulk data transfer (e.g., game file)
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 20.0 MHz
Data Rate: 66.44 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-30 11:31:29
Total Users: 18
Average Resource Utilization: 76.15%
eMBB Total Rate: 173.34 Mbps, URLLC Total Rate: 73.38 Mbps, mMTC Total Rate: 4.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  63.0/90 MHz       70.00%
URLLC         10  30.0/30 MHz       100.00%
mMTC           4  6.0/10 MHz        60.00%

New User Allocation:
User 20 → eMBB Slice
CQI: 11, Bandwidth: 20.0 MHz, Rate: 66.44 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     4 |          5 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          1 |          0.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          3 |          3    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          1 |          4.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         18 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |         66.44 |             20 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |          2 |          3    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |          2 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to update my social media status"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.0

Intent Analysis: N/A
Recommended Slice: mMTC - The request is a low‑volume status update that can tolerate moderate latency. mMTC supports the required low‑rate (0.1‑1 Mbps) with acceptable latency (100‑1000 ms) and leaves the heavily‑loaded URLLC slice untouched while preserving eMBB resources for high‑throughput services.
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.0 Mbps
Latency: 150.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-30 11:32:41
Total Users: 19
Average Resource Utilization: 76.92%
eMBB Total Rate: 173.34 Mbps, URLLC Total Rate: 73.38 Mbps, mMTC Total Rate: 5.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  63.0/90 MHz       70.00%
URLLC         10  30.0/30 MHz       100.00%
mMTC           5  7.0/10 MHz        70.00%

New User Allocation:
User 21 → mMTC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 1.00 Mbps, Latency: 150.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     4 |          5 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          1 |          0.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          3 |          3    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          1 |          4.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         18 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |         66.44 |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |          2 |          3    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |          2 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     6 |          1 |          1    |            150 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 2
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 3.8

Intent Analysis: The request demands ultra‑reliable, low‑latency connectivity for the control of critical infrastructure. This aligns with a URLLC (Ultra‑Reliable Low‑Latency Communications) use case where latency must be in the 1‑10 ms range and downtime must be avoided.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 3.8 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-30 11:33:55
Total Users: 20
Average Resource Utilization: 80.77%
eMBB Total Rate: 177.14 Mbps, URLLC Total Rate: 73.38 Mbps, mMTC Total Rate: 5.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  68.0/90 MHz       75.56%
URLLC         10  30.0/30 MHz       100.00%
mMTC           5  7.0/10 MHz        70.00%

New User Allocation:
User 22 → eMBB Slice
CQI: 2, Bandwidth: 5.0 MHz, Rate: 3.80 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     4 |          5 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          1 |          0.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          3 |          3    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          1 |          4.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         18 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |         66.44 |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | eMBB    |     2 |          5 |          3.8  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |          2 |          3    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |          2 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     6 |          1 |          1    |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I want to stream a webinar with interactive features"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 100.0

Intent Analysis: User wants to stream a webinar with interactive features. This requires moderate‑to‑high bandwidth for video and low latency for real‑time interaction (chat, Q&A), which aligns with eMBB slice capabilities.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 20.0 MHz
Data Rate: 100.0 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-30 11:34:39
Total Users: 21
Average Resource Utilization: 96.15%
eMBB Total Rate: 277.14 Mbps, URLLC Total Rate: 73.38 Mbps, mMTC Total Rate: 5.50 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  88.0/90 MHz       97.78%
URLLC         10  30.0/30 MHz       100.00%
mMTC           5  7.0/10 MHz        70.00%

New User Allocation:
User 23 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 100.00 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     4 |          5 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          1 |          0.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          3 |          3    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          1 |          4.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         18 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |         66.44 |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | eMBB    |     2 |          5 |          3.8  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | eMBB    |    12 |         20 |        100    |             20 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |          2 |          3    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |          2 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     6 |          1 |          1    |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 4
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 6.0, rate: 12.0

Intent Analysis: N/A
Recommended Slice: eMBB - User request for cloud‑based AI image processing aligns with the high‑throughput eMBB profile. The eMBB slice can provide 6‑20 MHz per user and supports rates up to 400 Mbps, with latency between 10‑100 ms. We allocate the minimum 6 MHz to fit the remaining capacity. With CQI 4 (≈16‑QAM ½ coding) the spectral efficiency is roughly 2 bit/s/Hz, yielding an estimated 12 Mbps for this user. While this rate is below the slice’s nominal 100‑400 Mbps, it is the best achievable given the current channel quality; link‑adaptation measures are recommended to improve the effective rate.
Bandwidth Allocation: 6.0 MHz
Data Rate: 12.0 Mbps
Latency: 30.0 ms

CAPACITY CHECK FAILED:
- Required: 6.0 MHz, Available: 2.0 MHz in eMBB slice

----------------------------------------
ALLOCATION FAILED FOR USER 24
----------------------------------------
Request: I need to use cloud-based AI services for image processing
Slice type: eMBB
Reason: Insufficient capacity in eMBB slice. Required: 6.0 MHz, Available: 2.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I need to synchronize distributed financial ledgers instantly"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 16.2

Intent Analysis: N/A
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 3.0 MHz
Data Rate: 16.2 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 3.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 25
----------------------------------------
Request: I need to synchronize distributed financial ledgers instantly
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 3.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to control a robotic arm in real time"
CQI: 2
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 5.0

Intent Analysis: N/A
Recommended Slice: URLLC - Real-time control applications require ultra-low latency (1-10ms) which is only supported by URLLC slice. eMBB (10-100ms latency) and mMTC (100-1000ms latency) cannot meet the latency requirements for robotic arm control.
Bandwidth Allocation: 1.0 MHz
Data Rate: 5.0 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 1.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 26
----------------------------------------
Request: I need to control a robotic arm in real time
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 1.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need emergency response coordination during a disaste"
CQI: 14
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 22.5

Intent Analysis: Emergency response coordination demands ultra‑reliable, low‑latency communication, which aligns best with the URLLC slice characteristics.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 22.5 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 5.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 27
----------------------------------------
Request: I need emergency response coordination during a disaste
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 5.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I want to use augmented reality navigation"
CQI: 6
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 6.0, rate: 120.0

Intent Analysis: The user request is for augmented‑reality navigation, which typically demands high data rates (≥100 Mbps) and moderate‑to‑low latency (<50 ms). Among the available slices, the eMBB slice is best suited because it offers the required high throughput and can support the latency range of 10‑100 ms. The eMBB slice is currently operating at 88 MHz of its 90 MHz total, leaving only 2 MHz of idle capacity. To meet the eMBB per‑user bandwidth constraints (6‑20 MHz) and to achieve the needed data rate, we must free at least 6 MHz and apply a high‑order modulation scheme (e.g., 256‑QAM or higher) to attain a spectral efficiency of roughly 20 bits/Hz, yielding ~120 Mbps within a 6 MHz allocation.
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 6.0 MHz
Data Rate: 120.0 Mbps
Latency: 30.0 ms

CAPACITY CHECK FAILED:
- Required: 6.0 MHz, Available: 2.0 MHz in eMBB slice

----------------------------------------
ALLOCATION FAILED FOR USER 28
----------------------------------------
Request: I want to use augmented reality navigation
Slice type: eMBB
Reason: Insufficient capacity in eMBB slice. Required: 6.0 MHz, Available: 2.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need immediate machine shutdown capability for safety incidents"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 15.0

Intent Analysis: N/A
Recommended Slice: URLLC - Safety shutdown requires ultra-low latency (1-10ms). URLLC slice provides the only suitable latency characteristics despite current 100% utilization. The critical nature of safety shutdown capability justifies resource reallocation.
Bandwidth Allocation: 2.0 MHz
Data Rate: 15.0 Mbps
Latency: 10.0 ms

CAPACITY CHECK FAILED:
- Required: 2.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 29
----------------------------------------
Request: I need immediate machine shutdown capability for safety incidents
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 2.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "My smart parking sensor needs to report if the spot is free"
CQI: 1
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.1

Intent Analysis: Transmit a very small status message (occupied/free) at low frequency; data volume is minimal and latency tolerance is moderate.
Recommended Slice: mMTC - Traffic is machine‑type, infrequent, and low‑rate, perfectly matching mMTC capabilities.; mMTC slice has sufficient remaining bandwidth (≈3 MHz) to accommodate the new user.; The eMBB and URLLC slices are at or near capacity and are not suitable for this low‑rate, latency‑tolerant service.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.1 Mbps
Latency: 1.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 30
----------------------------------------
Network Status @ 2026-03-30 11:44:16
Total Users: 22
Average Resource Utilization: 96.92%
eMBB Total Rate: 277.14 Mbps, URLLC Total Rate: 73.38 Mbps, mMTC Total Rate: 5.60 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  88.0/90 MHz       97.78%
URLLC         10  30.0/30 MHz       100.00%
mMTC           6  8.0/10 MHz        80.00%

New User Allocation:
User 30 → mMTC Slice
CQI: 1, Bandwidth: 1.0 MHz, Rate: 0.10 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     4 |          5 |          2.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | URLLC   |     3 |          1 |          0.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |          3 |          3    |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | URLLC   |     9 |          2 |          5    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |     9 |          1 |          4.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | URLLC   |    15 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | URLLC   |     3 |          1 |          2    |              0 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         6 | URLLC   |     4 |          2 |          1.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | URLLC   |    14 |          5 |         25    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | URLLC   |     4 |          5 |          4.38 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |    13 |         18 |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |     3 |          5 |          1.9  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | eMBB    |    11 |         20 |         66.44 |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | eMBB    |     2 |          5 |          3.8  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | eMBB    |    12 |         20 |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     6 |         20 |          5    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | mMTC    |     6 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | mMTC    |     3 |          2 |          3    |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | mMTC    |     2 |          2 |          0.5  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | mMTC    |     6 |          1 |          1    |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    15 |          1 |          0.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        30 | mMTC    |     1 |          1 |          0.1  |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI | BW (MHz)   | Rate (Mbps)   | Latency (ms)   | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Failed   | Failed  | URLLC          |                |     2 | N/A        | N/A           | N/A            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | eMBB    | URLLC          | No             |     3 | 5.0        | 1.9           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | mMTC    | mMTC           | Yes            |    15 | 1.0        | 0.5           | 500.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | URLLC   | URLLC          | Yes            |    15 | 5.0        | 25.0          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | URLLC   | mMTC           | No             |     3 | 1.0        | 2.0           | 0.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Success  | URLLC   | URLLC          | Yes            |     4 | 2.0        | 1.5           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | URLLC   | URLLC          | Yes            |    14 | 5.0        | 25.0          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | URLLC   | eMBB           | No             |     4 | 5.0        | 4.385         | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | eMBB    | eMBB           | Yes            |     6 | 20.0       | 5.0           | 30.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | URLLC   | eMBB           | No             |     4 | 5.0        | 2.5           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | URLLC   | eMBB           | No             |     3 | 1.0        | 0.5           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     3 | 3.0        | 3.0           | 1.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | URLLC   | eMBB           | No             |     9 | 2.0        | 5.0           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | mMTC    | mMTC           | Yes            |     6 | 1.0        | 0.5           | 500.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | eMBB           | Yes            |    13 | 18.0       | 100.0         | 10.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |     9 | 1.0        | 4.5           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Failed   | URLLC   | URLLC          |                |    15 | 2.0        | 10.0          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | mMTC    | mMTC           | Yes            |     3 | 2.0        | 3.0           | 100.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | mMTC    | mMTC           | Yes            |     2 | 2.0        | 0.5           | 200.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | eMBB    | eMBB           | Yes            |    11 | 20.0       | 66.44         | 20.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | mMTC    | eMBB           | No             |     6 | 1.0        | 1.0           | 150.0          | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | eMBB    | URLLC          | No             |     2 | 5.0        | 3.8           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | eMBB    | eMBB           | Yes            |    12 | 20.0       | 100.0         | 20.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Failed   | eMBB    | eMBB           |                |     4 | 6.0        | 12.0          | 30.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Failed   | URLLC   | URLLC          |                |    14 | 3.0        | 16.2          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Failed   | URLLC   | URLLC          |                |     2 | 1.0        | 5.0           | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Failed   | URLLC   | URLLC          |                |    14 | 5.0        | 22.5          | 5.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Failed   | eMBB    | eMBB           |                |     6 | 6.0        | 120.0         | 30.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Failed   | URLLC   | URLLC          |                |     3 | 2.0        | 15.0          | 10.0           | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Success  | mMTC    | mMTC           | Yes            |     1 | 1.0        | 0.1           | 1.0            | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 22/30 (73.3%)

Intent Understanding Evaluation:
Correctly identified intents: 14/22
Intent understanding rate: 63.6%

Workload Balancing Statistics:
Users with workload balancing: 22/30
Workload balancing rate: 73.3%

Slice Utilization Statistics:
Average eMBB utilization: 36.72%
Average URLLC utilization: 71.06%
Average mMTC utilization: 28.18%

Results exported to F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_south_minimax-m2.5.csv

[OK] TJU_south 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_south_minimax-m2.5.csv