============================================================
场景 4/5: TJU_gym
============================================================
输入: F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_gym.csv
输出: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_gym_minimax-m2.5.csv
------------------------------------------------------------
Starting prompt-based network slice management system...

Testing 30 users from ray tracing results CSV

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 1 (1/30)
Request: "A network of environmental sensors needs to report air quality"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 0.6

Intent Analysis: N/A
Recommended Slice: mMTC - Environmental sensors are IoT devices requiring mass connectivity with low data rates and tolerant latency. mMTC slice is specifically designed for massive machine-type communication with the appropriate profile for sensor networks.
Bandwidth Allocation: 2.0 MHz
Data Rate: 0.6 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 1
----------------------------------------
Network Status @ 2026-03-31 12:18:26
Total Users: 1
Average Resource Utilization: 1.54%
eMBB Total Rate: 0.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.60 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           0  0/90 MHz          0%
URLLC          0  0/30 MHz          0%
mMTC           1  2.0/10 MHz        20.00%

New User Allocation:
User 1 → mMTC Slice
CQI: 3, Bandwidth: 2.0 MHz, Rate: 0.60 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         1 | mMTC    |     3 |          2 |           0.6 |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 2 (2/30)
Request: "I want to browse websites and check email"
CQI: 11
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 15.0, rate: 108.0

Intent Analysis: web browsing and email
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 15.0 MHz
Data Rate: 108.0 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 2
----------------------------------------
Network Status @ 2026-03-31 12:18:53
Total Users: 2
Average Resource Utilization: 13.08%
eMBB Total Rate: 108.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 0.60 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          0  0/30 MHz          0%
mMTC           1  2.0/10 MHz        20.00%

New User Allocation:
User 2 → eMBB Slice
CQI: 11, Bandwidth: 15.0 MHz, Rate: 108.00 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |         15 |         108   |             20 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |           0.6 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 3 (3/30)
Request: "I need to monitor water level in a reservoir"
CQI: 13
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 0.8

Intent Analysis: Water level monitoring for a reservoir (IoT sensor data)
Recommended Slice: mMTC - The water‑level monitoring application generates low‑volume, periodic sensor readings that are best served by the mMTC slice (bandwidth 1‑3 MHz, rate ≤1 Mbps, latency 100‑1000 ms). With a CQI of 13 the channel offers high spectral efficiency, allowing a modest 2 MHz allocation while still respecting the slice’s rate ceiling. This leaves ample headroom for additional mMTC devices and avoids over‑provisioning the high‑speed eMBB slice.
Bandwidth Allocation: 2.0 MHz
Data Rate: 0.8 Mbps
Latency: 300.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 3
----------------------------------------
Network Status @ 2026-03-31 12:19:34
Total Users: 3
Average Resource Utilization: 14.62%
eMBB Total Rate: 108.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 1.40 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          0  0/30 MHz          0%
mMTC           2  4.0/10 MHz        40.00%

New User Allocation:
User 3 → mMTC Slice
CQI: 13, Bandwidth: 2.0 MHz, Rate: 0.80 Mbps, Latency: 300.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |         15 |         108   |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |           0.6 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |          2 |           0.8 |            300 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 4 (4/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 4
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.3

Intent Analysis: The user's wearable device needs to upload health data periodically. This is a low‑volume, machine‑type traffic that can tolerate relatively high latency and does not require the high data rates or ultra‑low latency of eMBB or URLLC slices. Therefore, the request best matches the mMTC (massive Machine‑Type Communications) slice, which is designed for many low‑power IoT devices with modest bandwidth and data‑rate needs.
Recommended Slice: mMTC - N/A
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.3 Mbps
Latency: 200.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 4
----------------------------------------
Network Status @ 2026-03-31 12:19:56
Total Users: 4
Average Resource Utilization: 15.38%
eMBB Total Rate: 108.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 1.70 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           1  15.0/90 MHz       16.67%
URLLC          0  0/30 MHz          0%
mMTC           3  5.0/10 MHz        50.00%

New User Allocation:
User 4 → mMTC Slice
CQI: 4, Bandwidth: 1.0 MHz, Rate: 0.30 Mbps, Latency: 200.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |         15 |         108   |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |           0.6 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |          2 |           0.8 |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |           0.3 |            200 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 5 (5/30)
Request: "I need to use cloud-based AI services for image processing"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 10.0, rate: 100.0

Intent Analysis: N/A
Recommended Slice: eMBB - eMBB provides the necessary high data rates (100‑400 Mbps) and latency (10‑100 ms) suitable for image processing. Current eMBB slice utilization is low (16.67 %), allowing resource allocation without congestion.
Bandwidth Allocation: 10.0 MHz
Data Rate: 100.0 Mbps
Latency: 10.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 5
----------------------------------------
Network Status @ 2026-03-31 12:20:30
Total Users: 5
Average Resource Utilization: 23.08%
eMBB Total Rate: 208.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 1.70 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  25.0/90 MHz       27.78%
URLLC          0  0/30 MHz          0%
mMTC           3  5.0/10 MHz        50.00%

New User Allocation:
User 5 → eMBB Slice
CQI: 8, Bandwidth: 10.0 MHz, Rate: 100.00 Mbps, Latency: 10.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |         15 |         108   |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |         10 |         100   |             10 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |          2 |           0.6 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |          2 |           0.8 |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |          1 |           0.3 |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 6 (6/30)
Request: "I need to stream 8K video content"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 80.0, rate: 200.0

Intent Analysis: high_bandwidth_media_consumption
Recommended Slice: eMBB - 8K video streaming requires high bandwidth (80-200 Mbps); CQI of 5 indicates moderate channel quality, suitable for eMBB with 16-QAM; eMBB slice designed for enhanced mobile broadband services; URLLC not required as streaming tolerates 10-100ms latency; mMTC slice inappropriate for high-bandwidth user applications
Bandwidth Allocation: 80.0 MHz
Data Rate: 200.0 Mbps
Latency: 25.0 ms

CAPACITY CHECK FAILED:
- Required: 80.0 MHz, Available: 65.0 MHz in eMBB slice

----------------------------------------
ALLOCATION FAILED FOR USER 6
----------------------------------------
Request: I need to stream 8K video content
Slice type: eMBB
Reason: Insufficient capacity in eMBB slice. Required: 80.0 MHz, Available: 65.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 7 (7/30)
Request: "I need to track the location of a shipping container"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.5, rate: 3.32

Intent Analysis: The user needs to track a shipping container, which is a classic IoT/asset tracking use case. This involves periodic transmission of small location data payloads (GPS coordinates, timestamps, container ID). This is NOT a high-bandwidth application like video streaming (eMBB) nor a latency-critical application like autonomous driving (URLLC). It requires low-power, wide-area connectivity for intermittent small data transmissions - typical mMTC behavior.
Recommended Slice: mMTC - Container tracking is a massive IoT application requiring: (1) Low bandwidth for small periodic data payloads, (2) Wide coverage for tracking across large geographic areas, (3) Energy efficiency for battery-powered trackers, (4) Support for large number of devices. mMTC slice is specifically designed for such machine-type communications with appropriate bandwidth (1-3 MHz) and rate (0.1-1 Mbps) ranges.
Bandwidth Allocation: 1.5 MHz
Data Rate: 3.32 Mbps
Latency: 150.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 7
----------------------------------------
Network Status @ 2026-03-31 12:21:21
Total Users: 6
Average Resource Utilization: 24.23%
eMBB Total Rate: 208.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 5.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  25.0/90 MHz       27.78%
URLLC          0  0/30 MHz          0%
mMTC           4  6.5/10 MHz        65.00%

New User Allocation:
User 7 → mMTC Slice
CQI: 6, Bandwidth: 1.5 MHz, Rate: 3.32 Mbps, Latency: 150.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 8 (8/30)
Request: "My wearable device needs to upload health data periodically"
CQI: 15
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 1.0

Intent Analysis: Low‑volume, periodic data transmission typical of IoT health monitors.
Recommended Slice: mMTC - mMTC is designed for massive machine‑type communications with low‑to‑moderate data rates (0.1‑1 Mbps).; Health‑data uploads from wearables are tolerant to higher latency (100‑1000 ms) and do not require ultra‑low latency of URLLC.; The required bandwidth (≈1 MHz) fits comfortably within the mMTC slice limits (1‑3 MHz).; CQI = 15 indicates excellent channel conditions, allowing the full mMTC rate to be utilized while still respecting the slice ceiling.
Bandwidth Allocation: 1.0 MHz
Data Rate: 1.0 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 8
----------------------------------------
Network Status @ 2026-03-31 12:21:53
Total Users: 7
Average Resource Utilization: 25.0%
eMBB Total Rate: 208.00 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 6.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           2  25.0/90 MHz       27.78%
URLLC          0  0/30 MHz          0%
mMTC           5  7.5/10 MHz        75.00%

New User Allocation:
User 8 → mMTC Slice
CQI: 15, Bandwidth: 1.0 MHz, Rate: 1.00 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 9 (9/30)
Request: "I need reliable connectivity for implanted medical devices"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 5.88

Intent Analysis: The user requires reliable, low‑latency connectivity for implanted medical devices, which is best served by the URLLC slice due to its strict latency (1‑10 ms) and high reliability characteristics.
Recommended Slice: eMBB - 5 MHz is the maximum allowable for URLLC and yields a comfortable margin above the 1 Mbps minimum while staying well below the 100 Mbps ceiling. The 10 % overhead reduction brings the usable rate to ≈5.3 Mbps, still within the slice’s rate range.
Bandwidth Allocation: 5.0 MHz
Data Rate: 5.88 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 9
----------------------------------------
Network Status @ 2026-03-31 12:22:33
Total Users: 8
Average Resource Utilization: 28.85%
eMBB Total Rate: 213.88 Mbps, URLLC Total Rate: 0.00 Mbps, mMTC Total Rate: 6.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  30.0/90 MHz       33.33%
URLLC          0  0/30 MHz          0%
mMTC           5  7.5/10 MHz        75.00%

New User Allocation:
User 9 → eMBB Slice
CQI: 7, Bandwidth: 5.0 MHz, Rate: 5.88 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 10 (10/30)
Request: "I need to check weather forecasts"
CQI: 5
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 2.4

Intent Analysis: Retrieve weather forecast data
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 2.0 MHz
Data Rate: 2.4 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 10
----------------------------------------
Network Status @ 2026-03-31 12:23:31
Total Users: 9
Average Resource Utilization: 30.38%
eMBB Total Rate: 213.88 Mbps, URLLC Total Rate: 2.40 Mbps, mMTC Total Rate: 6.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  30.0/90 MHz       33.33%
URLLC          1  2.0/30 MHz        6.67%
mMTC           5  7.5/10 MHz        75.00%

New User Allocation:
User 10 → URLLC Slice
CQI: 5, Bandwidth: 2.0 MHz, Rate: 2.40 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 11 (11/30)
Request: "I need to check the status of city-wide smart streetlights"
CQI: 6
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.5

Intent Analysis: N/A
Recommended Slice: mMTC - Streetlight monitoring involves IoT/M2M communication; Low to moderate data rate requirements; Large number of distributed devices across city; Status checks are tolerant of higher latency (100-1000ms); mMTC slice is specifically designed for massive IoT deployments; Periodic small data transmissions match mMTC characteristics
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.5 Mbps
Latency: 150.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 11
----------------------------------------
Network Status @ 2026-03-31 12:23:57
Total Users: 10
Average Resource Utilization: 31.15%
eMBB Total Rate: 213.88 Mbps, URLLC Total Rate: 2.40 Mbps, mMTC Total Rate: 6.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  30.0/90 MHz       33.33%
URLLC          1  2.0/30 MHz        6.67%
mMTC           6  8.5/10 MHz        85.00%

New User Allocation:
User 11 → mMTC Slice
CQI: 6, Bandwidth: 1.0 MHz, Rate: 0.50 Mbps, Latency: 150.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 12 (12/30)
Request: "I need microsecond-level latency for high-frequency tradin"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.6016

Intent Analysis: The user needs ultra‑low latency (microseconds) for high‑frequency trading. Among the defined slices, URLLC provides the lowest latency (1‑10 ms) and is the best match, even though true microsecond latency is beyond the standard URLLC spec.
Recommended Slice: URLLC - CQI 3 yields a low raw rate; we set the guaranteed rate to the minimum of the URLLC range (1 Mbps) to satisfy the slice constraints while still delivering the lowest possible latency.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.6016 Mbps
Latency: 1.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 12
----------------------------------------
Network Status @ 2026-03-31 12:24:31
Total Users: 11
Average Resource Utilization: 31.92%
eMBB Total Rate: 213.88 Mbps, URLLC Total Rate: 3.00 Mbps, mMTC Total Rate: 6.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           3  30.0/90 MHz       33.33%
URLLC          2  3.0/30 MHz        10.00%
mMTC           6  8.5/10 MHz        85.00%

New User Allocation:
User 12 → URLLC Slice
CQI: 3, Bandwidth: 1.0 MHz, Rate: 0.60 Mbps, Latency: 1.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 13 (13/30)
Request: "I want to use holographic communication"
CQI: 9
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 12.0, rate: 120.0

Intent Analysis: N/A
Recommended Slice: eMBB - N/A
Bandwidth Allocation: 12.0 MHz
Data Rate: 120.0 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 13
----------------------------------------
Network Status @ 2026-03-31 12:24:59
Total Users: 12
Average Resource Utilization: 41.15%
eMBB Total Rate: 333.88 Mbps, URLLC Total Rate: 3.00 Mbps, mMTC Total Rate: 6.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  42.0/90 MHz       46.67%
URLLC          2  3.0/30 MHz        10.00%
mMTC           6  8.5/10 MHz        85.00%

New User Allocation:
User 13 → eMBB Slice
CQI: 9, Bandwidth: 12.0 MHz, Rate: 120.00 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 14 (14/30)
Request: "I need to control critical infrastructure with zero downtime"
CQI: 12
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 27.5

Intent Analysis: Critical Infrastructure Control
Recommended Slice: URLLC - The requirement for 'zero downtime' and 'critical infrastructure control' aligns perfectly with URLLC (Ultra-Reliable Low-Latency Communication) slice characteristics. URLLC provides the required reliability and low latency for mission-critical applications that cannot tolerate service interruptions.
Bandwidth Allocation: 5.0 MHz
Data Rate: 27.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 14
----------------------------------------
Network Status @ 2026-03-31 12:25:20
Total Users: 13
Average Resource Utilization: 45.0%
eMBB Total Rate: 333.88 Mbps, URLLC Total Rate: 30.50 Mbps, mMTC Total Rate: 6.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           4  42.0/90 MHz       46.67%
URLLC          3  8.0/30 MHz        26.67%
mMTC           6  8.5/10 MHz        85.00%

New User Allocation:
User 14 → URLLC Slice
CQI: 12, Bandwidth: 5.0 MHz, Rate: 27.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 15 (15/30)
Request: "I need real-time fraud detection for financial transactions"
CQI: 7
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 9.57

Intent Analysis: The user requires real‑time fraud detection for financial transactions. This service demands very low latency (sub‑10 ms) to process each transaction promptly and a moderate data rate to handle transaction records. These characteristics match the URLLC slice profile rather than the higher‑throughput but higher‑latency eMBB slice or the low‑rate, high‑latency mMTC slice.
Recommended Slice: eMBB - The calculated rate (≈9.6 Mbps) lies comfortably within the URLLC allowed range of 1‑100 Mbps, and the latency target of 5 ms satisfies the 1‑10 ms URLLC requirement. No reduction is needed.
Bandwidth Allocation: 5.0 MHz
Data Rate: 9.57 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 15
----------------------------------------
Network Status @ 2026-03-31 12:25:56
Total Users: 14
Average Resource Utilization: 48.85%
eMBB Total Rate: 343.45 Mbps, URLLC Total Rate: 30.50 Mbps, mMTC Total Rate: 6.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  47.0/90 MHz       52.22%
URLLC          3  8.0/30 MHz        26.67%
mMTC           6  8.5/10 MHz        85.00%

New User Allocation:
User 15 → eMBB Slice
CQI: 7, Bandwidth: 5.0 MHz, Rate: 9.57 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |        5   |          9.57 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 16 (16/30)
Request: "I need instant facial recognition for public security threats"
CQI: 10
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 50.0

Intent Analysis: N/A
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 50.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 16
----------------------------------------
Network Status @ 2026-03-31 12:26:20
Total Users: 15
Average Resource Utilization: 52.69%
eMBB Total Rate: 343.45 Mbps, URLLC Total Rate: 80.50 Mbps, mMTC Total Rate: 6.52 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  47.0/90 MHz       52.22%
URLLC          4  13.0/30 MHz       43.33%
mMTC           6  8.5/10 MHz        85.00%

New User Allocation:
User 16 → URLLC Slice
CQI: 10, Bandwidth: 5.0 MHz, Rate: 50.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        5   |         50    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |        5   |          9.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 17 (17/30)
Request: "My smart trash can needs to signal that it's full"
CQI: 13
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.5, rate: 1.5

Intent Analysis: Status notification - bin full alert
Recommended Slice: mMTC - Smart trash can is an IoT device sending low-bandwidth status updates. This aligns with mMTC (massive Machine Type Communication) characteristics: small data packets, tolerance for higher latency, and massive device connectivity.
Bandwidth Allocation: 1.5 MHz
Data Rate: 1.5 Mbps
Latency: 500.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 17
----------------------------------------
Network Status @ 2026-03-31 12:26:42
Total Users: 16
Average Resource Utilization: 53.85%
eMBB Total Rate: 343.45 Mbps, URLLC Total Rate: 80.50 Mbps, mMTC Total Rate: 8.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  47.0/90 MHz       52.22%
URLLC          4  13.0/30 MHz       43.33%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 17 → mMTC Slice
CQI: 13, Bandwidth: 1.5 MHz, Rate: 1.50 Mbps, Latency: 500.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        5   |         50    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |        5   |          9.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        1.5 |          1.5  |            500 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 18 (18/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 1.13

Intent Analysis: N/A
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 3.0 MHz
Data Rate: 1.13 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 18
----------------------------------------
Network Status @ 2026-03-31 12:27:18
Total Users: 17
Average Resource Utilization: 56.15%
eMBB Total Rate: 343.45 Mbps, URLLC Total Rate: 81.63 Mbps, mMTC Total Rate: 8.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  47.0/90 MHz       52.22%
URLLC          5  16.0/30 MHz       53.33%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 18 → URLLC Slice
CQI: 3, Bandwidth: 3.0 MHz, Rate: 1.13 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        5   |         50    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |          1.13 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |        5   |          9.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        1.5 |          1.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 19 (19/30)
Request: "I need to monitor water level in a reservoir"
CQI: 3
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 1.16

Intent Analysis: The user wants to monitor water level in a reservoir. This is a typical IoT telemetry use‑case that requires a modest, periodic data transfer with moderate latency tolerance and high reliability. The traffic volume is low, but continuous connectivity is essential.
Recommended Slice: URLLC - URLLC provides the low‑latency (1‑10 ms) needed for reliable monitoring while offering a data rate above the 1 Mbps minimum. Allocating 2 MHz yields a rate of ~1.16 Mbps, satisfying both the slice constraints and the user’s low‑rate requirement.
Bandwidth Allocation: 2.0 MHz
Data Rate: 1.16 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 19
----------------------------------------
Network Status @ 2026-03-31 12:28:04
Total Users: 18
Average Resource Utilization: 57.69%
eMBB Total Rate: 343.45 Mbps, URLLC Total Rate: 82.79 Mbps, mMTC Total Rate: 8.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  47.0/90 MHz       52.22%
URLLC          6  18.0/30 MHz       60.00%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 19 → URLLC Slice
CQI: 3, Bandwidth: 2.0 MHz, Rate: 1.16 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        5   |         50    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |          1.13 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     3 |        2   |          1.16 |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |        5   |          9.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        1.5 |          1.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 20 (20/30)
Request: "I need to make a high-quality voice call"
CQI: 10
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 256.0

Intent Analysis: real‑time voice communication
Recommended Slice: URLLC - Voice call demands low latency (≤10 ms) and moderate bandwidth; eMBB slice imposes higher bandwidth and rate ranges unnecessary for voice; mMTC slice is saturated and unsuitable.
Bandwidth Allocation: 1.0 MHz
Data Rate: 256.0 Mbps
Latency: 50.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 20
----------------------------------------
Network Status @ 2026-03-31 12:28:49
Total Users: 19
Average Resource Utilization: 58.46%
eMBB Total Rate: 343.45 Mbps, URLLC Total Rate: 338.79 Mbps, mMTC Total Rate: 8.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           5  47.0/90 MHz       52.22%
URLLC          7  19.0/30 MHz       63.33%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 20 → URLLC Slice
CQI: 10, Bandwidth: 1.0 MHz, Rate: 256.00 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        5   |         50    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |          1.13 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     3 |        2   |          1.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |        256    |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |        5   |          9.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        1.5 |          1.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 21 (21/30)
Request: "I want to use holographic communication"
CQI: 12
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 20.0, rate: 100.0

Intent Analysis: User requests holographic communication, which demands high data rates, substantial bandwidth, and moderate latency. These characteristics best align with the eMBB slice capabilities (6‑20 MHz bandwidth, 100‑400 Mbps data rate, 10‑100 ms latency).
Recommended Slice: eMBB - Bandwidth within the eMBB‑allowed range (6‑20 MHz) and data rate satisfies the minimum 100 Mbps requirement while staying within the 100‑400 Mbps per‑user limit. Latency (20 ms) falls within the eMBB latency window (10‑100 ms).
Bandwidth Allocation: 20.0 MHz
Data Rate: 100.0 Mbps
Latency: 20.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 21
----------------------------------------
Network Status @ 2026-03-31 12:29:36
Total Users: 20
Average Resource Utilization: 73.85%
eMBB Total Rate: 443.45 Mbps, URLLC Total Rate: 338.79 Mbps, mMTC Total Rate: 8.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  67.0/90 MHz       74.44%
URLLC          7  19.0/30 MHz       63.33%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 21 → eMBB Slice
CQI: 12, Bandwidth: 20.0 MHz, Rate: 100.00 Mbps, Latency: 20.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        5   |         50    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |          1.13 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     3 |        2   |          1.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |        256    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |        5   |          9.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        100    |             20 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        1.5 |          1.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 22 (22/30)
Request: "I need to check the status of my smart home sensors"
CQI: 11
Ground Truth Slice: mMTC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 1.0, rate: 0.1

Intent Analysis: Check status of smart home sensors
Recommended Slice: URLLC - The request is low‑rate and latency‑tolerant, best served by mMTC.; mMTC slice is fully saturated (10/10 MHz, 100 % utilization) → no free resources.; URLLC slice has spare capacity (19/30 MHz used) and can accommodate the low‑rate request while meeting latency requirements.; eMBB slice could host the traffic but would be an overkill and would consume scarce high‑throughput resources.
Bandwidth Allocation: 1.0 MHz
Data Rate: 0.1 Mbps
Latency: 100.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 22
----------------------------------------
Network Status @ 2026-03-31 12:30:22
Total Users: 21
Average Resource Utilization: 74.62%
eMBB Total Rate: 443.45 Mbps, URLLC Total Rate: 338.89 Mbps, mMTC Total Rate: 8.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           6  67.0/90 MHz       74.44%
URLLC          8  20.0/30 MHz       66.67%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 22 → URLLC Slice
CQI: 11, Bandwidth: 1.0 MHz, Rate: 0.10 Mbps, Latency: 100.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        5   |         50    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |          1.13 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     3 |        2   |          1.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |        256    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |    11 |        1   |          0.1  |            100 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |        5   |          9.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        1.5 |          1.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 23 (23/30)
Request: "I need to detect and isolate power grid faults instantly"
CQI: 1
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 2.0

Intent Analysis: N/A
Recommended Slice: eMBB - Instant fault detection and isolation require ultra‑reliable low‑latency communication. URLLC supports the 1‑10 ms latency window needed for real‑time grid protection. With CQI = 1 the channel offers very limited spectral efficiency, therefore a modest bandwidth (2 MHz) and a conservative data‑rate target (2 Mbps) are allocated to maximize reliability while staying within slice limits.
Bandwidth Allocation: 2.0 MHz
Data Rate: 2.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 23
----------------------------------------
Network Status @ 2026-03-31 12:30:44
Total Users: 22
Average Resource Utilization: 76.15%
eMBB Total Rate: 445.45 Mbps, URLLC Total Rate: 338.89 Mbps, mMTC Total Rate: 8.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  69.0/90 MHz       76.67%
URLLC          8  20.0/30 MHz       66.67%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 23 → eMBB Slice
CQI: 1, Bandwidth: 2.0 MHz, Rate: 2.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        5   |         50    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |          1.13 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     3 |        2   |          1.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |        256    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |    11 |        1   |          0.1  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |        5   |          9.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | eMBB    |     1 |        2   |          2    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        1.5 |          1.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 24 (24/30)
Request: "I want to use remote surgery equipment"
CQI: 5
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 5.0, rate: 15.0

Intent Analysis: The request involves mission‑critical, real‑time control of surgical tools and high‑definition video feedback. This demands ultra‑reliable low‑latency communication with moderate to high data rates, best served by the URLLC slice.
Recommended Slice: URLLC - N/A
Bandwidth Allocation: 5.0 MHz
Data Rate: 15.0 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 24
----------------------------------------
Network Status @ 2026-03-31 12:31:11
Total Users: 23
Average Resource Utilization: 80.0%
eMBB Total Rate: 445.45 Mbps, URLLC Total Rate: 353.89 Mbps, mMTC Total Rate: 8.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           7  69.0/90 MHz       76.67%
URLLC          9  25.0/30 MHz       83.33%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 24 → URLLC Slice
CQI: 5, Bandwidth: 5.0 MHz, Rate: 15.00 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        5   |         50    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |          1.13 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     3 |        2   |          1.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |        256    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |    11 |        1   |          0.1  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         15    |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |        5   |          9.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | eMBB    |     1 |        2   |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        1.5 |          1.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 25 (25/30)
Request: "I want to stream music while browsing social media"
CQI: 2
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 6.0, rate: 3.0

Intent Analysis: Streaming music while browsing social media
Recommended Slice: eMBB - User traffic requires higher data rates than URLLC or mMTC can provide, and its latency tolerance fits eMBB
Bandwidth Allocation: 6.0 MHz
Data Rate: 3.0 Mbps
Latency: 30.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 25
----------------------------------------
Network Status @ 2026-03-31 12:32:05
Total Users: 24
Average Resource Utilization: 84.62%
eMBB Total Rate: 448.45 Mbps, URLLC Total Rate: 353.89 Mbps, mMTC Total Rate: 8.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  75.0/90 MHz       83.33%
URLLC          9  25.0/30 MHz       83.33%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 25 → eMBB Slice
CQI: 2, Bandwidth: 6.0 MHz, Rate: 3.00 Mbps, Latency: 30.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        5   |         50    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |          1.13 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     3 |        2   |          1.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |        256    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |    11 |        1   |          0.1  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         15    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |        5   |          9.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | eMBB    |     1 |        2   |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     2 |        6   |          3    |             30 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        1.5 |          1.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 26 (26/30)
Request: "I need to balance electrical load in real-time across microgrids"
CQI: 6
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 7.5

Intent Analysis: Real‑time balancing of electrical load across microgrids
Recommended Slice: URLLC - The request demands low‑latency, reliable communication for control‑level telemetry. URLLC supports the required 1‑10 ms latency and can provide the necessary data rate (well within the 1‑100 Mbps range). A 3 MHz allocation leaves a 2 MHz safety margin for other URLLC users.
Bandwidth Allocation: 3.0 MHz
Data Rate: 7.5 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 26
----------------------------------------
Network Status @ 2026-03-31 12:32:42
Total Users: 25
Average Resource Utilization: 86.92%
eMBB Total Rate: 448.45 Mbps, URLLC Total Rate: 361.39 Mbps, mMTC Total Rate: 8.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  75.0/90 MHz       83.33%
URLLC         10  28.0/30 MHz       93.33%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 26 → URLLC Slice
CQI: 6, Bandwidth: 3.0 MHz, Rate: 7.50 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        5   |         50    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |          1.13 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     3 |        2   |          1.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |        256    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |    11 |        1   |          0.1  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         15    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        3   |          7.5  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |        5   |          9.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | eMBB    |     1 |        2   |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     2 |        6   |          3    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        1.5 |          1.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 27 (27/30)
Request: "I need to transmit real-time patient vital signs during critical care"
CQI: 3
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 2.0, rate: 1.2

Intent Analysis: N/A
Recommended Slice: URLLC - Latency requirement (≤10 ms) matches the URLLC slice (1‑10 ms); Data‑rate requirement (≈1 Mbps) fits the URLLC rate envelope (1‑100 Mbps); CQI = 3 indicates a modest channel quality; allocating a wider bandwidth in the URLLC slice compensates the lower spectral efficiency
Bandwidth Allocation: 2.0 MHz
Data Rate: 1.2 Mbps
Latency: 5.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 27
----------------------------------------
Network Status @ 2026-03-31 12:33:35
Total Users: 26
Average Resource Utilization: 88.46%
eMBB Total Rate: 448.45 Mbps, URLLC Total Rate: 362.59 Mbps, mMTC Total Rate: 8.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           8  75.0/90 MHz       83.33%
URLLC         11  30.0/30 MHz       100.00%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 27 → URLLC Slice
CQI: 3, Bandwidth: 2.0 MHz, Rate: 1.20 Mbps, Latency: 5.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        5   |         50    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |          1.13 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     3 |        2   |          1.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |        256    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |    11 |        1   |          0.1  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         15    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        3   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        2   |          1.2  |              5 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |        5   |          9.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | eMBB    |     1 |        2   |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     2 |        6   |          3    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        1.5 |          1.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 28 (28/30)
Request: "I need reliable communication for firefighters inside buildings"
CQI: 4
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 3.0, rate: 1.8

Intent Analysis: The request is for mission‑critical, real‑time communication that demands very low latency (1‑10 ms) and high reliability. This matches the characteristics of the URLLC slice rather than eMBB (latency 10‑100 ms) or mMTC (latency 100‑1000 ms). Therefore the user should be attached to the URLLC slice.
Recommended Slice: URLLC - CQI 4 indicates moderate channel quality, so a conservative spectral efficiency of 0.6 bits/s/Hz is used to ensure robustness. The chosen bandwidth (3 MHz) falls within the URLLC per‑user range (1‑5 MHz), the resulting rate (1.8 Mbps) satisfies the URLLC rate window (1‑100 Mbps), and the latency (5 ms) meets the URLLC latency requirement (1‑10 ms).
Bandwidth Allocation: 3.0 MHz
Data Rate: 1.8 Mbps
Latency: 5.0 ms

CAPACITY CHECK FAILED:
- Required: 3.0 MHz, Available: 0.0 MHz in URLLC slice

----------------------------------------
ALLOCATION FAILED FOR USER 28
----------------------------------------
Request: I need reliable communication for firefighters inside buildings
Slice type: URLLC
Reason: Insufficient capacity in URLLC slice. Required: 3.0 MHz, Available: 0.0 MHz

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 29 (29/30)
Request: "I need to participate in an online multiplayer game"
CQI: 15
Ground Truth Slice: URLLC
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 15.0, rate: 10.0

Intent Analysis: Low‑latency, interactive data exchange typical of multiplayer games.
Recommended Slice: eMBB - eMBB is the only slice with free resources (15 MHz remaining).; Bandwidth falls within the eMBB range (6‑20 MHz).; Latency can be met by high‑priority scheduling (eMBB supports ≤ 100 ms).; The resulting rate (≈ 82.5 Mbps) is lower than the typical eMBB lower bound (100 Mbps) but is the maximum possible under current load.
Bandwidth Allocation: 15.0 MHz
Data Rate: 10.0 Mbps
Latency: 50.0 ms

ALLOCATION SUCCESSFUL: All constraints satisfied

----------------------------------------
ALLOCATION RESULT FOR USER 29
----------------------------------------
Network Status @ 2026-03-31 12:35:39
Total Users: 27
Average Resource Utilization: 100.0%
eMBB Total Rate: 458.45 Mbps, URLLC Total Rate: 362.59 Mbps, mMTC Total Rate: 8.02 Mbps

Slice      Users  Resource Usage    Utilization
-------  -------  ----------------  -------------
eMBB           9  90.0/90 MHz       100.00%
URLLC         11  30.0/30 MHz       100.00%
mMTC           7  10.0/10 MHz       100.00%

New User Allocation:
User 29 → eMBB Slice
CQI: 15, Bandwidth: 15.0 MHz, Rate: 10.00 Mbps, Latency: 50.0 ms

Current User Allocations:
+-----------+---------+-------+------------+---------------+----------------+----------+
|   User ID | Slice   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Status   |
+===========+=========+=======+============+===============+================+==========+
|        10 | URLLC   |     5 |        2   |          2.4  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        12 | URLLC   |     3 |        1   |          0.6  |              1 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        14 | URLLC   |    12 |        5   |         27.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        16 | URLLC   |    10 |        5   |         50    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        18 | URLLC   |     3 |        3   |          1.13 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        19 | URLLC   |     3 |        2   |          1.16 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        20 | URLLC   |    10 |        1   |        256    |             50 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        22 | URLLC   |    11 |        1   |          0.1  |            100 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        24 | URLLC   |     5 |        5   |         15    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        26 | URLLC   |     6 |        3   |          7.5  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        27 | URLLC   |     3 |        2   |          1.2  |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        13 | eMBB    |     9 |       12   |        120    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        15 | eMBB    |     7 |        5   |          9.57 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         2 | eMBB    |    11 |       15   |        108    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        21 | eMBB    |    12 |       20   |        100    |             20 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        23 | eMBB    |     1 |        2   |          2    |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        25 | eMBB    |     2 |        6   |          3    |             30 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        29 | eMBB    |    15 |       15   |         10    |             50 | NEW      |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         5 | eMBB    |     8 |       10   |        100    |             10 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         9 | eMBB    |     7 |        5   |          5.88 |              5 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         1 | mMTC    |     3 |        2   |          0.6  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        11 | mMTC    |     6 |        1   |          0.5  |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|        17 | mMTC    |    13 |        1.5 |          1.5  |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         3 | mMTC    |    13 |        2   |          0.8  |            300 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         4 | mMTC    |     4 |        1   |          0.3  |            200 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         7 | mMTC    |     6 |        1.5 |          3.32 |            150 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+
|         8 | mMTC    |    15 |        1   |          1    |            500 |          |
+-----------+---------+-------+------------+---------------+----------------+----------+

--------------------------------------------------------------------------------------------------------------------------------------------
PROCESSING USER 30 (30/30)
Request: "I want to browse websites and check email"
CQI: 8
Ground Truth Slice: eMBB
--------------------------------------------------------------------------------------------------------------------------------------------

[DEBUG] Raw result parsed successfully

[DEBUG] Normalized bandwidth: 12.0, rate: 103.7

Intent Analysis: The user needs a moderate‑speed connection with acceptable latency (typical web browsing/email traffic). This profile best matches the eMBB slice.
Recommended Slice: eMBB - A 6 MHz allocation would yield ≈ 52 Mbps, below the eMBB minimum of 100 Mbps. To satisfy the slice’s rate requirement, 12 MHz is allocated, achieving ≈ 104 Mbps.
Bandwidth Allocation: 12.0 MHz
Data Rate: 103.7 Mbps
Latency: 15.0 ms

CAPACITY CHECK FAILED:
- Required: 12.0 MHz, Available: 0.0 MHz in eMBB slice

----------------------------------------
ALLOCATION FAILED FOR USER 30
----------------------------------------
Request: I want to browse websites and check email
Slice type: eMBB
Reason: Insufficient capacity in eMBB slice. Required: 12.0 MHz, Available: 0.0 MHz

============================================================
SUMMARY OF USER ALLOCATIONS
============================================================
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|   User ID | Status   | Slice   | Ground Truth   | Intent Match   |   CQI |   BW (MHz) |   Rate (Mbps) |   Latency (ms) | Adjusted   |
+===========+==========+=========+================+================+=======+============+===============+================+============+
|         1 | Success  | mMTC    | mMTC           | Yes            |     3 |        2   |        0.6    |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         2 | Success  | eMBB    | eMBB           | Yes            |    11 |       15   |      108      |             20 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         3 | Success  | mMTC    | mMTC           | Yes            |    13 |        2   |        0.8    |            300 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         4 | Success  | mMTC    | mMTC           | Yes            |     4 |        1   |        0.3    |            200 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         5 | Success  | eMBB    | eMBB           | Yes            |     8 |       10   |      100      |             10 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         6 | Failed   | eMBB    | eMBB           |                |     5 |       80   |      200      |             25 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         7 | Success  | mMTC    | mMTC           | Yes            |     6 |        1.5 |        3.32   |            150 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         8 | Success  | mMTC    | mMTC           | Yes            |    15 |        1   |        1      |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|         9 | Success  | eMBB    | URLLC          | No             |     7 |        5   |        5.88   |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        10 | Success  | URLLC   | eMBB           | No             |     5 |        2   |        2.4    |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        11 | Success  | mMTC    | mMTC           | Yes            |     6 |        1   |        0.5    |            150 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        12 | Success  | URLLC   | URLLC          | Yes            |     3 |        1   |        0.6016 |              1 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        13 | Success  | eMBB    | eMBB           | Yes            |     9 |       12   |      120      |             20 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        14 | Success  | URLLC   | URLLC          | Yes            |    12 |        5   |       27.5    |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        15 | Success  | eMBB    | URLLC          | No             |     7 |        5   |        9.57   |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        16 | Success  | URLLC   | URLLC          | Yes            |    10 |        5   |       50      |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        17 | Success  | mMTC    | mMTC           | Yes            |    13 |        1.5 |        1.5    |            500 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        18 | Success  | URLLC   | URLLC          | Yes            |     3 |        3   |        1.13   |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        19 | Success  | URLLC   | mMTC           | No             |     3 |        2   |        1.16   |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        20 | Success  | URLLC   | eMBB           | No             |    10 |        1   |      256      |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        21 | Success  | eMBB    | eMBB           | Yes            |    12 |       20   |      100      |             20 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        22 | Success  | URLLC   | mMTC           | No             |    11 |        1   |        0.1    |            100 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        23 | Success  | eMBB    | URLLC          | No             |     1 |        2   |        2      |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        24 | Success  | URLLC   | URLLC          | Yes            |     5 |        5   |       15      |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        25 | Success  | eMBB    | eMBB           | Yes            |     2 |        6   |        3      |             30 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        26 | Success  | URLLC   | URLLC          | Yes            |     6 |        3   |        7.5    |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        27 | Success  | URLLC   | URLLC          | Yes            |     3 |        2   |        1.2    |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        28 | Failed   | URLLC   | URLLC          |                |     4 |        3   |        1.8    |              5 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        29 | Success  | eMBB    | URLLC          | No             |    15 |       15   |       10      |             50 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+
|        30 | Failed   | eMBB    | eMBB           |                |     8 |       12   |      103.7    |             15 | No         |
+-----------+----------+---------+----------------+----------------+-------+------------+---------------+----------------+------------+

Statistics:
Success rate: 27/30 (90.0%)

Intent Understanding Evaluation:
Correctly identified intents: 19/27
Intent understanding rate: 70.4%

Workload Balancing Statistics:
Users with workload balancing: 27/30
Workload balancing rate: 90.0%

Slice Utilization Statistics:
Average eMBB utilization: 49.09%
Average URLLC utilization: 37.28%
Average mMTC utilization: 80.74%

Results exported to F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_gym_minimax-m2.5.csv

[OK] TJU_gym 完成! 结果已保存到: F:\code\wirelessagent\run_results\batch_run\prompt_based\minimax-m2.5\network_slicing_results_TJU_gym_minimax-m2.5.csv