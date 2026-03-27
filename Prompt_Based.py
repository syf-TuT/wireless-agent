# 2025-03-25 13:30 Create prompt-based network slicing implementation
# 2025-03-25 14:45 Add constraint checking function for allocation validation
import json
import math
import random
from datetime import datetime
import re
import copy
from tabulate import tabulate
import pandas as pd
import csv
from typing import Dict, List, Any, Optional, TypedDict

# ====================== Token Tracking ======================
TOKEN_STATS = {
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_tokens": 0,
    "llm_call_count": 0
}

def get_token_usage():
    """Get current token usage statistics"""
    return TOKEN_STATS.copy()

def reset_token_stats():
    """Reset token statistics"""
    global TOKEN_STATS
    TOKEN_STATS = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "llm_call_count": 0
    }

def llm_with_token_tracking(llm, messages, operation_name="LLM"):
    """Wrapper for LLM invocation that tracks token usage"""
    global TOKEN_STATS

    # Estimate prompt tokens before call
    prompt_tokens_estimate = 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        for m in messages:
            if hasattr(m, 'content'):
                prompt_tokens_estimate += len(enc.encode(str(m.content)))
            else:
                prompt_tokens_estimate += len(enc.encode(str(m)))
    except:
        for m in messages:
            if hasattr(m, 'content'):
                prompt_tokens_estimate += len(str(m.content)) // 4
            else:
                prompt_tokens_estimate += len(str(m)) // 4

    # Invoke LLM
    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"[LLM Error] {operation_name}: {e}")
        return None

    # Try to get actual token usage from response
    completion_tokens = 0
    total_tokens = 0

    try:
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            prompt_tokens = usage.get('input_tokens', prompt_tokens_estimate)
            completion_tokens = usage.get('output_tokens', 0)
            total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
        elif hasattr(response, 'response_metadata') and response.response_metadata:
            resp_meta = response.response_metadata
            if 'usage' in resp_meta:
                usage = resp_meta['usage']
                prompt_tokens = usage.get('prompt_tokens', prompt_tokens_estimate)
                completion_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
            else:
                completion_tokens = len(response.content) // 4
                total_tokens = prompt_tokens_estimate + completion_tokens
        else:
            completion_tokens = len(response.content) // 4
            total_tokens = prompt_tokens_estimate + completion_tokens
    except Exception:
        completion_tokens = len(response.content) // 4 if response and hasattr(response, 'content') else 0
        total_tokens = prompt_tokens_estimate + completion_tokens

    actual_prompt_tokens = prompt_tokens_estimate
    actual_completion_tokens = completion_tokens
    actual_total_tokens = actual_prompt_tokens + actual_completion_tokens

    TOKEN_STATS["total_prompt_tokens"] += actual_prompt_tokens
    TOKEN_STATS["total_completion_tokens"] += actual_completion_tokens
    TOKEN_STATS["total_tokens"] += actual_total_tokens
    TOKEN_STATS["llm_call_count"] += 1

    print(f"[Token] {operation_name}: prompt~{actual_prompt_tokens}, completion~{actual_completion_tokens}, total~{actual_total_tokens}")

    return response

# LLM Configuration
from llm_config import get_llm

# Default model name (can be changed via run_all_scenarios_prompt_based.py)
MODEL_NAME = "glm-5"

# Get LLM instance
llm = get_llm(MODEL_NAME)

def call_llm(prompt):
    """Call the LLM API with the given prompt using configured model"""
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return "Error: Unable to get a response from the LLM API."


# ====================== CSV Data Loading Function ======================

def load_user_data_from_csv(file_path, num_users=None):
    """Load user data from ray tracing CSV file"""
    try:
        df = pd.read_csv(file_path)
        users = []
        
        # Limit to specified number of users if needed
        if num_users is not None and num_users < len(df):
            df = df.head(num_users)
        
        for _, row in df.iterrows():
            # Get user ID (RX_ID)
            user_id = str(row['RX_ID'])
            
            # Create location string from X, Y, Z coordinates
            location = f"({row['X']}, {row['Y']}, {row['Z']})"
            
            # Get request, CQI, and ground truth label
            request = row['User_Request']
            cqi = int(row['CQI'])
            
            # Get ground truth label if available
            ground_truth = row.get('Request_Label', None)
            
            user = {
                "user_id": user_id,
                "location": location,
                "request": request,
                "cqi": cqi,
                "ground_truth": ground_truth
            }
            users.append(user)
        
        return users
    except Exception as e:
        print(f"Error loading user data from CSV: {e}")
        # Return some default users as fallback
        return [
            {
                "user_id": "1",
                "location": "(-39.01, -0.50, 1.50)",
                "request": "I need to stream 8K video content",
                "cqi": 15,
                "ground_truth": "eMBB"
            },
            {
                "user_id": "2",
                "location": "(-62.97, 141.28, 1.50)",
                "request": "I want to watch 4K video",
                "cqi": 9,
                "ground_truth": "eMBB"
            }
        ]

# ====================== CSV Results Export Function ======================

def export_results_to_csv(results, slice_stats, intent_stats, file_path="prompt_based_slicing_results.csv"):
    """Export test results to a CSV file with enhanced analytics"""
    try:
        # Define CSV headers
        headers = [
            "User ID", "Allocation Status", "Slice", "Ground Truth", "Intent Correct", "CQI",
            "Bandwidth (MHz)", "Rate (Mbps)", "Latency (ms)", "Adjustments Made",
            "eMBB Total Rate Before (Mbps)", "eMBB Total Rate After (Mbps)",
            "URLLC Total Rate Before (Mbps)", "URLLC Total Rate After (Mbps)",
            "mMTC Total Rate Before (Mbps)", "mMTC Total Rate After (Mbps)",
            "Avg Resource Util Before (%)", "Avg Resource Util After (%)",
            "Token Used", "Prompt Tokens", "Completion Tokens", "LLM Calls",
            "Request"
        ]

        # Open file for writing
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write headers
            writer.writerow(headers)

            # Write data rows
            for result in results:
                allocation_status = "Failed" if result.get("allocation_failed", True) else "Success"

                row = [
                    result["user_id"],
                    allocation_status,
                    result["slice_type"],
                    result.get("ground_truth", "Unknown"),
                    "Yes" if result.get("intent_correct", None) is True else
                    "No" if result.get("intent_correct", None) is False else "N/A",
                    result["cqi"],
                    result.get("bandwidth", "N/A"),
                    result.get("rate", "N/A"),
                    result.get("latency", "N/A"),
                    "Yes" if result.get("adjustments_made", False) else "No",
                    result.get("embb_total_rate_before", "N/A"),
                    result.get("embb_total_rate_after", "N/A"),
                    result.get("urllc_total_rate_before", "N/A"),
                    result.get("urllc_total_rate_after", "N/A"),
                    result.get("mmtc_total_rate_before", "N/A"),
                    result.get("mmtc_total_rate_after", "N/A"),
                    result.get("avg_resource_util_before", "N/A"),
                    result.get("avg_resource_util_after", "N/A"),
                    # Token statistics
                    result.get("token_used", 0),
                    result.get("prompt_tokens", 0),
                    result.get("completion_tokens", 0),
                    result.get("llm_call_count", 0),
                    result["request"]
                ]
                writer.writerow(row)

            # Add empty row as separator
            writer.writerow([])

            # Add summary statistics
            writer.writerow(["SUMMARY STATISTICS"])
            writer.writerow(["Average Resource Utilization (%)", slice_stats["avg_resource_util"]])
            writer.writerow(["Final Resource Utilization (%)", slice_stats["final_resource_util"]])
            writer.writerow(["Final eMBB Total Rate (Mbps)", slice_stats["final_embb_total_rate"]])
            writer.writerow(["Final URLLC Total Rate (Mbps)", slice_stats["final_urllc_total_rate"]])
            writer.writerow(["Final mMTC Total Rate (Mbps)", slice_stats["final_mmtc_total_rate"]])

            # Add intent understanding rate
            writer.writerow([])
            writer.writerow(["INTENT UNDERSTANDING EVALUATION"])
            writer.writerow(["Total Evaluated Requests", intent_stats["total"]])
            writer.writerow(["Correctly Identified Intents", intent_stats["correct"]])
            writer.writerow(["Intent Understanding Rate (%)", intent_stats["rate"]])

        print(f"\nResults exported to {file_path}")
        return True
    except Exception as e:
        print(f"Error exporting results to CSV: {e}")
        return False

# ====================== Global Network State ======================

# Persistent state of network slices (global variable)
GLOBAL_NETWORK_STATE = {
    "embb_slice": {
        "users": [],
        "total_capacity": 90,
        "resource_usage": 0,
        "utilization_rate": "0%"
    },
    "urllc_slice": {
        "users": [],
        "total_capacity": 30,
        "resource_usage": 0,
        "utilization_rate": "0%"
    },
    "mmtc_slice": {
        "users": [],
        "total_capacity": 10,
        "resource_usage": 0,
        "utilization_rate": "0%"
    },
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_users": 0
}

# Store initial network state for reset operations
INITIAL_NETWORK_STATE = copy.deepcopy(GLOBAL_NETWORK_STATE)

# Store network state before each allocation for comparison
PREVIOUS_NETWORK_STATE = None

def get_current_network_state():
    """Get a copy of the current network state"""
    return GLOBAL_NETWORK_STATE.copy()

def update_network_state(new_state):
    """Update global network state"""
    global GLOBAL_NETWORK_STATE, PREVIOUS_NETWORK_STATE
    # Store current state before updating
    PREVIOUS_NETWORK_STATE = GLOBAL_NETWORK_STATE.copy()
    # Update state
    GLOBAL_NETWORK_STATE = new_state
    return True

def reset_network_state():
    """Reset network state to initial values for fresh testing"""
    global GLOBAL_NETWORK_STATE, PREVIOUS_NETWORK_STATE

    # Reset to initial state
    GLOBAL_NETWORK_STATE = INITIAL_NETWORK_STATE.copy()

    # Reset timestamp
    GLOBAL_NETWORK_STATE["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    PREVIOUS_NETWORK_STATE = None

    return True

def calculate_utilization_rate(usage, capacity):
    """Calculate utilization rate percentage"""
    return f"{(usage / capacity * 100):.2f}%"

def calculate_total_transmission_rates():
    """Calculate total transmission rate for each slice"""
    current_state = get_current_network_state()

    embb_total_rate = sum(user["rate"] for user in current_state["embb_slice"]["users"])
    urllc_total_rate = sum(user["rate"] for user in current_state["urllc_slice"]["users"])
    mmtc_total_rate = sum(user["rate"] for user in current_state["mmtc_slice"]["users"])

    return round(embb_total_rate, 2), round(urllc_total_rate, 2), round(mmtc_total_rate, 2)

def calculate_average_resource_utilization():
    """Calculate weighted average resource utilization across all slices"""
    current_state = get_current_network_state()

    embb_usage = current_state["embb_slice"]["resource_usage"]
    embb_capacity = current_state["embb_slice"]["total_capacity"]
    urllc_usage = current_state["urllc_slice"]["resource_usage"]
    urllc_capacity = current_state["urllc_slice"]["total_capacity"]
    mmtc_usage = current_state["mmtc_slice"]["resource_usage"]
    mmtc_capacity = current_state["mmtc_slice"]["total_capacity"]

    total_usage = embb_usage + urllc_usage + mmtc_usage
    total_capacity = embb_capacity + urllc_capacity + mmtc_capacity

    avg_util = (total_usage / total_capacity) * 100 if total_capacity > 0 else 0
    return round(avg_util, 2)

# ====================== CQI-Related Functions ======================

def calculate_rate_from_cqi(bandwidth, cqi):
    """Calculate data rate based on CQI using Shannon's formula

    bandwidth: in MHz
    cqi: Channel Quality Indicator (1-15)
    returns: data rate in Mbps
    """
    snr = 10 ** (cqi / 10)
    rate = bandwidth * math.log10(1 + snr) * 10
    return round(rate, 2)

def apply_heuristic_bandwidth(slice_type, request, min_bandwidth, max_bandwidth):
    """Apply heuristic rules to determine bandwidth based on request type"""
    request_lower = request.lower()

    if slice_type == "eMBB":
        if any(keyword in request_lower for keyword in ["video", "stream", "watch", "movie", "4k", "8k"]):
            return float(min(max_bandwidth, 15.0))
        elif any(keyword in request_lower for keyword in ["download", "file", "upload"]):
            return float(min(max_bandwidth, 12.0))
        elif any(keyword in request_lower for keyword in ["conference", "meeting", "call"]):
            return float(min(max_bandwidth, 10.0))
        elif any(keyword in request_lower for keyword in ["message", "messaging", "chat", "text"]):
            return float(min(max_bandwidth, 8.0))
        else:
            return float(min(max_bandwidth, 8.0))
    elif slice_type == "URLLC":
        if any(keyword in request_lower for keyword in ["surgery", "medical", "emergency"]):
            return float(min(max_bandwidth, 5.0))
        elif any(keyword in request_lower for keyword in ["control", "automation", "robot"]):
            return float(min(max_bandwidth, 3.0))
        else:
            return float(min(max_bandwidth, 2.0))
    else:  # mMTC
        if any(keyword in request_lower for keyword in ["sensor", "meter", "monitor", "tracking", "iot"]):
            return float(min(max_bandwidth, 2))
        else:
            return float(min(max_bandwidth, 1))

def apply_heuristic_latency(slice_type, request, min_latency, max_latency):
    """Apply heuristic rules to determine latency based on request type"""
    request_lower = request.lower()

    if slice_type == "eMBB":
        if any(keyword in request_lower for keyword in ["video", "stream", "watch", "movie", "4k", "8k"]):
            return float(min(max_latency, 50.0))
        elif any(keyword in request_lower for keyword in ["download", "file", "upload"]):
            return float(min(max_latency, 80.0))
        elif any(keyword in request_lower for keyword in ["conference", "meeting", "call"]):
            return float(min(max_latency, 30.0))
        else:
            return float(min(max_latency, 40))
    elif slice_type == "URLLC":
        if any(keyword in request_lower for keyword in ["surgery", "medical", "emergency"]):
            return float(max(min_latency, 1.0))
        elif any(keyword in request_lower for keyword in ["control", "automation", "robot"]):
            return float(max(min_latency, 3.0))
        else:
            return float(max(min_latency, 5.0))
    else:  # mMTC
        if any(keyword in request_lower for keyword in ["sensor", "meter", "tracking"]):
            return float(min(max_latency, 500.0))
        else:
            return float(min(max_latency, 1000.0))

# ====================== Network State Reporting Functions ======================

def generate_concise_report(current_state, new_user_id=None, adjustment_result=None):
    """Generate a concise network status report"""
    embb_slice = current_state["embb_slice"]
    urllc_slice = current_state["urllc_slice"]
    mmtc_slice = current_state["mmtc_slice"]

    stats = [
        ["Slice", "Users", "Resource Usage", "Utilization"],
        ["eMBB", len(embb_slice["users"]), f"{embb_slice['resource_usage']}/{embb_slice['total_capacity']} MHz",
         embb_slice["utilization_rate"]],
        ["URLLC", len(urllc_slice["users"]), f"{urllc_slice['resource_usage']}/{urllc_slice['total_capacity']} MHz",
         urllc_slice["utilization_rate"]],
        ["mMTC", len(mmtc_slice["users"]), f"{mmtc_slice['resource_usage']}/{mmtc_slice['total_capacity']} MHz",
         mmtc_slice["utilization_rate"]]
    ]

    # Calculate total transmission rates
    embb_total_rate, urllc_total_rate, mmtc_total_rate = calculate_total_transmission_rates()

    # Calculate average resource utilization
    avg_resource_util = calculate_average_resource_utilization()

    report = [
        f"Network Status @ {current_state['timestamp']}",
        f"Total Users: {current_state['total_users']}",
        f"Average Resource Utilization: {avg_resource_util}%",
        f"eMBB Total Rate: {embb_total_rate:.2f} Mbps, URLLC Total Rate: {urllc_total_rate:.2f} Mbps, mMTC Total Rate: {mmtc_total_rate:.2f} Mbps",
        "",
        tabulate(stats, headers="firstrow", tablefmt="simple")
    ]

    # If a new user was added, show their details
    if new_user_id:
        new_user = None
        new_user_slice = None

        for slice_key in ["embb_slice", "urllc_slice", "mmtc_slice"]:
            for user in current_state[slice_key]["users"]:
                if user["user_id"] == new_user_id:
                    new_user = user
                    if slice_key == "embb_slice":
                        new_user_slice = "eMBB"
                    elif slice_key == "urllc_slice":
                        new_user_slice = "URLLC"
                    else:
                        new_user_slice = "mMTC"
                    break
            if new_user:
                break

        if new_user:
            report.append("\nNew User Allocation:")
            report.append(f"User {new_user_id} → {new_user_slice} Slice")
            report.append(
                f"CQI: {new_user['cqi']}, Bandwidth: {new_user['bandwidth']} MHz, Rate: {new_user['rate']:.2f} Mbps, Latency: {new_user['latency']} ms")

    return "\n".join(report)

def generate_user_allocation_table(current_state, new_user_id=None, adjusted_user_ids=None):
    """Generate a table showing all user allocations"""
    if adjusted_user_ids is None:
        adjusted_user_ids = []

    all_users = []

    # Add eMBB users
    for user in current_state["embb_slice"]["users"]:
        status = ""
        if user["user_id"] == new_user_id:
            status = "NEW"
        elif user["user_id"] in adjusted_user_ids:
            status = "ADJUSTED"

        all_users.append({
            "user_id": user["user_id"],
            "slice": "eMBB",
            "cqi": user["cqi"],
            "bandwidth": user["bandwidth"],
            "rate": user["rate"],
            "latency": user["latency"],
            "status": status
        })

    # Add URLLC users
    for user in current_state["urllc_slice"]["users"]:
        status = ""
        if user["user_id"] == new_user_id:
            status = "NEW"
        elif user["user_id"] in adjusted_user_ids:
            status = "ADJUSTED"

        all_users.append({
            "user_id": user["user_id"],
            "slice": "URLLC",
            "cqi": user["cqi"],
            "bandwidth": user["bandwidth"],
            "rate": user["rate"],
            "latency": user["latency"],
            "status": status
        })

    # Add mMTC users
    for user in current_state["mmtc_slice"]["users"]:
        status = ""
        if user["user_id"] == new_user_id:
            status = "NEW"
        elif user["user_id"] in adjusted_user_ids:
            status = "ADJUSTED"

        all_users.append({
            "user_id": user["user_id"],
            "slice": "mMTC",
            "cqi": user["cqi"],
            "bandwidth": user["bandwidth"],
            "rate": user["rate"],
            "latency": user["latency"],
            "status": status
        })

    # Sort by slice type and then by user_id
    all_users.sort(key=lambda x: (x["slice"], x["user_id"]))

    # Format data for table
    rows = []
    for user in all_users:
        row = [
            user["user_id"],
            user["slice"],
            user["cqi"],
            f"{user['bandwidth']:.2f}",
            f"{user['rate']:.2f}",
            f"{user['latency']:.2f}",
            user["status"]
        ]
        rows.append(row)

    headers = ["User ID", "Slice", "CQI", "BW (MHz)", "Rate (Mbps)", "Latency (ms)", "Status"]
    table = tabulate(rows, headers=headers, tablefmt="grid")

    return f"\nCurrent User Allocations:\n{table}"


# ====================== Network Slicing Prompt Function ======================

# System prompt for the LLM
SYSTEM_PROMPT = """You are a 5G network slicing expert responsible for allocating users to appropriate network slices and managing resources.

Current network has three types of slices:
- eMBB (Enhanced Mobile Broadband): For high bandwidth applications like video streaming
  * Bandwidth range: 6-20 MHz
  * Data rate range: 100-400 Mbps
  * Latency range: 10-100ms
  * Total capacity: 90 MHz

- URLLC (Ultra-Reliable Low-Latency Communications): For low latency applications like remote control
  * Bandwidth range: 1-5 MHz
  * Data rate range: 1-100 Mbps
  * Latency range: 1-10ms
  * Total capacity: 30 MHz

- mMTC (massive Machine-Type Communications): For large-scale IoT device connectivity
  * Bandwidth range: 1-3 MHz
  * Data rate range: 0.1-1 Mbps
  * Latency range: 100-1000ms
  * Total capacity: 10 MHz

KEY INDICATORS FOR SLICES:
- Words indicating eMBB: stream, download, upload, video, HD, 4K, 8K, movie, watch, gaming, browse
- Words indicating URLLC: control, real-time, monitor, automation, sensors, immediate, mission-critical
- Words indicating mMTC: sensor, meter, tracking, iot, monitoring, telemetry, smart city, environment

IMPORTANT: You must strictly adhere to the resource constraints for each slice type.
The system will verify these constraints after your allocation, and any violation will cause the allocation to fail.

Format your response as JSON with the following fields:
{
  "intent_analysis": "Explanation of user's needs and application type",
  "recommended_slice": "eMBB or URLLC or mMTC",
  "slice_reason": "Explanation for slice choice",
  "bandwidth_allocation": number value in MHz,
  "data_rate": float value in Mbps,
  "latency": number value in ms,
  "workload_balanced": boolean,
  "can_accommodate": boolean
}
"""

def process_user_with_prompt(user_id, location, request, cqi, network_state):
    """Process a user request using the prompt-based approach"""
    embb_slice = network_state["embb_slice"]
    urllc_slice = network_state["urllc_slice"]
    mmtc_slice = network_state["mmtc_slice"]

    embb_usage = embb_slice["resource_usage"]
    urllc_usage = urllc_slice["resource_usage"]
    mmtc_usage = mmtc_slice["resource_usage"]
    embb_capacity = embb_slice["total_capacity"]
    urllc_capacity = urllc_slice["total_capacity"]
    mmtc_capacity = mmtc_slice["total_capacity"]
    embb_utilization = float(embb_slice["utilization_rate"].replace("%", ""))
    urllc_utilization = float(urllc_slice["utilization_rate"].replace("%", ""))
    mmtc_utilization = float(mmtc_slice["utilization_rate"].replace("%", ""))

    prompt = f"""Please allocate network resources for the following user:

USER INFORMATION:
- User ID: {user_id}
- Location: {location}
- Request: "{request}"
- CQI (Channel Quality Indicator): {cqi}

CURRENT NETWORK STATE:
- eMBB Slice:
  * Users: {len(embb_slice["users"])}
  * Resource Usage: {embb_usage}/{embb_capacity} MHz
  * Utilization Rate: {embb_utilization:.2f}%

- URLLC Slice:
  * Users: {len(urllc_slice["users"])}
  * Resource Usage: {urllc_usage}/{urllc_capacity} MHz
  * Utilization Rate: {urllc_utilization:.2f}%

- mMTC Slice:
  * Users: {len(mmtc_slice["users"])}
  * Resource Usage: {mmtc_usage}/{mmtc_capacity} MHz
  * Utilization Rate: {mmtc_utilization:.2f}%

Based on this information:
1. Analyze the user's intent
2. Recommend appropriate network slice (eMBB, URLLC, or mMTC)
3. Allocate bandwidth and calculate data rate
4. Adjust rate if needed to meet slice requirements
5. Consider workload balance between slices
6. Verify capacity availability

IMPORTANT: Remember to strictly adhere to the following constraints:
- eMBB: Bandwidth 6-20 MHz, Rate 100-400 Mbps, Latency 10-100ms
- URLLC: Bandwidth 1-5 MHz, Rate 1-100 Mbps, Latency 1-10ms
- mMTC: Bandwidth 1-3 MHz, Rate 0.1-1 Mbps, Latency 100-1000ms

Provide your response in the JSON format specified in your instructions.
"""

    # Call the LLM and get the response
    response = call_llm(prompt)

    # Parse the JSON response
    try:
        # Remove thinking tags if present
        clean_response = response
        if "<think>" in clean_response:
            # Remove thinking content between <think> and </think>
            while "<think>" in clean_response and "</think>" in clean_response:
                start = clean_response.find("<think>")
                end = clean_response.find("</think>") + len("</think>")
                clean_response = clean_response[:start] + clean_response[end:]

        # Debug: print after thinking tag removal
        print(f"\n[DEBUG] After thinking removal (first 300 chars): {clean_response[:300]}")

        # Remove markdown code block markers if present
        if "```json" in clean_response:
            # Remove the ```json marker and any text before it
            clean_response = clean_response.split("```json")[1]
        if "```" in clean_response:
            # Remove the closing ``` marker and any text after it
            clean_response = clean_response.split("```")[0]

        # Debug: print the cleaned response
        print(f"\n[DEBUG] Clean response (first 400 chars): {clean_response[:400]}")

        # Find JSON block in response
        # Try to find the first { and last } to extract the JSON
        start_idx = clean_response.find('{')
        end_idx = clean_response.rfind('}') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = clean_response[start_idx:end_idx]
            result = json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")

        # Validate that required keys are present
        # The LLM may use different key names, so we map them
        # Expected: intent_analysis, recommended_slice, bandwidth_allocation, data_rate, latency, can_accommodate
        # Or: intent_analysis, recommended_slice, allocated_bandwidth_mhz, calculated_data_rate_mbps, latency_ms, can_accommodate

        # Normalize key names - handle different possible structures
        # Some models return flat keys, others return nested structures
        normalized_result = {
            'intent_analysis': result.get('analysis') or result.get('intent_analysis') or result.get('intent', 'N/A'),
            'recommended_slice': result.get('recommended_slice') or result.get('slice_type') or result.get('slice', 'N/A'),
            'bandwidth': 0,
            'data_rate': 0,
            'latency': 0,
            'slice_reason': result.get('rationale') or result.get('reason') or result.get('slice_reason') or result.get('reasoning', ''),
            'can_accommodate': True,
            'workload_balanced': False
        }

        # Handle top-level slice_selection structure
        if 'slice_selection' in result and isinstance(result['slice_selection'], dict):
            slice_sel = result['slice_selection']
            normalized_result['recommended_slice'] = slice_sel.get('recommended_slice') or normalized_result['recommended_slice']

        # Handle top-level recommendation structure
        if 'recommendation' in result and isinstance(result['recommendation'], dict):
            rec = result['recommendation']
            normalized_result['recommended_slice'] = rec.get('slice') or normalized_result['recommended_slice']

        # Handle top-level allocation structure
        if 'allocation' in result and isinstance(result['allocation'], dict):
            alloc = result['allocation']

            # Extract recommended_slice from nested structures
            if 'slice_recommendation' in alloc and isinstance(alloc['slice_recommendation'], dict):
                normalized_result['recommended_slice'] = alloc['slice_recommendation'].get('recommended_slice') or normalized_result['recommended_slice']

            # Extract from selectedSlice (camelCase)
            if 'selectedSlice' in alloc:
                normalized_result['recommended_slice'] = alloc['selectedSlice'] or normalized_result['recommended_slice']

            # Extract from slice field
            if 'slice' in alloc:
                normalized_result['recommended_slice'] = alloc['slice'] or normalized_result['recommended_slice']

            # Extract intent_analysis from nested structures
            if 'intent_analysis' in alloc:
                normalized_result['intent_analysis'] = alloc['intent_analysis']

            # Extract from final_recommendation (most accurate)
            if 'final_recommendation' in alloc and isinstance(alloc['final_recommendation'], dict):
                final_rec = alloc['final_recommendation']
                normalized_result['recommended_slice'] = final_rec.get('slice') or normalized_result['recommended_slice']
                normalized_result['bandwidth'] = final_rec.get('bandwidth_mhz') or normalized_result['bandwidth']
                normalized_result['data_rate'] = final_rec.get('data_rate_mbps') or normalized_result['data_rate']
                normalized_result['latency'] = final_rec.get('latency_ms') or normalized_result['latency']

            # Extract from rate_adjustment if available
            if 'rate_adjustment' in alloc and isinstance(alloc['rate_adjustment'], dict):
                rate_adj = alloc['rate_adjustment']
                if rate_adj.get('adjustment_required'):
                    normalized_result['bandwidth'] = rate_adj.get('adjusted_bandwidth_mhz') or normalized_result['bandwidth']
                    normalized_result['data_rate'] = rate_adj.get('adjusted_data_rate_mbps') or normalized_result['data_rate']

            # Extract from resource_allocation
            if 'resource_allocation' in alloc and isinstance(alloc['resource_allocation'], dict):
                res_alloc = alloc['resource_allocation']
                if normalized_result['bandwidth'] == 0:
                    normalized_result['bandwidth'] = res_alloc.get('bandwidth_assigned_mhz') or res_alloc.get('allocated_bandwidth_mhz') or 0
                if normalized_result['data_rate'] == 0:
                    normalized_result['data_rate'] = res_alloc.get('target_data_rate_mbps') or res_alloc.get('calculated_data_rate_mbps') or 0
                if normalized_result['latency'] == 0:
                    normalized_result['latency'] = res_alloc.get('latency_allocated_ms') or res_alloc.get('latency_estimate_ms') or 0

            # Extract from final_assignment (most accurate final values)
            if 'final_assignment' in alloc and isinstance(alloc['final_assignment'], dict):
                final_assign = alloc['final_assignment']
                normalized_result['recommended_slice'] = final_assign.get('slice') or normalized_result['recommended_slice']
                if normalized_result['bandwidth'] == 0:
                    normalized_result['bandwidth'] = final_assign.get('bandwidth_mhz') or 0
                if normalized_result['data_rate'] == 0:
                    normalized_result['data_rate'] = final_assign.get('max_data_rate_mbps') or final_assign.get('data_rate_mbps') or 0
                if normalized_result['latency'] == 0:
                    normalized_result['latency'] = final_assign.get('latency_ms') or 0

            # Extract from camelCase fields (common pattern)
            if normalized_result['bandwidth'] == 0:
                normalized_result['bandwidth'] = alloc.get('bandwidth_MHz') or alloc.get('bandwidthMHz') or alloc.get('allocatedBandwidthMHz') or alloc.get('allocated_bandwidth_MHz') or alloc.get('bandwidth_mhz') or alloc.get('bandwidth', 0)
            if normalized_result['data_rate'] == 0:
                normalized_result['data_rate'] = alloc.get('estimatedDataRateMbps') or alloc.get('estimated_data_rate_Mbps') or alloc.get('estimated_net_data_rate_Mbps') or alloc.get('estimated_gross_data_rate_Mbps') or alloc.get('data_rate_mbps') or alloc.get('data_rate', 0)
            if normalized_result['latency'] == 0:
                normalized_result['latency'] = alloc.get('latency_assumption_ms') or alloc.get('latencyMs') or alloc.get('latency_assigned_ms') or alloc.get('latency_ms') or alloc.get('latency', 0)

        # Handle top-level resource_allocation structure
        if 'resource_allocation' in result and isinstance(result['resource_allocation'], dict):
            res_alloc = result['resource_allocation']

            # Extract from slice_recommendation
            if 'slice_recommendation' in res_alloc and isinstance(res_alloc['slice_recommendation'], dict):
                normalized_result['recommended_slice'] = res_alloc['slice_recommendation'].get('recommended_slice') or normalized_result['recommended_slice']

            normalized_result['recommended_slice'] = res_alloc.get('slice') or normalized_result['recommended_slice']

            # Extract from final_allocation_summary (most accurate)
            if 'final_allocation_summary' in res_alloc and isinstance(res_alloc['final_allocation_summary'], dict):
                final_summary = res_alloc['final_allocation_summary']
                normalized_result['recommended_slice'] = final_summary.get('slice') or normalized_result['recommended_slice']
                if normalized_result['bandwidth'] == 0:
                    normalized_result['bandwidth'] = final_summary.get('bandwidth_mhz') or 0
                if normalized_result['data_rate'] == 0:
                    normalized_result['data_rate'] = final_summary.get('guaranteed_data_rate_mbps') or final_summary.get('final_allocated_rate_mbps') or 0
                if normalized_result['latency'] == 0:
                    normalized_result['latency'] = final_summary.get('estimated_latency_ms') or 0

            # Extract from bandwidth_allocation
            if 'bandwidth_allocation' in res_alloc and isinstance(res_alloc['bandwidth_allocation'], dict):
                bw_alloc = res_alloc['bandwidth_allocation']
                if normalized_result['bandwidth'] == 0:
                    normalized_result['bandwidth'] = bw_alloc.get('allocated_bandwidth_mhz') or 0

            # Extract from data_rate_calculation
            if 'data_rate_calculation' in res_alloc and isinstance(res_alloc['data_rate_calculation'], dict):
                rate_calc = res_alloc['data_rate_calculation']
                if normalized_result['data_rate'] == 0:
                    normalized_result['data_rate'] = rate_calc.get('final_allocated_rate_mbps') or rate_calc.get('adjusted_data_rate_mbps') or rate_calc.get('calculated_data_rate_mbps') or 0

            # Extract from latency_estimate
            if 'latency_estimate' in res_alloc and isinstance(res_alloc['latency_estimate'], dict):
                lat_est = res_alloc['latency_estimate']
                if normalized_result['latency'] == 0:
                    normalized_result['latency'] = lat_est.get('estimated_latency_ms') or 0

            # Fallback to top-level fields
            if normalized_result['bandwidth'] == 0:
                normalized_result['bandwidth'] = res_alloc.get('bandwidth_MHz') or res_alloc.get('bandwidth_allocated_mhz') or res_alloc.get('bandwidth_assigned_mhz') or res_alloc.get('allocated_bandwidth_mhz') or 0
            if normalized_result['data_rate'] == 0:
                normalized_result['data_rate'] = res_alloc.get('estimated_data_rate_Mbps') or res_alloc.get('calculated_data_rate_mbps') or res_alloc.get('target_data_rate_mbps') or 0
            if normalized_result['latency'] == 0:
                normalized_result['latency'] = res_alloc.get('latency_assumption_ms') or res_alloc.get('actual_latency_ms') or res_alloc.get('target_latency_ms') or res_alloc.get('latency_ms') or res_alloc.get('latency_assigned_ms') or res_alloc.get('latency_estimate_ms') or 0

        # Handle top-level final_allocation structure
        if 'final_allocation' in result and isinstance(result['final_allocation'], dict):
            final_alloc = result['final_allocation']
            normalized_result['recommended_slice'] = final_alloc.get('slice_type') or normalized_result['recommended_slice']
            if normalized_result['bandwidth'] == 0:
                normalized_result['bandwidth'] = final_alloc.get('bandwidth_mhz') or 0
            if normalized_result['data_rate'] == 0:
                normalized_result['data_rate'] = final_alloc.get('guaranteed_rate_mbps') or final_alloc.get('max_rate_mbps') or final_alloc.get('data_rate_mbps') or 0
            if normalized_result['latency'] == 0:
                normalized_result['latency'] = final_alloc.get('latency_ms') or 0

        # Handle top-level rate_adjustment structure
        if 'rate_adjustment' in result and isinstance(result['rate_adjustment'], dict):
            rate_adj = result['rate_adjustment']
            if rate_adj.get('adjustment_required'):
                if normalized_result['bandwidth'] == 0:
                    normalized_result['bandwidth'] = rate_adj.get('adjusted_bandwidth_mhz') or normalized_result['bandwidth']
                if normalized_result['data_rate'] == 0:
                    normalized_result['data_rate'] = rate_adj.get('adjusted_rate_mbps') or normalized_result['data_rate']

        # Extract recommended_slice from nested structures (top level)
        if 'slice_recommendation' in result and isinstance(result['slice_recommendation'], dict):
            normalized_result['recommended_slice'] = result['slice_recommendation'].get('selected_slice') or result['slice_recommendation'].get('recommended_slice') or normalized_result['recommended_slice']

        # Extract bandwidth from nested structures
        if 'bandwidth_allocation' in result:
            bw_alloc = result['bandwidth_allocation']
            if isinstance(bw_alloc, dict):
                normalized_result['bandwidth'] = bw_alloc.get('allocated_bandwidth_mhz') or bw_alloc.get('bandwidth_mhz') or bw_alloc.get('allocated_bandwidth') or bw_alloc.get('bandwidth', 0)
            else:
                normalized_result['bandwidth'] = bw_alloc
        elif 'allocation' in result and isinstance(result['allocation'], dict):
            if normalized_result['bandwidth'] == 0:
                normalized_result['bandwidth'] = result['allocation'].get('bandwidth_mhz') or result['allocation'].get('allocated_bandwidth_mhz', 0)
        else:
            if normalized_result['bandwidth'] == 0:
                normalized_result['bandwidth'] = result.get('allocated_bandwidth_mhz') or result.get('bandwidth_allocation') or result.get('bandwidth', 0)

        # Extract data_rate from nested structures
        if 'data_rate_calculation' in result:
            rate_calc = result['data_rate_calculation']
            if isinstance(rate_calc, dict):
                normalized_result['data_rate'] = rate_calc.get('adjusted_rate_mbps') or rate_calc.get('calculated_rate_mbps') or rate_calc.get('rate', 0)
            else:
                normalized_result['data_rate'] = rate_calc
        elif 'allocation' in result and isinstance(result['allocation'], dict):
            if normalized_result['data_rate'] == 0:
                normalized_result['data_rate'] = result['allocation'].get('estimated_data_rate_mbps') or result['allocation'].get('adjusted_rate_mbps') or result['allocation'].get('calculated_rate_mbps') or result['allocation'].get('rate', 0)
        else:
            if normalized_result['data_rate'] == 0:
                normalized_result['data_rate'] = result.get('allocated_rate_mbps') or result.get('calculated_data_rate_mbps') or result.get('data_rate') or result.get('rate', 0)

        # Extract latency from nested structures
        if 'latency_allocation' in result:
            lat_alloc = result['latency_allocation']
            if isinstance(lat_alloc, dict):
                normalized_result['latency'] = lat_alloc.get('estimated_latency_ms') or lat_alloc.get('latency', 0)
            else:
                normalized_result['latency'] = lat_alloc
        elif 'allocation' in result and isinstance(result['allocation'], dict):
            if normalized_result['latency'] == 0:
                normalized_result['latency'] = result['allocation'].get('latency_target_ms') or result['allocation'].get('expected_latency_ms') or result['allocation'].get('latency_estimate_ms') or result['allocation'].get('latency', 0)
        else:
            if normalized_result['latency'] == 0:
                normalized_result['latency'] = result.get('latency_ms') or result.get('latency', 0)

        # Convert to float if they are not already
        try:
            normalized_result['bandwidth'] = float(normalized_result['bandwidth'])
        except (ValueError, TypeError):
            normalized_result['bandwidth'] = 0

        try:
            normalized_result['data_rate'] = float(normalized_result['data_rate'])
        except (ValueError, TypeError):
            normalized_result['data_rate'] = 0

        try:
            normalized_result['latency'] = float(normalized_result['latency'])
        except (ValueError, TypeError):
            normalized_result['latency'] = 0

        # Debug: print what we got
        print(f"\n[DEBUG] Raw result: {result}")
        print(f"\n[DEBUG] Normalized bandwidth: {normalized_result['bandwidth']}, rate: {normalized_result['data_rate']}")
        print(f"\nIntent Analysis: {normalized_result['intent_analysis']}")
        print(f"Recommended Slice: {normalized_result['recommended_slice']} - {normalized_result['slice_reason']}")
        print(f"Bandwidth Allocation: {normalized_result['bandwidth']} MHz")
        print(f"Data Rate: {normalized_result['data_rate']} Mbps")
        print(f"Latency: {normalized_result['latency']} ms")
        print(f"Workload Balanced: {'Yes' if normalized_result['workload_balanced'] else 'No'}")

        slice_type = normalized_result['recommended_slice']
        bandwidth = normalized_result['bandwidth']
        rate = normalized_result['data_rate']
        latency = normalized_result['latency']

        # Map slice type to slice key
        if slice_type == "eMBB":
            slice_key = "embb_slice"
        elif slice_type == "URLLC":
            slice_key = "urllc_slice"
        else:  # mMTC
            slice_key = "mmtc_slice"

        # If the LLM indicates we can accommodate the user
        if normalized_result.get("can_accommodate", False):
            # Check capacity availability
            available_capacity = network_state[slice_key]["total_capacity"] - network_state[slice_key]["resource_usage"]

            if available_capacity < bandwidth:
                print(f"\nCAPACITY CHECK FAILED:")
                print(f"- Required: {bandwidth} MHz, Available: {available_capacity} MHz in {slice_type} slice")

                return {
                    "user_id": user_id,
                    "request": request,
                    "cqi": cqi,
                    "slice_type": slice_type,
                    "bandwidth": bandwidth,
                    "rate": rate,
                    "latency": latency,
                    "allocation_failed": True,
                    "intent_analysis": normalized_result["intent_analysis"],
                    "slice_reason": normalized_result["slice_reason"],
                    "failure_reason": f"Insufficient capacity in {slice_type} slice. Required: {bandwidth} MHz, Available: {available_capacity} MHz"
                }

            # All checks passed, proceed with allocation
            new_user = {
                "user_id": user_id,
                "rate": rate,
                "latency": latency,
                "cqi": cqi,
                "bandwidth": bandwidth
            }

            # Add user and update resource usage
            network_state[slice_key]["users"].append(new_user)
            network_state[slice_key]["resource_usage"] += bandwidth

            # Update utilization rate
            network_state[slice_key]["utilization_rate"] = calculate_utilization_rate(
                network_state[slice_key]["resource_usage"],
                network_state[slice_key]["total_capacity"]
            )

            # Update total user count
            network_state["total_users"] = (
                len(network_state["embb_slice"]["users"]) +
                len(network_state["urllc_slice"]["users"]) +
                len(network_state["mmtc_slice"]["users"])
            )

            # Update timestamp
            network_state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Update global state
            update_network_state(network_state)

            print(f"\nALLOCATION SUCCESSFUL: All constraints satisfied")

            return {
                "user_id": user_id,
                "request": request,
                "cqi": cqi,
                "slice_type": slice_type,
                "bandwidth": bandwidth,
                "rate": rate,
                "latency": latency,
                "allocation_failed": False,
                "intent_analysis": normalized_result["intent_analysis"],
                "slice_reason": normalized_result["slice_reason"],
                "workload_balanced": normalized_result.get("workload_balanced", False),
                "adjustments_made": False
            }
        else:
            return {
                "user_id": user_id,
                "request": request,
                "cqi": cqi,
                "slice_type": slice_type,
                "bandwidth": bandwidth,
                "rate": rate,
                "latency": latency,
                "allocation_failed": True,
                "intent_analysis": normalized_result["intent_analysis"],
                "slice_reason": normalized_result["slice_reason"],
                "failure_reason": "LLM determined it cannot accommodate user based on current network state"
            }

    except (json.JSONDecodeError, UnicodeEncodeError, Exception) as e:
        try:
            error_msg = f"Error parsing LLM response: {e}"
            print(error_msg)
        except:
            print(f"Error parsing LLM response (could not display error message due to encoding)")
        try:
            # Try to print a truncated version avoiding encoding issues
            safe_response = response[:500] if len(response) > 500 else response
            # Try to encode to handle special characters
            try:
                safe_response = safe_response.encode('utf-8', errors='replace').decode('utf-8')
            except:
                pass
            print(f"Raw response (truncated): {safe_response}...")
        except:
            print("Raw response could not be printed due to encoding issues")

        return {
            "user_id": user_id,
            "request": request,
            "cqi": cqi,
            "slice_type": "Failed",
            "allocation_failed": True,
            "error": str(e)
        }

# ====================== Main Process Function ======================

def process_user_request(user_id, location, request, cqi, ground_truth=None):
    """Main function for processing user requests with prompt-based approach"""
    # Get current network state
    network_state = get_current_network_state()

    # Record initial state for comparison
    initial_embb_usage = network_state["embb_slice"]["resource_usage"]
    initial_urllc_usage = network_state["urllc_slice"]["resource_usage"]
    initial_mmtc_usage = network_state["mmtc_slice"]["resource_usage"]
    initial_embb_util = network_state["embb_slice"]["utilization_rate"]
    initial_urllc_util = network_state["urllc_slice"]["utilization_rate"]
    initial_mmtc_util = network_state["mmtc_slice"]["utilization_rate"]

    # Calculate initial rates
    initial_embb_total_rate, initial_urllc_total_rate, initial_mmtc_total_rate = calculate_total_transmission_rates()
    initial_avg_resource_util = calculate_average_resource_utilization()

    # Process user with the prompt-based approach
    result = process_user_with_prompt(user_id, location, request, cqi, network_state)

    # Get updated network state
    updated_state = get_current_network_state()

    # Print concise report and user allocation table
    if not result.get("allocation_failed", True):
        print("\n" + "-" * 40)
        print(f"ALLOCATION RESULT FOR USER {user_id}")
        print("-" * 40)
        concise_report = generate_concise_report(updated_state, user_id)
        print(concise_report)

        # Print complete user allocation table
        user_table = generate_user_allocation_table(updated_state, user_id)
        print(user_table)
    else:
        print("\n" + "-" * 40)
        print(f"ALLOCATION FAILED FOR USER {user_id}")
        print("-" * 40)
        print(f"Request: {request}")
        print(f"Slice type: {result.get('slice_type', 'Unknown')}")
        print(f"Reason: {result.get('failure_reason', 'Unknown error')}")

    # Add comparison data
    final_embb_usage = updated_state["embb_slice"]["resource_usage"]
    final_urllc_usage = updated_state["urllc_slice"]["resource_usage"]
    final_mmtc_usage = updated_state["mmtc_slice"]["resource_usage"]
    final_embb_util = updated_state["embb_slice"]["utilization_rate"]
    final_urllc_util = updated_state["urllc_slice"]["utilization_rate"]
    final_mmtc_util = updated_state["mmtc_slice"]["utilization_rate"]

    # Calculate final rates
    final_embb_total_rate, final_urllc_total_rate, final_mmtc_total_rate = calculate_total_transmission_rates()
    final_avg_resource_util = calculate_average_resource_utilization()

    result["embb_usage_before"] = initial_embb_usage
    result["embb_usage_after"] = final_embb_usage
    result["embb_util_before"] = initial_embb_util
    result["embb_util_after"] = final_embb_util
    result["urllc_usage_before"] = initial_urllc_usage
    result["urllc_usage_after"] = final_urllc_usage
    result["urllc_util_before"] = initial_urllc_util
    result["urllc_util_after"] = final_urllc_util
    result["mmtc_usage_before"] = initial_mmtc_usage
    result["mmtc_usage_after"] = final_mmtc_usage
    result["mmtc_util_before"] = initial_mmtc_util
    result["mmtc_util_after"] = final_mmtc_util

    # Add rate tracking
    result["embb_total_rate_before"] = initial_embb_total_rate
    result["embb_total_rate_after"] = final_embb_total_rate
    result["urllc_total_rate_before"] = initial_urllc_total_rate
    result["urllc_total_rate_after"] = final_urllc_total_rate
    result["mmtc_total_rate_before"] = initial_mmtc_total_rate
    result["mmtc_total_rate_after"] = final_mmtc_total_rate
    result["avg_resource_util_before"] = initial_avg_resource_util
    result["avg_resource_util_after"] = final_avg_resource_util

    # Check if intent understanding matches ground truth
    if ground_truth is not None and not result.get("allocation_failed", True):
        result["intent_correct"] = (result["slice_type"] == ground_truth)
    else:
        result["intent_correct"] = None

    result["ground_truth"] = ground_truth

    return result

# ====================== Main Function ======================

def main(num_users=4, export_file="prompt_based_slicing_results.csv", ray_tracing_csv=None):
    """Main program with CSV-based user testing

    Parameters:
    - num_users: Number of users to test (default: 4)
    - export_file: Path to export results CSV file
    - ray_tracing_csv: Path to ray tracing CSV file
    """
    print("Starting prompt-based network slice management system...\n")

    # Reset token statistics at the start
    reset_token_stats()

    # Load users from CSV (limit to specified number)
    if ray_tracing_csv:
        users = load_user_data_from_csv(ray_tracing_csv, num_users)
    else:
        # Use default fallback users if no CSV provided
        users = load_user_data_from_csv(r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_east.csv", num_users)

    print(f"Testing {len(users)} users from ray tracing results CSV")

    # Initialize results tracker
    detailed_results = []

    # Track slice utilization and constraint violations
    embb_utils = []
    urllc_utils = []
    mmtc_utils = []

    workload_balanced_count = 0

    # Process each user
    for i, user in enumerate(users):
        print(f"\n{'-' * 140}")
        print(f"PROCESSING USER {user['user_id']} ({i + 1}/{len(users)})")
        print(f"Request: \"{user['request']}\"")
        print(f"CQI: {user['cqi']}")
        if user.get('ground_truth'):
            print(f"Ground Truth Slice: {user['ground_truth']}")
        print(f"{'-' * 140}")

        # Reset network state for clean testing
        reset_network_state()

        # Process the user
        result = process_user_request(
            user_id=user['user_id'],
            location=user['location'],
            request=user['request'],
            cqi=user['cqi'],
            ground_truth=user.get('ground_truth')
        )

        # Get token usage for this user
        current_tokens = get_token_usage()
        result["token_used"] = current_tokens["total_tokens"]
        result["prompt_tokens"] = current_tokens["total_prompt_tokens"]
        result["completion_tokens"] = current_tokens["total_completion_tokens"]
        result["llm_call_count"] = current_tokens["llm_call_count"]

        # Store detailed result for CSV export
        detailed_results.append(result)

        # Track workload balancing
        if result.get("workload_balanced", False):
            workload_balanced_count += 1

        # Track slice utilization
        if not result.get("allocation_failed", True):
            try:
                if "embb_util_after" in result:
                    embb_util_str = result["embb_util_after"]
                    if isinstance(embb_util_str, str):
                        embb_util = float(embb_util_str.replace("%", ""))
                    else:
                        embb_util = float(embb_util_str)
                    embb_utils.append(embb_util)

                if "urllc_util_after" in result:
                    urllc_util_str = result["urllc_util_after"]
                    if isinstance(urllc_util_str, str):
                        urllc_util = float(urllc_util_str.replace("%", ""))
                    else:
                        urllc_util = float(urllc_util_str)
                    urllc_utils.append(urllc_util)

                if "mmtc_util_after" in result:
                    mmtc_util_str = result["mmtc_util_after"]
                    if isinstance(mmtc_util_str, str):
                        mmtc_util = float(mmtc_util_str.replace("%", ""))
                    else:
                        mmtc_util = float(mmtc_util_str)
                    mmtc_utils.append(mmtc_util)
            except (ValueError, AttributeError):
                pass

    # Get final total transmission rates
    final_embb_total_rate, final_urllc_total_rate, final_mmtc_total_rate = calculate_total_transmission_rates()

    # Get final average resource utilization
    final_avg_resource_util = calculate_average_resource_utilization()

    # Calculate average slice utilization
    avg_embb_util = sum(embb_utils) / len(embb_utils) if embb_utils else 0
    avg_urllc_util = sum(urllc_utils) / len(urllc_utils) if urllc_utils else 0
    avg_mmtc_util = sum(mmtc_utils) / len(mmtc_utils) if mmtc_utils else 0

    # Calculate intent understanding rate
    correct_intents = 0
    total_evaluated = 0

    for result in detailed_results:
        if result.get("intent_correct") is not None:
            total_evaluated += 1
            if result["intent_correct"]:
                correct_intents += 1

    intent_rate = 0 if total_evaluated == 0 else (correct_intents / total_evaluated) * 100

    # Calculate workload balancing rate
    workload_balanced_rate = 0 if len(detailed_results) == 0 else (workload_balanced_count / len(detailed_results)) * 100

    # Print summary of all results
    print("\n" + "=" * 60)
    print("SUMMARY OF USER ALLOCATIONS")
    print("=" * 60)

    # Generate summary table
    summary_rows = []
    for res in detailed_results:
        status = "Success" if not res.get("allocation_failed", True) else "Failed"
        intent_match = ""
        if res.get("intent_correct") is not None:
            intent_match = "Yes" if res["intent_correct"] else "No"

        summary_rows.append([
            res["user_id"],
            status,
            res.get("slice_type", "Failed"),
            res.get("ground_truth", "N/A"),
            intent_match,
            res["cqi"],
            res.get("bandwidth", "N/A"),
            f"{res.get('rate', 'N/A')}" if res.get("rate") is not None else "N/A",
            res.get("latency", "N/A"),
            "Yes" if res.get("adjustments_made", False) else "No"
        ])

    headers = ["User ID", "Status", "Slice", "Ground Truth", "Intent Match", "CQI", "BW (MHz)", "Rate (Mbps)", "Latency (ms)", "Adjusted"]
    summary_table = tabulate(summary_rows, headers=headers, tablefmt="grid")
    print(summary_table)

    # Print statistics
    success_count = sum(1 for res in detailed_results if not res.get("allocation_failed", True))
    total_count = len(detailed_results)
    embb_count = sum(1 for res in detailed_results if res.get("slice_type") == "eMBB")
    urllc_count = sum(1 for res in detailed_results if res.get("slice_type") == "URLLC")
    mmtc_count = sum(1 for res in detailed_results if res.get("slice_type") == "mMTC")

    print("\nStatistics:")
    print(f"Success rate: {success_count}/{total_count} ({success_count / total_count * 100:.1f}%)")

    # Print intent understanding statistics
    print("\nIntent Understanding Evaluation:")
    print(f"Correctly identified intents: {correct_intents}/{total_evaluated}")
    print(f"Intent understanding rate: {intent_rate:.1f}%")

    # Print workload balancing statistics
    print("\nWorkload Balancing Statistics:")
    print(f"Users with workload balancing: {workload_balanced_count}/{total_count}")
    print(f"Workload balancing rate: {workload_balanced_rate:.1f}%")

    # Print slice utilization statistics
    print("\nSlice Utilization Statistics:")
    print(f"Average eMBB utilization: {avg_embb_util:.2f}%")
    print(f"Average URLLC utilization: {avg_urllc_util:.2f}%")
    print(f"Average mMTC utilization: {avg_mmtc_util:.2f}%")

    # Prepare data for CSV export
    slice_stats = {
        "avg_resource_util": f"{final_avg_resource_util:.2f}",
        "final_resource_util": f"{final_avg_resource_util:.2f}",
        "final_embb_total_rate": final_embb_total_rate,
        "final_urllc_total_rate": final_urllc_total_rate,
        "final_mmtc_total_rate": final_mmtc_total_rate
    }

    intent_stats = {
        "total": total_evaluated,
        "correct": correct_intents,
        "rate": f"{intent_rate:.2f}"
    }

    # Export results to CSV
    export_results_to_csv(detailed_results, slice_stats, intent_stats, export_file)

if __name__ == "__main__":
    # Test with 1 user for verification
    main(num_users=1, export_file="prompt_based_slicing_results.csv")