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

    print(
        f"[Token] {operation_name}: prompt~{actual_prompt_tokens}, completion~{actual_completion_tokens}, total~{actual_total_tokens}")

    return response


# LLM Configuration
from llm_config import get_llm

# Default model name (can be changed via run_all_scenarios_prompt_based.py)
MODEL_NAME = "glm-5"

# Get LLM instance
llm = get_llm(MODEL_NAME)


def call_llm(prompt, operation_name="Prompt_Based"):
    """Call the LLM API with the given prompt using configured model, with token tracking"""
    try:
        # Wrap prompt in HumanMessage for token tracking
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content=prompt)]
        response = llm_with_token_tracking(llm, messages, operation_name)
        return response.content if response else None
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None
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
    avg_resource_util = calculate_average_resource_utilization()

    report = [
        f"Network Status @ {current_state['timestamp']}",
        f"Total Users: {current_state['total_users']}",
        f"Average Resource Utilization: {avg_resource_util}%",
        f"eMBB Total Rate: {embb_total_rate:.2f} Mbps, URLLC Total Rate: {urllc_total_rate:.2f} Mbps, mMTC Total Rate: {mmtc_total_rate:.2f} Mbps",
        "",
        tabulate(stats, headers="firstrow", tablefmt="simple")
    ]

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

    for user in current_state["embb_slice"]["users"]:
        status = "NEW" if user["user_id"] == new_user_id else (
            "ADJUSTED" if user["user_id"] in adjusted_user_ids else "")
        all_users.append(
            {"user_id": user["user_id"], "slice": "eMBB", "cqi": user["cqi"], "bandwidth": user["bandwidth"],
             "rate": user["rate"], "latency": user["latency"], "status": status})

    for user in current_state["urllc_slice"]["users"]:
        status = "NEW" if user["user_id"] == new_user_id else (
            "ADJUSTED" if user["user_id"] in adjusted_user_ids else "")
        all_users.append(
            {"user_id": user["user_id"], "slice": "URLLC", "cqi": user["cqi"], "bandwidth": user["bandwidth"],
             "rate": user["rate"], "latency": user["latency"], "status": status})

    for user in current_state["mmtc_slice"]["users"]:
        status = "NEW" if user["user_id"] == new_user_id else (
            "ADJUSTED" if user["user_id"] in adjusted_user_ids else "")
        all_users.append(
            {"user_id": user["user_id"], "slice": "mMTC", "cqi": user["cqi"], "bandwidth": user["bandwidth"],
             "rate": user["rate"], "latency": user["latency"], "status": status})

    all_users.sort(key=lambda x: (x["slice"], x["user_id"]))

    rows = []
    for user in all_users:
        rows.append([user["user_id"], user["slice"], user["cqi"], f"{user['bandwidth']:.2f}", f"{user['rate']:.2f}",
                     f"{user['latency']:.2f}", user["status"]])

    headers = ["User ID", "Slice", "CQI", "BW (MHz)", "Rate (Mbps)", "Latency (ms)", "Status"]
    table = tabulate(rows, headers=headers, tablefmt="grid")

    return f"\nCurrent User Allocations:\n{table}"


# ====================== Network Slicing Prompt Function ======================

# System prompt for the LLM
SYSTEM_PROMPT = """You are a 5G network slicing expert responsible for allocating users to appropriate network slices and managing resources. You need to perform the following sequential tasks:

1. INTENT UNDERSTANDING: Analyze the user's request to understand their application needs.

2. SLICE RECOMMENDATION: Recommend the most appropriate network slice:
   - eMBB (Enhanced Mobile Broadband): For high bandwidth applications like video streaming, AR/VR, and large file downloads.
     * Bandwidth range: 6-20 MHz
     * Data rate range: 100-400 Mbps
     * Latency range: 10-100ms
   - URLLC (Ultra-Reliable Low-Latency Communications): For low latency and high reliability applications like remote control, autonomous driving, and industrial automation.
     * Bandwidth range: 1-5 MHz
     * Data rate range: 1-100 Mbps
     * Latency range: 1-10ms
   - mMTC (massive Machine Type Communications): For massive IoT applications like smart meters, environmental sensors, and fleet tracking (low power, infrequent small data transmissions).
     * Bandwidth range: 0.1-1 MHz
     * Data rate range: 0.01-1 Mbps
     * Latency range: 100-1000ms

3. RATE ALLOCATION: Allocate bandwidth and calculate data rate based on the user's CQI (Channel Quality Indicator) and typical requirements:
   - CQI ranges from 1-15 (higher is better signal quality)
   - For eMBB and URLLC, a rough estimate for data rate is:
     * Low CQI (1-5): bandwidth × 5 Mbps
     * Medium CQI (6-10): bandwidth × 10 Mbps
     * High CQI (11-15): bandwidth × 15 Mbps
   - For mMTC, due to device hardware limitations, data rate is primarily determined by CQI rather than bandwidth:
     * Low CQI (1-5): 0.05 Mbps
     * Medium CQI (6-10): 0.1 Mbps
     * High CQI (11-15): 0.5 Mbps

4. RATE ADJUSTMENT: Adjust bandwidth if needed to meet slice requirements:
   - For eMBB: Ensure rate is between 100-400 Mbps
   - For URLLC: Ensure rate is between 1-100 Mbps
   - For mMTC: Ensure rate is between 0.01-1 Mbps
   - Adjust bandwidth up or down to meet these requirements.

5. WORKLOAD BALANCE: If one slice is significantly more utilized than the others (>20% difference) and the user could be accommodated in multiple slices, prefer the less utilized slice.

6. CAPACITY CHECK: Verify if the slice has enough capacity for the new user:
   - eMBB total capacity: 90 MHz
   - URLLC total capacity: 30 MHz
   - mMTC total capacity: 20 MHz
   - If not enough capacity, report failure.

IMPORTANT: You must strictly adhere to the resource constraints:
- eMBB: Bandwidth 6-20 MHz, Rate 100-400 Mbps, Latency 10-100ms
- URLLC: Bandwidth 1-5 MHz, Rate 1-100 Mbps, Latency 1-10ms
- mMTC: Bandwidth 0.1-1 MHz, Rate 0.01-1 Mbps, Latency 100-1000ms
The system will verify these constraints after your allocation, and any violation will cause the allocation to fail.

Format your response as JSON with the following fields:
{
  "intent_analysis": "Explanation of user's needs and application type",
  "recommended_slice": "eMBB, URLLC, or mMTC",
  "slice_reason": "Explanation for slice choice",
  "bandwidth_allocation": float value in MHz,
  "data_rate": float value in Mbps,
  "latency": integer value in ms,
  "workload_balanced": boolean,
  "can_accommodate": boolean,
  "final_allocation": {
    "user_id": "string",
    "slice_type": "eMBB, URLLC, or mMTC",
    "bandwidth": float in MHz,
    "rate": float in Mbps,
    "latency": integer in ms
  }
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

    response = call_llm(prompt)

    try:
        # Clean response
        clean_response = response
        if "<think>" in clean_response:
            while "<think>" in clean_response and "</think>" in clean_response:
                start = clean_response.find("<think>")
                end = clean_response.find("</think>") + len("</think>")
                clean_response = clean_response[:start] + clean_response[end:]

        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', clean_response, re.IGNORECASE | re.DOTALL)
        if json_match:
            clean_response = json_match.group(1)

        start_idx = clean_response.find('{')
        end_idx = clean_response.rfind('}') + 1

        if start_idx != -1 and end_idx > start_idx:
            json_str = clean_response[start_idx:end_idx]
            result = json.loads(json_str)
        else:
            raise ValueError("No JSON found in response")

        if not isinstance(result, dict):
            raise ValueError("Parsed JSON is not a dictionary")

        # ======================================================================
        #  AST-LIKE RECURSIVE JSON PARSER
        # ======================================================================

        def extract_number(val):
            if isinstance(val, (int, float)): return float(val)
            if isinstance(val, str):
                # 兼容科学计数法，例如 5e6
                m = re.findall(r"[-+]?(?:\d*\.*\d+)(?:[eE][-+]?\d+)?", val)
                if m: return float(m[0])
            if isinstance(val, dict):
                for v in val.values():
                    num = extract_number(v)
                    if num > 0: return num
            return 0.0

        def extract_slice_type(obj):
            """精准匹配 Key 以防止被网络状态里的 URLLC 误导"""
            target_keys = ['recommended_slice', 'slice_type', 'selected_slice', 'slice', 'assigned_slice',
                           'final_slice']

            def search_dict(d):
                if isinstance(d, dict):
                    # 1. 优先查 Key
                    for k, v in d.items():
                        if str(k).lower() in target_keys and isinstance(v, str):
                            v_up = v.upper()
                            if 'EMBB' in v_up: return 'eMBB'
                            if 'URLLC' in v_up: return 'URLLC'
                            if 'MMTC' in v_up: return 'mMTC'
                    # 2. 递归子字典
                    for v in d.values():
                        res = search_dict(v)
                        if res: return res
                elif isinstance(d, list):
                    for item in d:
                        res = search_dict(item)
                        if res: return res
                return None

            found = search_dict(obj)
            if found: return found

            # 如果深度解析失败，使用严格的正则匹配键值对
            raw = json.dumps(obj).upper()
            m = re.search(r'"(?:RECOMMENDED_SLICE|SLICE_TYPE|SELECTED_SLICE|SLICE)"\s*:\s*".*?(EMBB|URLLC|MMTC).*?"',
                          raw)
            if m:
                if 'EMBB' in m.group(1): return 'eMBB'
                if 'URLLC' in m.group(1): return 'URLLC'
                if 'MMTC' in m.group(1): return 'mMTC'

            return 'eMBB'  # 安全默认值

        def recursive_find_num(obj, priority_keywords, secondary_keywords):
            if isinstance(obj, dict):
                exclude = ['available', 'total', 'capacity', 'remaining', 'maximum', 'max_', 'min_', 'range']
                for k, v in obj.items():
                    k_l = str(k).lower()
                    if not any(ex in k_l for ex in exclude) and any(pk in k_l for pk in priority_keywords):
                        n = extract_number(v)
                        if n > 0: return n
                for k, v in obj.items():
                    k_l = str(k).lower()
                    if not any(ex in k_l for ex in exclude) and any(sk in k_l for sk in secondary_keywords):
                        n = extract_number(v)
                        if n > 0: return n
                for v in obj.values():
                    n = recursive_find_num(v, priority_keywords, secondary_keywords)
                    if n > 0: return n
            elif isinstance(obj, list):
                for item in obj:
                    n = recursive_find_num(item, priority_keywords, secondary_keywords)
                    if n > 0: return n
            return 0.0

        def recursive_find_str(obj, keywords):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if any(kw in str(k).lower() for kw in keywords):
                        if isinstance(v, str): return v
                        if isinstance(v, list) and len(v) > 0 and all(isinstance(x, str) for x in v): return "; ".join(
                            v)
                for v in obj.values():
                    s = recursive_find_str(v, keywords)
                    if s != "N/A": return s
            elif isinstance(obj, list):
                for item in obj:
                    s = recursive_find_str(item, keywords)
                    if s != "N/A": return s
            return "N/A"

        normalized_result = {
            'bandwidth': 0.0,
            'data_rate': 0.0,
            'latency': 0.0,
            'recommended_slice': "N/A",
            'intent_analysis': "N/A",
            'slice_reason': "N/A",
            'workload_balanced': True,
            'can_accommodate': True
        }

        # 1. 提取切片类型
        normalized_result['recommended_slice'] = extract_slice_type(result)

        slice_type = normalized_result['recommended_slice']
        if slice_type == "eMBB":
            slice_key = "embb_slice"
        elif slice_type == "URLLC":
            slice_key = "urllc_slice"
        else:
            slice_key = "mmtc_slice"

        # 2. 提取数字
        normalized_result['bandwidth'] = recursive_find_num(
            result,
            ['bandwidth_mhz', 'allocated_bandwidth', 'bandwidth_allocation', 'assigned_bandwidth'],
            ['bandwidth', 'bw']
        )
        normalized_result['data_rate'] = recursive_find_num(
            result,
            ['data_rate_mbps', 'calculated_data_rate', 'net_rate', 'allocated_rate', 'guaranteed_rate'],
            ['data_rate', 'rate', 'throughput']
        )
        normalized_result['latency'] = recursive_find_num(
            result,
            ['estimated_latency', 'expected_latency', 'latency_ms', 'target_latency', 'latency_allocated'],
            ['latency', 'delay']
        )

        # 3. 提取文本
        normalized_result['intent_analysis'] = recursive_find_str(result, ['intent', 'analysis', 'classification',
                                                                           'user_analysis'])
        normalized_result['slice_reason'] = recursive_find_str(result,
                                                               ['rationale', 'reason', 'justification', 'explanation'])

        bandwidth = normalized_result['bandwidth']
        rate = normalized_result['data_rate']
        latency = normalized_result['latency']

        # Debug print
        print(f"\n[DEBUG] Raw result parsed successfully")
        print(f"\n[DEBUG] Normalized bandwidth: {bandwidth}, rate: {rate}")
        print(f"\nIntent Analysis: {normalized_result['intent_analysis']}")
        print(f"Recommended Slice: {slice_type} - {normalized_result['slice_reason']}")
        print(f"Bandwidth Allocation: {bandwidth} MHz")
        print(f"Data Rate: {rate} Mbps")
        print(f"Latency: {latency} ms")

        # If the LLM indicates we can accommodate the user
        if normalized_result.get("can_accommodate", True):
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
            pass
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
    network_state = get_current_network_state()
    initial_embb_usage = network_state["embb_slice"]["resource_usage"]
    initial_urllc_usage = network_state["urllc_slice"]["resource_usage"]
    initial_mmtc_usage = network_state["mmtc_slice"]["resource_usage"]
    initial_embb_util = network_state["embb_slice"]["utilization_rate"]
    initial_urllc_util = network_state["urllc_slice"]["utilization_rate"]
    initial_mmtc_util = network_state["mmtc_slice"]["utilization_rate"]

    initial_embb_total_rate, initial_urllc_total_rate, initial_mmtc_total_rate = calculate_total_transmission_rates()
    initial_avg_resource_util = calculate_average_resource_utilization()

    result = process_user_with_prompt(user_id, location, request, cqi, network_state)
    updated_state = get_current_network_state()

    if not result.get("allocation_failed", True):
        print("\n" + "-" * 40)
        print(f"ALLOCATION RESULT FOR USER {user_id}")
        print("-" * 40)
        concise_report = generate_concise_report(updated_state, user_id)
        print(concise_report)
        user_table = generate_user_allocation_table(updated_state, user_id)
        print(user_table)
    else:
        print("\n" + "-" * 40)
        print(f"ALLOCATION FAILED FOR USER {user_id}")
        print("-" * 40)
        print(f"Request: {request}")
        print(f"Slice type: {result.get('slice_type', 'Unknown')}")
        print(f"Reason: {result.get('failure_reason', 'Unknown error')}")

    final_embb_usage = updated_state["embb_slice"]["resource_usage"]
    final_urllc_usage = updated_state["urllc_slice"]["resource_usage"]
    final_mmtc_usage = updated_state["mmtc_slice"]["resource_usage"]
    final_embb_util = updated_state["embb_slice"]["utilization_rate"]
    final_urllc_util = updated_state["urllc_slice"]["utilization_rate"]
    final_mmtc_util = updated_state["mmtc_slice"]["utilization_rate"]

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

    result["embb_total_rate_before"] = initial_embb_total_rate
    result["embb_total_rate_after"] = final_embb_total_rate
    result["urllc_total_rate_before"] = initial_urllc_total_rate
    result["urllc_total_rate_after"] = final_urllc_total_rate
    result["mmtc_total_rate_before"] = initial_mmtc_total_rate
    result["mmtc_total_rate_after"] = final_mmtc_total_rate
    result["avg_resource_util_before"] = initial_avg_resource_util
    result["avg_resource_util_after"] = final_avg_resource_util

    if ground_truth is not None and not result.get("allocation_failed", True):
        result["intent_correct"] = (result["slice_type"] == ground_truth)
    else:
        result["intent_correct"] = None

    result["ground_truth"] = ground_truth

    return result


# ====================== Main Function ======================

def main(num_users=4, export_file="prompt_based_slicing_results.csv", ray_tracing_csv=None):
    print("Starting prompt-based network slice management system...\n")
    reset_token_stats()

    if ray_tracing_csv:
        users = load_user_data_from_csv(ray_tracing_csv, num_users)
    else:
        users = load_user_data_from_csv(r"F:\code\wirelessagent\ray_tracing_results\ray_tracing_results_east.csv",
                                        num_users)

    print(f"Testing {len(users)} users from ray tracing results CSV")

    detailed_results = []
    embb_utils = []
    urllc_utils = []
    mmtc_utils = []
    workload_balanced_count = 0

    for i, user in enumerate(users):
        print(f"\n{'-' * 140}")
        print(f"PROCESSING USER {user['user_id']} ({i + 1}/{len(users)})")
        print(f"Request: \"{user['request']}\"")
        print(f"CQI: {user['cqi']}")
        if user.get('ground_truth'):
            print(f"Ground Truth Slice: {user['ground_truth']}")
        print(f"{'-' * 140}")

        reset_network_state()

        result = process_user_request(
            user_id=user['user_id'],
            location=user['location'],
            request=user['request'],
            cqi=user['cqi'],
            ground_truth=user.get('ground_truth')
        )

        current_tokens = get_token_usage()
        result["token_used"] = current_tokens["total_tokens"]
        result["prompt_tokens"] = current_tokens["total_prompt_tokens"]
        result["completion_tokens"] = current_tokens["total_completion_tokens"]
        result["llm_call_count"] = current_tokens["llm_call_count"]

        detailed_results.append(result)

        if result.get("workload_balanced", False):
            workload_balanced_count += 1

        if not result.get("allocation_failed", True):
            try:
                if "embb_util_after" in result:
                    embb_util_str = result["embb_util_after"]
                    embb_util = float(embb_util_str.replace("%", "")) if isinstance(embb_util_str, str) else float(
                        embb_util_str)
                    embb_utils.append(embb_util)

                if "urllc_util_after" in result:
                    urllc_util_str = result["urllc_util_after"]
                    urllc_util = float(urllc_util_str.replace("%", "")) if isinstance(urllc_util_str, str) else float(
                        urllc_util_str)
                    urllc_utils.append(urllc_util)

                if "mmtc_util_after" in result:
                    mmtc_util_str = result["mmtc_util_after"]
                    mmtc_util = float(mmtc_util_str.replace("%", "")) if isinstance(mmtc_util_str, str) else float(
                        mmtc_util_str)
                    mmtc_utils.append(mmtc_util)
            except (ValueError, AttributeError):
                pass

    final_embb_total_rate, final_urllc_total_rate, final_mmtc_total_rate = calculate_total_transmission_rates()
    final_avg_resource_util = calculate_average_resource_utilization()

    avg_embb_util = sum(embb_utils) / len(embb_utils) if embb_utils else 0
    avg_urllc_util = sum(urllc_utils) / len(urllc_utils) if urllc_utils else 0
    avg_mmtc_util = sum(mmtc_utils) / len(mmtc_utils) if mmtc_utils else 0

    correct_intents = 0
    total_evaluated = 0

    for result in detailed_results:
        if result.get("intent_correct") is not None:
            total_evaluated += 1
            if result["intent_correct"]:
                correct_intents += 1

    intent_rate = 0 if total_evaluated == 0 else (correct_intents / total_evaluated) * 100
    workload_balanced_rate = 0 if len(detailed_results) == 0 else (workload_balanced_count / len(
        detailed_results)) * 100

    print("\n" + "=" * 60)
    print("SUMMARY OF USER ALLOCATIONS")
    print("=" * 60)

    summary_rows = []
    for res in detailed_results:
        status = "Success" if not res.get("allocation_failed", True) else "Failed"
        intent_match = "Yes" if res.get("intent_correct") else "No" if res.get("intent_correct") is not None else ""

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

    headers = ["User ID", "Status", "Slice", "Ground Truth", "Intent Match", "CQI", "BW (MHz)", "Rate (Mbps)",
               "Latency (ms)", "Adjusted"]
    summary_table = tabulate(summary_rows, headers=headers, tablefmt="grid")
    print(summary_table)

    success_count = sum(1 for res in detailed_results if not res.get("allocation_failed", True))
    total_count = len(detailed_results)

    print("\nStatistics:")
    print(f"Success rate: {success_count}/{total_count} ({success_count / total_count * 100:.1f}%)")

    print("\nIntent Understanding Evaluation:")
    print(f"Correctly identified intents: {correct_intents}/{total_evaluated}")
    print(f"Intent understanding rate: {intent_rate:.1f}%")

    print("\nWorkload Balancing Statistics:")
    print(f"Users with workload balancing: {workload_balanced_count}/{total_count}")
    print(f"Workload balancing rate: {workload_balanced_rate:.1f}%")

    print("\nSlice Utilization Statistics:")
    print(f"Average eMBB utilization: {avg_embb_util:.2f}%")
    print(f"Average URLLC utilization: {avg_urllc_util:.2f}%")
    print(f"Average mMTC utilization: {avg_mmtc_util:.2f}%")

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

    export_results_to_csv(detailed_results, slice_stats, intent_stats, export_file)


if __name__ == "__main__":
    main(num_users=1, export_file="prompt_based_slicing_results.csv")