The rapid evolution of wireless networks presents unprecedented challenges in managing complex and dynamic systems. Existing methods are increasingly facing fundamental limitations in addressing these challenges. In this paper, we introduce WirelessAgent, a novel framework that harnesses large language models (LLMs) to create autonomous AI agents for diverse wireless network tasks. This framework integrates four core modules that mirror human cognitive processes: perception, memory, planning, and action. To implement it, we provide a basic usage based on agentic workflows and the LangGraph architecture. We demonstrate the effectiveness of WirelessAgent through a comprehensive case study on network slicing. The numerical results show that WirelessAgent achieves 44.4% higher bandwidth utilization than the Prompt-based method, while performing only $4.3\%$ below the Rule-based optimality. Notably, WirelessAgent delivers near-optimal network throughput across diverse network scenarios. These underscore the framework's potential for intelligent and autonomous control in future wireless networks.

## Requirements:
    python3.11
    pip install -r requirements.txt

## how to use:

Step 1: Go to the OpenStreetMap website to download the "Tongji" map;

Step 2: Select an area and output the "Tongji" campus layout (Tongji.osm);

Step 3: Add the file path of Tongji.osm to the RayTracing_cqi.py;

Step 4: Add the RayTracing results to the WA_DS_V3_NKB.py for network slicing;

Step 5: Output the network slicing results.




