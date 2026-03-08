

## Requirements:
    python3.11
    pip install -r requirements.txt

## how to use:

Step 1: Go to the OpenStreetMap website to download the "Tongji" map;

Step 2: Select an area and output the "Tongji" campus layout (Tongji.osm);

Step 3: Add the file path of Tongji.osm to the RayTracing_cqi.py;

Step 4: Add the RayTracing results to the WA_DS_V3_NKB.py for network slicing;

Step 5: 代码中指定llm模型
# 指定使用 MiniMax
llm = get_llm("minimax")

# 指定模型版本
llm = get_llm("minimax", model="abab6.5g-chat", temperature=0.5)

Step 6: Output the network slicing results.




