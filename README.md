

## Requirements:
    python3.11
    pip install -r requirements.txt

## how to use:

Step 1: Go to the OpenStreetMap website to download the "Tongji" map;

Step 2: Select an area and output the "Tongji" campus layout (Tongji.osm);

Step 3: Add the file path of Tongji.osm to the RayTracing_cqi.py;

Step 4: Run the agent (choose one version):

    # Without Knowledge Base
    cd no_knowledge_base
    python WA_DS_V3_NKB.py

    # With Knowledge Base + RAG
    cd with_knowledge_base
    python WA_DS_V3_KB.py

Step 5: Configure LLM (optional, defaults to DeepSeek):

    # Default: DeepSeek
    from llm_config import get_llm
    llm = get_llm()

    # Use MiniMax
    llm = get_llm("minimax")

    # Use custom model
    llm = get_llm("minimax", model="abab6.5g-chat", temperature=0.5)

Supported LLM providers: deepseek, minimax, openai, azure_openai

Step 6: Output the network slicing results.

