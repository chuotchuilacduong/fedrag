from fedcond_grag.model.graph_llm import GraphLLM
from fedcond_grag.model.dual_graph_llm import DualGraphLLM

load_model = {
    "graph_llm": GraphLLM,
    "dual_graph_llm": DualGraphLLM,
}

llama_model_path = {
    "7b": "meta-llama/Llama-2-7b-hf",
    "7b_chat": "meta-llama/Llama-2-7b-chat-hf",
    "13b": "meta-llama/Llama-2-13b-hf",
    "13b_chat": "meta-llama/Llama-2-13b-chat-hf",
}
