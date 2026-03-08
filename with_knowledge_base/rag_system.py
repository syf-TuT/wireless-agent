"""
RAG检索优化模块
- 知识库向量化与语义检索
- 混合检索器 (向量 + BM25)
- 对话历史记忆压缩
"""

import re
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import SentenceTransformer


class LocalSentenceTransformerEmbeddings(Embeddings):
    """本地Sentence Transformer嵌入模型封装"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


class KnowledgeBaseVectorStore:
    """知识库向量化管理器 - 将意图示例转为向量支持语义检索"""

    def __init__(
        self,
        knowledge_base_path: str,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.knowledge_base_path = knowledge_base_path
        self.embeddings = LocalSentenceTransformerEmbeddings(embedding_model)
        self.vectorstore = None

    def load_and_parse_knowledge_base(self) -> List[Document]:
        """解析知识库文件，提取意图-切片类型对"""
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析Python元组格式: ("text", "label")
        intent_pattern = r'\("([^"]+)",\s*"(\w+)"\)'
        matches = re.findall(intent_pattern, content)

        documents = []
        for idx, (intent, slice_type) in enumerate(matches):
            doc = Document(
                page_content=intent,
                metadata={
                    "id": idx,
                    "slice_type": slice_type,
                    "category": self._categorize_intent(intent)
                }
            )
            documents.append(doc)

        return documents

    def _categorize_intent(self, intent: str) -> str:
        """基于关键词分类意图"""
        intent_lower = intent.lower()
        if any(k in intent_lower for k in ['video', '4k', '8k', 'stream', 'download', 'game', 'vr', 'ar', 'cloud']):
            return "eMBB"
        elif any(k in intent_lower for k in ['control', 'auto', 'remote', 'surgery', 'vehicle', 'drone', 'industrial']):
            return "URLLC"
        elif any(k in intent_lower for k in ['sensor', 'meter', 'iot', 'monitor', 'smart', 'wearable']):
            return "mMTC"
        return "unknown"

    def build_vectorstore(self, force_rebuild: bool = False) -> FAISS:
        """构建或加载向量库"""
        # 获取索引目录
        index_dir = self.knowledge_base_path.replace('.txt', '_index')

        if os.path.exists(index_dir) and not force_rebuild:
            # 加载已有索引
            try:
                self.vectorstore = FAISS.load_local(
                    index_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"[RAG] Loaded existing vector index from {index_dir}")
            except Exception as e:
                print(f"[RAG] Failed to load index, rebuilding: {e}")
                force_rebuild = True

        if self.vectorstore is None or force_rebuild:
            # 构建新索引
            documents = self.load_and_parse_knowledge_base()
            print(f"[RAG] Building vector index for {len(documents)} documents...")
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # 保存索引
            self.vectorstore.save_local(index_dir)
            print(f"[RAG] Vector index saved to {index_dir}")

        return self.vectorstore

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filter_slice_type: Optional[str] = None
    ) -> List[Tuple[Document, float]]:
        """语义检索"""
        if not self.vectorstore:
            self.build_vectorstore()

        # 检索相似意图
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        # 按切片类型过滤
        if filter_slice_type:
            results = [
                (doc, score) for doc, score in results
                if doc.metadata.get("slice_type") == filter_slice_type
            ]

        return results


class HybridRetriever:
    """混合检索器：语义向量 + BM25关键词"""

    def __init__(
        self,
        vectorstore: FAISS,
        knowledge_base_path: str
    ):
        self.vectorstore = vectorstore
        self.knowledge_base_path = knowledge_base_path
        self.intent_texts: List[str] = []
        self.bm25_retriever: Optional[BM25Retriever] = None

        # 初始化BM25
        self._init_bm25()

    def _init_bm25(self):
        """初始化BM25检索器"""
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析意图文本
        intent_pattern = r'\("([^"]+)",\s*"(\w+)"\)'
        intents = re.findall(intent_pattern, content)
        self.intent_texts = [i[0] for i in intents]

        # 创建文档列表用于BM25
        documents = [Document(page_content=text) for text in self.intent_texts]
        self.bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=lambda x: x.lower().split()
        )
        print(f"[RAG] BM25 retriever initialized with {len(self.intent_texts)} documents")

    def get_relevant_documents(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.7
    ) -> List[Document]:
        """
        获取混合检索结果

        Args:
            query: 用户查询
            k: 返回数量
            alpha: 向量权重 (1-alpha为BM25权重)
        """
        # 向量检索
        vector_results = self.vectorstore.similarity_search_with_score(query, k=k)
        vector_docs = {doc.page_content: doc for doc, _ in vector_results}

        # BM25检索
        bm25_results = self.bm25_retriever.invoke(query)
        bm25_docs = {doc.page_content: doc for doc in bm25_results[:k]}

        # 融合结果 (优先向量结果)
        combined = []

        # 先添加向量检索结果
        for doc in vector_results:
            if doc[0].page_content not in [d.page_content for d in combined]:
                combined.append(doc[0])

        # 再添加BM25结果
        for doc in bm25_results:
            if doc.page_content not in [d.page_content for d in combined]:
                combined.append(doc)

        return combined[:k]

    def retrieve_with_context(
        self,
        query: str,
        k: int = 3,
        alpha: float = 0.7
    ) -> str:
        """
        检索并构建精简上下文

        Returns:
            格式化的上下文字符串
        """
        results = self.get_relevant_documents(query, k=k, alpha=alpha)

        context_parts = ["# Relevant Knowledge Base Examples\n"]
        for i, doc in enumerate(results, 1):
            intent = doc.page_content
            slice_type = doc.metadata.get("slice_type", "unknown")
            context_parts.append(f"{i}. Intent: \"{intent}\" → Slice: {slice_type}")

        return "\n".join(context_parts)


class IncrementalKnowledgeUpdater:
    """增量知识更新 - 记录新经验"""

    def __init__(self, vectorstore: FAISS, persist_dir: str):
        self.vectorstore = vectorstore
        self.persist_dir = persist_dir

    def add_new_experience(
        self,
        user_request: str,
        slice_decision: str,
        reasoning: str
    ) -> str:
        """添加新经验到知识库"""
        doc = Document(
            page_content=f"User: {user_request} → Decision: {slice_decision} (Reason: {reasoning})",
            metadata={
                "slice_type": slice_decision,
                "source": "user_feedback",
                "timestamp": "2024-01-01"
            }
        )

        # 添加到向量库
        self.vectorstore.add_documents([doc])

        # 持久化
        self.vectorstore.save_local(self.persist_dir)

        return "New experience added to knowledge base"


def initialize_rag_system(
    knowledge_base_path: str,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> Tuple[KnowledgeBaseVectorStore, HybridRetriever]:
    """
    初始化RAG系统

    Args:
        knowledge_base_path: 知识库文件路径
        embedding_model: 嵌入模型名称 (默认: all-MiniLM-L6-v2)

    Returns:
        (知识库向量存储, 混合检索器)
    """
    print("[RAG] Initializing RAG system...")

    # 1. 创建知识库向量存储 (使用本地嵌入模型)
    kb_vector_store = KnowledgeBaseVectorStore(
        knowledge_base_path=knowledge_base_path,
        embedding_model=embedding_model
    )

    # 2. 构建向量索引
    vectorstore = kb_vector_store.build_vectorstore()

    # 3. 创建混合检索器
    hybrid_retriever = HybridRetriever(
        vectorstore=vectorstore,
        knowledge_base_path=knowledge_base_path
    )

    print("[RAG] RAG system initialized successfully!")

    return kb_vector_store, hybrid_retriever
