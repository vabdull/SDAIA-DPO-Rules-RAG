"""
RAG Engine for SDAIA Personal Data Protection Officer Regulations
Powered by Ollama (Qwen) and nomic-embed-text
"""

import json
import math
import os
import pickle
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Generator, Tuple, Optional
import numpy as np


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for robust keyword and lexical matching."""
    if not text:
        return ""
    # Strip diacritics / tashkeel
    tashkeel = re.compile(r'[\u064B-\u0652\u0653-\u0655\u0670]')
    text = re.sub(tashkeel, '', text)
    # Normalize Alef forms
    text = re.sub(r'[إأآا]', 'ا', text)
    # Normalize Taa Marbuta and Haa
    text = re.sub(r'ة', 'ه', text)
    # Normalize Yaa and Alef Maqsura
    text = re.sub(r'[ىي]', 'ي', text)
    # Normalize punctuation and digits
    text = re.sub(r'[ـ]', '', text)  # tatweel
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def check_health(self) -> bool:
        """Check if Ollama server is alive."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", headers={"User-Agent": "StreamlitRAG/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List all models available locally in Ollama, prioritizing qwen2.5:7b."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", headers={"User-Agent": "StreamlitRAG/1.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                # Prioritize qwen2.5:7b first, then other qwen models, then remaining
                qwen_7b = [m for m in models if "qwen2.5:7b" in m.lower() or "qwen2.5:latest" in m.lower()]
                other_qwen = [m for m in models if "qwen" in m.lower() and m not in qwen_7b]
                other_models = [m for m in models if "qwen" not in m.lower() and "embed" not in m.lower()]
                return qwen_7b + other_qwen + other_models
        except Exception as e:
            print(f"Error fetching Ollama models: {e}")
            return ["qwen2.5:7b"]

    def get_embedding(self, text: str, model: str = "nomic-embed-text") -> Optional[List[float]]:
        """Get embedding vector for a single text using nomic-embed-text or embedding model."""
        try:
            payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/embeddings",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("embedding")
        except Exception as e:
            print(f"Embedding error: {e}")
            return None

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "qwen2.5:7b",
        temperature: float = 0.2
    ) -> Generator[str, None, None]:
        """Stream chat completions from Ollama /api/chat."""
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
            }
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        msg = chunk.get("message", {})
                        content = msg.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


@dataclass
class DocumentChunk:
    id: str
    title: str
    article: str
    page: int
    content: str
    embedding: Optional[List[float]] = None


ARABIC_STOPWORDS = {
    "في", "من", "على", "الى", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك", "هؤلاء",
    "التي", "الذي", "الذين", "اللاتي", "اللواتي", "ما", "ماذا", "لماذا", "كيف", "متى",
    "اين", "أين", "هل", "هو", "هي", "هم", "هن", "نحن", "انا", "أنا", "انت", "أنت",
    "كان", "كانت", "يكون", "تكون", "ان", "أن", "إن", "اذا", "إذا", "او", "أو", "ثم",
    "كل", "بعض", "غير", "بين", "لدى", "عند", "قبل", "بعد", "حيث", "فان", "فإن"
}

def light_stem_arabic(word: str) -> str:
    """Strip common Arabic affixes (prefixes: ال, لل, و, ف, ب, ك; suffixes: ها, ون, ين, ات)."""
    w = word
    # Prefixes
    if w.startswith("لل") and len(w) > 4:
        w = w[2:]
    elif w.startswith("ال") and len(w) > 4:
        w = w[2:]
    elif (w.startswith("و") or w.startswith("ف")) and len(w) > 3:
        w = w[1:]
        if w.startswith("ال") and len(w) > 4:
            w = w[2:]
    elif (w.startswith("ب") or w.startswith("ك")) and len(w) > 3:
        w = w[1:]
        if w.startswith("ال") and len(w) > 4:
            w = w[2:]

    # Suffixes
    if w.endswith("ات") and len(w) > 4:
        w = w[:-2]
    elif (w.endswith("ون") or w.endswith("ين")) and len(w) > 4:
        w = w[:-2]
    elif w.endswith("ها") and len(w) > 4:
        w = w[:-2]
    return w


LEGAL_SYNONYMS = {
    "حالات": ["احوال", "اوضاع"],
    "حاله": ["حال"],
    "احوال": ["حالات"],
    "شروط": ["متطلبات", "معايير"],
    "شرط": ["متطلب", "معيار"],
    "متطلبات": ["شروط", "معايير"],
    "واجبات": ["مهام", "ادوار", "مسؤوليات"],
    "مهام": ["واجبات", "ادوار", "مسؤوليات"],
    "ابلاغ": ["اعلان", "تزويد", "اخطار", "منصه"],
    "اعلان": ["ابلاغ", "اشعار", "تزويد"],
    "تسجيل": ["تزويد", "توثيق", "منصه"],
    "منصه": ["حوكمه", "وطنيه", "تزويد"],
    "عقوبات": ["مخالفات", "جزاءات"],
}

class BM25Retriever:
    """Lightweight Arabic-aware BM25 lexical retriever with zero external dependencies."""

    def __init__(self, chunks: List[DocumentChunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_size = len(chunks)
        self.doc_lengths = []
        self.avg_doc_len = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.term_freqs: List[Dict[str, int]] = []

        self._build_index()

    def _tokenize(self, text: str, expand_synonyms: bool = False) -> List[str]:
        norm = normalize_arabic(text)
        tokens = []
        for raw_w in norm.split():
            if len(raw_w) < 2 or raw_w in ARABIC_STOPWORDS:
                continue
            stem = light_stem_arabic(raw_w)
            if stem and stem not in ARABIC_STOPWORDS:
                tokens.append(stem)
                if stem != raw_w:
                    tokens.append(raw_w)
                if expand_synonyms and stem in LEGAL_SYNONYMS:
                    for syn in LEGAL_SYNONYMS[stem]:
                        tokens.append(syn)
        return tokens

    def _build_index(self):
        total_len = 0
        for chunk in self.chunks:
            # Combine title and content, giving title 3x weight
            full_text = f"{chunk.title} {chunk.title} {chunk.title} {chunk.content}"
            tokens = self._tokenize(full_text, expand_synonyms=False)
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)

            for t in set(tokens):
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avg_doc_len = total_len / max(self.corpus_size, 1)

    def search(self, query: str) -> List[float]:
        q_tokens = self._tokenize(query, expand_synonyms=True)
        scores = [0.0] * self.corpus_size
        if not q_tokens:
            return scores

        for token in q_tokens:
            df = self.doc_freqs.get(token, 0)
            if df == 0:
                continue
            # IDF formula
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))
            for i in range(self.corpus_size):
                tf = self.term_freqs[i].get(token, 0)
                if tf > 0:
                    doc_len = self.doc_lengths[i]
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1e-5)))
                    scores[i] += idf * (numerator / denominator)

        # Normalize scores to [0, 1]
        max_s = max(scores) if scores else 0.0
        if max_s > 0:
            scores = [s / max_s for s in scores]
        return scores


class RAGPipeline:
    def __init__(
        self,
        data_json_path: str,
        cache_dir: str = "data/cache",
        embedding_model: str = "nomic-embed-text"
    ):
        self.data_json_path = data_json_path
        self.cache_dir = cache_dir
        self.embedding_model = embedding_model
        self.ollama = OllamaClient()
        self.chunks: List[DocumentChunk] = []
        self.document_info: Dict[str, Any] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Retriever] = None

        os.makedirs(self.cache_dir, exist_ok=True)
        self.load_data()
        self.build_or_load_embeddings()

    def load_data(self):
        """Load document metadata and chunks from JSON."""
        with open(self.data_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.document_info = data.get("document_info", {})
        self.chunks = []
        for sec in data.get("sections", []):
            self.chunks.append(
                DocumentChunk(
                    id=sec["id"],
                    title=sec["title"],
                    article=sec["article"],
                    page=sec["page"],
                    content=sec["content"]
                )
            )

        # Initialize BM25 lexical retriever
        self.bm25 = BM25Retriever(self.chunks)

    def build_or_load_embeddings(self):
        """Build or load cached embeddings for document chunks."""
        cache_file = os.path.join(self.cache_dir, f"embeddings_{self.embedding_model.replace(':', '_')}.pkl")

        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                if len(cached_data) == len(self.chunks):
                    for i, emb in enumerate(cached_data):
                        self.chunks[i].embedding = emb
                    self.embeddings_matrix = np.array(cached_data)
                    return
            except Exception as e:
                print(f"Error loading cache: {e}. Recomputing...")

        # Compute embeddings via Ollama
        vectors = []
        for chunk in self.chunks:
            # Embed article title + content
            embed_input = f"search_document: {chunk.title}\n{chunk.content}"
            emb = self.ollama.get_embedding(embed_input, model=self.embedding_model)
            if emb is None:
                # Fallback to zero vector if embedding service is down
                emb = [0.0] * 768
            chunk.embedding = emb
            vectors.append(emb)

        self.embeddings_matrix = np.array(vectors)

        # Save to cache
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(vectors, f)
        except Exception as e:
            print(f"Error saving embeddings cache: {e}")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        search_mode: str = "hybrid",
        alpha: float = 0.65
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve the most relevant chunks using:
        - 'hybrid': Dense vector similarity + BM25 keyword score
        - 'semantic': Dense vector embeddings only
        - 'keyword': BM25 keyword matching only
        """
        n_chunks = len(self.chunks)
        dense_scores = np.zeros(n_chunks)
        sparse_scores = np.zeros(n_chunks)

        # 1. Sparse BM25 Search
        if search_mode in ("hybrid", "keyword") and self.bm25:
            sparse_scores = np.array(self.bm25.search(query))

        # 2. Dense Semantic Search
        if search_mode in ("hybrid", "semantic") and self.embeddings_matrix is not None:
            # nomic-embed-text query prefix
            query_embed_input = f"search_query: {query}"
            q_emb = self.ollama.get_embedding(query_embed_input, model=self.embedding_model)
            if q_emb is not None:
                q_vec = np.array(q_emb)
                # Cosine similarity
                norm_q = np.linalg.norm(q_vec)
                norm_docs = np.linalg.norm(self.embeddings_matrix, axis=1)
                norm_docs[norm_docs == 0] = 1e-10
                if norm_q > 0:
                    dense_scores = np.dot(self.embeddings_matrix, q_vec) / (norm_docs * norm_q)
                    # Clip between 0 and 1
                    dense_scores = np.clip(dense_scores, 0, 1)

        # Combine scores
        if search_mode == "semantic":
            final_scores = dense_scores
        elif search_mode == "keyword":
            final_scores = sparse_scores
        else:  # hybrid
            # Weighted sum
            final_scores = alpha * dense_scores + (1 - alpha) * sparse_scores

        # Rank
        ranked_indices = np.argsort(final_scores)[::-1][:top_k]
        results = [(self.chunks[i], float(final_scores[i])) for i in ranked_indices]
        return results

    def build_system_prompt(self, context_chunks: List[Tuple[DocumentChunk, float]]) -> str:
        """Construct a high-fidelity Arabic system instruction and reference context."""
        context_str = ""
        for i, (chunk, score) in enumerate(context_chunks, 1):
            context_str += f"""
---
[المصدر {i}]: {chunk.title} (الصفحة: {chunk.page})
{chunk.content}
"""

        system_prompt = f"""أنت مستشار قانوني وتنظيمي خبير متخصص في أنظمة ولوائح الهيئة السعودية للبيانات والذكاء الاصطناعي (سدايا).
مهمتك هي الإجابة عن أسئلة المستخدم بدقة وموثوقية بالاعتماد حصراً على النصوص والوثيقة الرسمية التالية: "قواعد تعيين مسؤول حماية البيانات الشخصية" (الصادرة في أغسطس 2024).

القواعد الإرشادية للإجابة:
1. استند فقط إلى النصوص المرفقة في السياق أدناه.
2. اذكر رقم المادة أو القسم المستند إليه دائماً (مثلاً: "وفقاً للمادة الخامسة..." أو "بحسب المادة الرابعة...").
3. كن دقيقاً، واضحاً، ومنظماً في شكل نقاط أو فقرات موجزة.
4. إذا لم يكن السؤال مغطى أو مذكوراً في النصوص المرفقة، بيّن بكل أمانة أن الوثيقة لم تتطرق لذلك صراحةً، دون تأليف أو تخمين.
5. حافظ على الأسلوب المهني واللغة العربية الفصحى السليمة.

[النصوص الرسمية المرجعية]:
{context_str}
"""
        return system_prompt

    def ask_stream(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        model: str = "qwen2.5:7b",
        top_k: int = 3,
        search_mode: str = "hybrid",
        temperature: float = 0.2
    ) -> Tuple[Generator[str, None, None], List[Tuple[DocumentChunk, float]]]:
        """
        Execute RAG search and return streaming generator + retrieved citations.
        """
        retrieved_chunks = self.retrieve(query, top_k=top_k, search_mode=search_mode)
        system_prompt = self.build_system_prompt(retrieved_chunks)

        messages = [{"role": "system", "content": system_prompt}]

        # Append recent history if any
        if chat_history:
            # keep last 4 exchanges to preserve context window
            for msg in chat_history[-4:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})

        stream = self.ollama.stream_chat(messages, model=model, temperature=temperature)
        return stream, retrieved_chunks

    def load_custom_pdf(self, pdf_file) -> int:
        """Parse uploaded PDF file and replace or augment current chunks."""
        from pypdf import PdfReader
        reader = PdfReader(pdf_file)
        new_chunks = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                new_chunks.append(
                    DocumentChunk(
                        id=f"custom_page_{i+1}",
                        title=f"مستند مرفوع - صفحة {i+1}",
                        article=f"صفحة {i+1}",
                        page=i+1,
                        content=text
                    )
                )

        if new_chunks:
            self.chunks = new_chunks
            self.bm25 = BM25Retriever(self.chunks)
            # Compute embeddings for new chunks
            vectors = []
            for chunk in self.chunks:
                emb = self.ollama.get_embedding(f"search_document: {chunk.content[:1000]}", model=self.embedding_model)
                if emb is None:
                    emb = [0.0] * 768
                chunk.embedding = emb
                vectors.append(emb)
            self.embeddings_matrix = np.array(vectors)
            self.document_info = {
                "title": getattr(pdf_file, "name", "مستند PDF مخصص"),
                "authority": "مستند خارجي مرفوع",
                "classification": "مخصص",
                "version": "1.0",
                "date": "الآن",
                "legal_basis": "نص مخصص مرفوع من المستخدم"
            }
        return len(new_chunks)
