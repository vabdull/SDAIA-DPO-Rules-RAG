"""
SDAIA Personal Data Protection Officer Rules - Streamlit RAG Web Application
Powered by Ollama (Qwen) & nomic-embed-text
"""

import os
import sys
import time
import streamlit as st
from rag_engine import RAGPipeline, OllamaClient

# Configure page settings
st.set_page_config(
    page_title="المستشار الذكي | قواعد تعيين مسؤول حماية البيانات (سدايا)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling with Arabic (RTL) Support and SDAIA Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800&family=Cairo:wght@400;600;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Tajawal', 'Cairo', sans-serif;
        direction: rtl;
        text-align: right;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header styling */
    .sdaia-header {
        background: linear-gradient(135deg, #091E3A 0%, #0D2D59 50%, #008770 100%);
        border-radius: 16px;
        padding: 24px 30px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }

    .sdaia-title {
        font-family: 'Cairo', sans-serif;
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .sdaia-subtitle {
        font-size: 14px;
        color: #B2D8D8;
        margin-top: 6px;
        margin-bottom: 0;
    }

    .sdaia-badges {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }

    .sdaia-badge {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        color: #E6FFFA;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .sdaia-badge.gold {
        background: rgba(245, 158, 11, 0.25);
        color: #FDE68A;
        border-color: rgba(245, 158, 11, 0.4);
    }

    /* Quick Question Prompt Chips */
    .chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 18px;
    }

    /* Source Citation Card */
    .citation-box {
        background: rgba(13, 148, 136, 0.06);
        border: 1px solid rgba(13, 148, 136, 0.25);
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 8px;
        font-size: 13px;
        line-height: 1.6;
    }

    .citation-header {
        font-weight: 700;
        color: #0D9488;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Chat bubble enhancements */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 12px;
    }

    /* Sidebar customization */
    [data-testid="stSidebar"] {
        direction: rtl;
        text-align: right;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(13, 148, 136, 0.2);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 22px;
        font-weight: 800;
        color: #00A389;
    }
    .metric-label {
        font-size: 12px;
        color: #94A3B8;
        margin-top: 4px;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-family: 'Tajawal', sans-serif;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "rag_pipeline" not in st.session_state:
    with st.spinner("جاري تهيئة محرك البحث واستدعاء الوثيقة..."):
        st.session_state.rag_pipeline = RAGPipeline("data/sdaia_dpo_rules.json")

rag: RAGPipeline = st.session_state.rag_pipeline
ollama_client = rag.ollama

# Sidebar Configuration
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #091E3A, #008770); padding: 18px; border-radius: 12px; text-align: center; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.15);">
        <div style="font-size: 28px; margin-bottom: 4px;">🇸🇦 🛡️</div>
        <div style="font-family: 'Cairo', sans-serif; font-weight: 800; font-size: 16px; color: #FFFFFF;">الهيئة السعودية للبيانات والذكاء الاصطناعي</div>
        <div style="font-size: 11px; color: #A7F3D0; font-weight: 600; letter-spacing: 0.5px;">SDAIA DATA PROTECTION RAG</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### ⚙️ إعدادات النموذج والبحث")

    # Ollama Health Check
    is_healthy = ollama_client.check_health()
    if is_healthy:
        st.success("🟢 متصل بخادم Ollama المحلي (Port: 11434)")
    else:
        st.error("🔴 تعذر الاتصال بـ Ollama. تأكد من تشغيل `ollama serve`.")

    # Model Selection
    available_models = ollama_client.list_models()
    qwen_defaults = [m for m in available_models if "qwen" in m.lower()]
    default_model = qwen_defaults[0] if qwen_defaults else (available_models[0] if available_models else "qwen2.5:7b")

    selected_model = st.selectbox(
        "🧠 نموذج الذكاء الاصطناعي (Ollama):",
        options=available_models if available_models else ["qwen2.5:7b", "qwen3.5:4b", "qwen2.5:14b", "qwen2.5:1.5b"],
        index=available_models.index(default_model) if default_model in available_models else 0,
        help="يوصى بنماذج عائلة Qwen مثل qwen2.5:7b أو qwen3.5:4b لدعمها الفائق للغة العربية والامتثال للأنظمة."
    )

    st.markdown("---")
    st.markdown("### 🔍 إعدادات الاسترجاع (RAG)")

    search_mode = st.radio(
        "نوع محرك الاسترجاع:",
        options=["hybrid", "semantic", "keyword"],
        format_func=lambda x: {
            "hybrid": "الهجين (دلالي متجهات + كلمات مفتاحية BM25) ⭐",
            "semantic": "الدلالي فقط (Dense Vector)",
            "keyword": "الكلمات المفتاحية فقط (BM25 Lexical)"
        }[x],
        index=0,
        help="البحث الهجين يجمع بين قوة التضمين الدلالي ودقة الأرقام والكلمات المفتاحية القانونية."
    )

    top_k = st.slider(
        "عدد المواد المسترجعة (Top-K):",
        min_value=2,
        max_value=8,
        value=5,
        step=1,
        help="عدد المواد والنصوص التي يتم تمريرها في سياق النموذج للإجابة."
    )

    temperature = st.slider(
        "درجة الابتكار (Temperature):",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="قيمة منخفضة (0.1 - 0.3) تضمن الالتزام التام بالنص النظامي دون اجتهاد زائد."
    )

    st.markdown("---")
    st.markdown("### 📑 بيانات الوثيقة النشطة")
    doc_info = rag.document_info
    st.markdown(f"""
    - **الوثيقة:** {doc_info.get('title', 'غير محدد')}
    - **الجهة:** {doc_info.get('authority', 'سدايا')}
    - **الإصدار:** {doc_info.get('version', '1.0')} ({doc_info.get('date', 'أغسطس 2024')})
    - **المواد المفهرسة:** {len(rag.chunks)} مادة / قسم
    """)

    # Option to upload additional document
    with st.expander("📤 رفع مستند PDF إضافي"):
        uploaded_pdf = st.file_uploader("اختر ملف PDF لتجربته:", type=["pdf"])
        if uploaded_pdf is not None:
            if st.button("معالجة المستند المرفوع"):
                with st.spinner("جاري قراءة واستخراج النصوص وتوليد المتجهات..."):
                    n_chunks = rag.load_custom_pdf(uploaded_pdf)
                    st.success(f"تمت معالجة {n_chunks} صفحة من المستند الجديد بنجاح!")
                    st.rerun()

        if doc_info.get("classification") == "مخصص":
            if st.button("🔄 العودة إلى وثيقة سدايا الأصلية"):
                st.session_state.rag_pipeline = RAGPipeline("data/sdaia_dpo_rules.json")
                st.rerun()

    # Clear Chat Button
    if st.button("🗑️ مسح سجل المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.markdown("---")
    with st.expander("👥 فريق العمل — المجموعة 5 (Group 5)", expanded=False):
        st.markdown("""
        **برنامج تطوير حلول الذكاء الاصطناعي التوليدي**  
        *أكاديمية سدايا (SDAIA Academy)*

        - 🌟 **Abdullah Alotaibi**
        - 🌟 **Afnan Alharbi**
        - 🌟 **Leena Aljuwaysir**
        - 🌟 **Aljawharah Almutairi**
        - 🌟 **Majed Alsehli**
        - 🌟 **Raid Almousa**
        """)


# Top Brand Header
st.markdown("""
<div class="sdaia-header">
    <div>
        <h1 class="sdaia-title">
            🛡️ المستشار الذكي لقواعد تعيين مسؤول حماية البيانات الشخصية
        </h1>
        <p class="sdaia-subtitle">
            نظام استرجاع وتوليد ذكي (RAG) يستند إلى وثيقة سدايا الرسمية | تطوير: المجموعة 5 (Group 5)
        </p>
    </div>
    <div class="sdaia-badges">
        <span class="sdaia-badge" style="background: rgba(37,99,235,0.35); border-color: rgba(96,165,250,0.5);">المجموعة 5 (Group 5)</span>
        <span class="sdaia-badge">سدايا (SDAIA)</span>
        <span class="sdaia-badge gold">الإصدار 1.0 (أغسطس 2024)</span>
        <span class="sdaia-badge">تصنيف: عام</span>
    </div>
</div>
""", unsafe_allow_html=True)


# Main Tabs
tab_chat, tab_doc_reader, tab_rag_lab = st.tabs([
    "💬 محادثة واستفسار نظامي (RAG Chat)",
    "📖 استعراض الوثيقة الرسمية (Document Reader)",
    "🔬 مختبر الاسترجاع والتقسيم (RAG Lab)"
])


# ==========================================
# TAB 1: RAG CHAT
# ==========================================
with tab_chat:
    # Quick Suggested Questions
    st.markdown("<p style='font-size:14px; color:#64748B; margin-bottom:8px;'>💡 أسئلة مقترحة وسريعة للاختبار:</p>", unsafe_allow_html=True)
    suggested_cols = st.columns(3)

    sample_questions = [
        ("متى يلزم تعيين مسؤول حماية البيانات؟", "ما هي الحالات التي يجب فيها على جهة التحكم تعيين مسؤول حماية البيانات الشخصية؟"),
        ("ما هي شروط ومؤهلات المسؤول؟", "ما هي الشروط والمتطلبات الواجب توافرها عند تعيين مسؤول حماية البيانات الشخصية وفقاً للمادة الرابعة؟"),
        ("ما هي معايير المعالجة على نطاق واسع؟", "ما هي معايير تحديد ما إذا كانت معالجة البيانات الشخصية على نطاق واسع؟"),
        ("ما هي مهام وأدوار مسؤول الحماية؟", "ما هي أدوار ومهام مسؤول حماية البيانات الشخصية وفقاً للمادة الثامنة؟"),
        ("ما هي أمثلة المراقبة المنتظمة؟", "اذكر أمثلة على عمليات المراقبة المنتظمة والممنهجة لأصحاب البيانات الشخصية."),
        ("كيف يتم الإبلاغ والمنصة الوطنية؟", "كيف يتم توثيق التعيين وإبلاغ الهيئة وما هي المنصة المحددة لتزويد بيانات التواصل؟")
    ]

    for idx, (label, query_text) in enumerate(sample_questions):
        col = suggested_cols[idx % 3]
        if col.button(label, key=f"sq_{idx}", use_container_width=True):
            st.session_state.pending_query = query_text
            st.rerun()

    st.markdown("---")

    # Render Chat History
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        citations = msg.get("citations", [])

        with st.chat_message(role, avatar="🧑‍💼" if role == "user" else "🤖"):
            st.markdown(content)
            if citations and role == "assistant":
                with st.expander(f"📚 المصادر والمراجع النظامية ({len(citations)} مواد مسترجعة)", expanded=False):
                    for i, cit in enumerate(citations, 1):
                        score_pct = int(cit.get("score", 0) * 100)
                        st.markdown(f"""
                        <div class="citation-box">
                            <div class="citation-header">
                                <span>📌 المصدر {i}: {cit.get('title')}</span>
                                <span style="background:rgba(13,148,136,0.15); padding:2px 8px; border-radius:12px; font-size:11px;">
                                    صفحة {cit.get('page')} | تطابق: {score_pct}%
                                </span>
                            </div>
                            <div>{cit.get('content')}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Handle Pending Query or New User Input
    user_input = st.chat_input("اكتب سؤالك حول قواعد تعيين مسؤول حماية البيانات الشخصية...")
    active_query = None

    if st.session_state.pending_query:
        active_query = st.session_state.pending_query
        st.session_state.pending_query = None
    elif user_input:
        active_query = user_input

    if active_query:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(active_query)

        # Generate Assistant Response with Streaming
        with st.chat_message("assistant", avatar="🤖"):
            # Step 1: Retrieval
            with st.spinner("🔍 جاري البحث واسترجاع المواد النظامية ذات الصلة..."):
                retrieved_chunks = rag.retrieve(
                    active_query,
                    top_k=top_k,
                    search_mode=search_mode
                )
                system_prompt = rag.build_system_prompt(retrieved_chunks)
                messages = [{"role": "system", "content": system_prompt}]
                if len(st.session_state.messages) > 1:
                    for prev in st.session_state.messages[-4:-1]:
                        messages.append({"role": prev["role"], "content": prev["content"]})
                messages.append({"role": "user", "content": active_query})

            # Step 2: Stream Token by Token
            response_placeholder = st.empty()
            full_response = ""
            try:
                stream_gen = rag.ollama.stream_chat(
                    messages=messages,
                    model=selected_model,
                    temperature=temperature
                )
                for token in stream_gen:
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

                # Prepare citations
                citations_data = []
                for chunk, score in retrieved_chunks:
                    citations_data.append({
                        "title": chunk.title,
                        "article": chunk.article,
                        "page": chunk.page,
                        "content": chunk.content,
                        "score": score
                    })

                # Show citations
                with st.expander(f"📚 المصادر والمراجع المستند إليها ({len(citations_data)} مواد)", expanded=True):
                    for i, cit in enumerate(citations_data, 1):
                        score_pct = int(cit["score"] * 100)
                        st.markdown(f"""
                        <div class="citation-box">
                            <div class="citation-header">
                                <span>📌 المصدر {i}: {cit['title']}</span>
                                <span style="background:rgba(13,148,136,0.15); padding:2px 8px; border-radius:12px; font-size:11px;">
                                    صفحة {cit['page']} | تطابق: {score_pct}%
                                </span>
                            </div>
                            <div>{cit['content']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # Save to state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "citations": citations_data
                })

            except Exception as e:
                err_msg = f"حدث خطأ أثناء معالجة الطلب عبر نموذج Ollama: {str(e)}"
                response_placeholder.error(err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                    "citations": []
                })


# ==========================================
# TAB 2: DOCUMENT READER
# ==========================================
with tab_doc_reader:
    st.markdown("### 📖 نصوص قواعد تعيين مسؤول حماية البيانات الشخصية")
    st.markdown("يمكنك قراءة وتصفح كامل مواد الوثيقة الرسمية الصادرة عن سدايا:")

    doc_search = st.text_input("🔍 تصفية وبحث داخل نصوص الوثيقة:", placeholder="مثال: حساسة، تسرب، غرامة، منصة...")

    # Display Articles
    filtered_chunks = rag.chunks
    if doc_search:
        filtered_chunks = [c for c in rag.chunks if doc_search in c.title or doc_search in c.content]
        st.info(f"تم العثور على {len(filtered_chunks)} مادة تطابق البحث.")

    for chunk in filtered_chunks:
        with st.expander(f"📑 {chunk.title} (الصفحة: {chunk.page})", expanded=bool(doc_search)):
            st.markdown(chunk.content)

    st.markdown("---")
    # Download Full Document
    if os.path.exists("data/sdaia_dpo_rules.md"):
        with open("data/sdaia_dpo_rules.md", "r", encoding="utf-8") as f:
            full_md = f.read()
        st.download_button(
            label="📥 تحميل الوثيقة كاملة بصيغة Markdown",
            data=full_md,
            file_name="sdaia_dpo_rules.md",
            mime="text/markdown"
        )


# ==========================================
# TAB 3: RAG LAB (Retrieval & Chunks)
# ==========================================
with tab_rag_lab:
    st.markdown("### 🔬 مختبر اختبار الاسترجاع والتقطيع الدلالي (RAG Inspector)")
    st.markdown("اختبر كيفية مطابقة محرك البحث الدلالي والمتجهي للأسئلة مع نصوص الوثيقة:")

    lab_query = st.text_input(
        "أدخل عبارة أو استفسار للاختبار:",
        value="ما هي الحالات التي يجب فيها تعيين مسؤول حماية البيانات الشخصية؟"
    )

    lab_col1, lab_col2 = st.columns([1, 1])
    with lab_col1:
        lab_mode = st.selectbox("نمط البحث للاختبار:", ["hybrid", "semantic", "keyword"], index=0)
    with lab_col2:
        lab_k = st.slider("عدد النتائج المعروضة:", 2, 12, 5)

    if lab_query:
        test_results = rag.retrieve(lab_query, top_k=lab_k, search_mode=lab_mode)
        st.markdown(f"#### 🎯 نتائج الترتيب لأفضل {len(test_results)} قطع مسترجعة:")

        for rank, (chunk, score) in enumerate(test_results, 1):
            col_r1, col_r2 = st.columns([4, 1])
            with col_r1:
                st.markdown(f"**#{rank} {chunk.title}** (صفحة {chunk.page})")
                st.caption(chunk.content[:200] + ("..." if len(chunk.content) > 200 else ""))
            with col_r2:
                st.metric(label="درجة التطابق", value=f"{score:.3f}")
            st.divider()

    st.markdown("### 📊 إحصاءات التقطيع (Chunks Statistics)")
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">12</div>
            <div class="metric-label">عدد المواد والأقسام</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{selected_model.split(':')[0]}</div>
            <div class="metric-label">النموذج النشط</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_cols[2]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">768</div>
            <div class="metric-label">أبعاد التضمين (Nomic)</div>
        </div>
        """, unsafe_allow_html=True)
    with stat_cols[3]:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">Hybrid</div>
            <div class="metric-label">تقنية الاسترجاع</div>
        </div>
        """, unsafe_allow_html=True)
