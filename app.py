import streamlit as st
import os
import time  # استيراد مكتبة الوقت
import re    # استخراج مدة الانتظار
from translations import lang_dict
from utils import extract_text_from_files
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # إضافة MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. الإعدادات الأساسية للواجهة ---
st.set_page_config(page_title="AI Academic Teacher", layout="wide", page_icon="🎓")

# ** تهيئة المتغيرات لضمان استقرار الجلسة **
if 'selected_lang' not in st.session_state:
    st.session_state.selected_lang = "English"

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'switch_warning_shown' not in st.session_state:
    st.session_state.switch_warning_shown = False

# دالة لمعالجة تغيير اللغة فورياً
def on_lang_change():
    if len(st.session_state.messages) <= 1:
        st.session_state.messages = [] 

# --- 2. السايدبار ---
with st.sidebar:
    st.session_state.selected_lang = st.selectbox(
        "🌐 Language", 
        ["English", "Arabic", "Japanese", "French"], 
        index=["English", "Arabic", "Japanese", "French"].index(st.session_state.selected_lang),
        on_change=on_lang_change
    )
    
    t = lang_dict[st.session_state.selected_lang]
    
    api_key_input = st.text_input(t["api_key_label"], type="password", placeholder=t["api_key_placeholder"])
    
    st.markdown("---")
    st.markdown(f"### {t['sidebar_header']}")
    st.markdown(f"**{t['upload_label']}**")
    
    uploaded_files = st.file_uploader("", type=['pdf', 'docx', 'pptx', 'txt'], accept_multiple_files=True, key="file_uploader_v3")
    
    if len(st.session_state.messages) > 0:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(t["clear_chat"], key="reset_chat"):
                st.session_state.messages = []
                st.session_state.vectorstore = None
                st.session_state.switch_warning_shown = False
                st.rerun()
        with col2:
            history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
            st.download_button(t["download_chat"], history_text, file_name="academic_session.txt", key="dl_btn_v3")

    st.markdown("---")
    st.info(t["footer"])

if t["direction"] == "rtl":
    st.markdown("""<style> .main { direction: rtl; text-align: right; } </style>""", unsafe_allow_html=True)

# --- 3. محرك معالجة البيانات ---

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def process_docs_to_vectorstore(uploaded_files):
    raw_text = extract_text_from_files(uploaded_files)
    if not raw_text or not raw_text.strip():
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    
    embeddings = load_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# --- 4. دالة الفلترة ---

def get_filtered_context(user_input, vectorstore):
    docs = vectorstore.search(
        user_input, 
        search_type="mmr", 
        k=12, 
        fetch_k=40
    )
    
    if not docs:
        return "NO_DATA_FOUND"
    
    context_parts = []
    for doc in docs:
        page_content = doc.page_content
        
        # --- لمسة المهندس: مرشح استبعاد الفهرس ---
        # إذا كانت الصفحة تحتوي على نقاط متتابعة (تستخدم عادة في الفهارس)
        # أو تحتوي على كلمات تدل على الفهرس في أولها، نقوم بتخطيها
        index_indicators = ["........", " . . . ", "...."]
        if any(indicator in page_content for indicator in index_indicators):
            continue # تخطي هذه القطعة النصية لأنها "ضجيج" فهرس
        # ---------------------------------------

        raw_page = doc.metadata.get("page")
        
        # إذا وجد رقم الصفحة، نجهزه بالتنسيق المطلوب (مع إزاحة +1)
        if isinstance(raw_page, int):
            page_header = f"[REFERENCE SOURCE: Page {raw_page + 1}]\n"
        else:
            page_header = "" 
        
        # دمج الترويسة مع محتوى النص النظيف
        content = f"{page_header}{page_content}"
        context_parts.append(content)
    
    # في حال تم استبعاد كل القطع لأنها فهرس (حالة نادرة)
    if not context_parts:
        return "NO_DATA_FOUND"

    return "\n\n---\n\n".join(context_parts)

# --- 5. نظام الـ Prompts المطور (الذكاء المزدوج مع الذاكرة) ---

def get_academic_chain(api_key, has_context, model_name="llama-3.3-70b-versatile"):
    llm = ChatGroq(
        temperature=0.0,
        model_name=model_name, 
        groq_api_key=api_key
    )
    
    if not has_context:
        system_instruction = """You are a World-Class Academic Polymath and Polyglot Teacher. 
            You possess deep, specialized expertise across diverse academic domains including:
            - Medicine & Life Sciences.
            - Engineering & Applied Technology.
            - Linguistics & Translation.
            - Literature & Philosophy.
            - Humanities, Social Sciences, & Media.
            - Strategic Business & Entrepreneurship.
            You are designed to analyze any scholarly book or document with the precision of a subject-matter expert.
            Currently, NO documents are uploaded. 
            INSTRUCTIONS:
            1. Answer in the requested language ({language}). 
            2. Answer general academic questions or inquiries about documents/books.
            3. Be professional, brief and friendly.
            4. Guide the student to upload materials for deep analysis.
            5. Persona & Social Interaction: > - Maintain a professional, academic, and decisive tone. Avoid over-apologizing.
            - Social Greeting Protocol: If the user greets you (e.g., "Hi", "Hello", "مرحبا") or engages in small talk, respond as a polite, welcoming teacher. Acknowledge the greeting briefly, then immediately invite them to ask about the academic content or the uploaded documents.
            - Boundary Setting: If the user attempts to drift into irrelevant personal or non-academic topics, politely steer them back by saying: "As your academic guide, I am here to focus on your studies and the materials provided. Let's get back to the subject matter."
            - Creator Identity: If the user asks about your creator, developer, or who designed you, respond with pride that you were developed and designed by Eng. Rai Matsumura who is a professional Japanese-Syrian AI Engineer and English Literature Specialist based in Tokyo, Japan. He studied at Damascus University and Syrian Virtual University (SVU).
           6. ACADEMIC ENGAGEMENT (NO-CONTEXT MODE): 
            - If the student asks about academic or scientific concepts, theories, or historical facts, scientists, provide a high-level expert explanation based on your internal knowledge. 
            - Encourage intellectual curiosity by explaining the "Why" and "How" behind the concept.
            - Transition naturally by mentioning that for a more tailored analysis based on their specific curriculum, they should upload their related study materials.
                    """
    else:
        system_instruction = """You are a World-Class Academic Polymath and Polyglot Teacher. 
            You possess deep, specialized expertise across diverse academic domains including:
            - Medicine & Life Sciences.
            - Engineering & Applied Technology.
            - Linguistics & Translation.
            - Literature & Philosophy.
            - Humanities, Social Sciences, & Media.
            - Strategic Business & Entrepreneurship.
            You are designed to analyze any scholarly book or document with the precision of a subject-matter expert.
            Documents ARE uploaded. 

            STRICT OPERATIONAL RULES:
            1. GROUNDING & PRIORITIZATION: 
            - Your absolute priority is the provided context.
            - Always perform a multi-step scan of the context to find specific SYMBOLS and  Terms (like acronyms or technical names) 
            - You MUST perform an exhaustive search across all provided context parts. Even if the definition is brief, extract it.
            - NEVER say "This specific detail is not in the documents" if the term/symbol appears in your provided context snippets. Check twice.

            2. SEARCH & RESPONSE LOGIC: 
            - If the answer is found in the context: Provide it directly according to the requested mode (Rule 6/7).
            - If the term exists but the details are sparse: Use your internal academic expertise to "flesh out" the explanation while citing the existing parts from the document.
            - If (and only if) the information is absolutely nowhere in the context: Move silently to Rule 9.

            3. NO CONTRADICTIONS: Never say "This specific detail is not in the documents" and then provide the answer in the same breath. If you are going to provide an answer based on your general knowledge, do it smoothly as a "Teacher" would.

            4. Language: Respond in ({language}).
            5. Persona & Social Interaction: > - Maintain a professional, academic, and decisive tone. Avoid over-apologizing.
            - Social Greeting Protocol: If the user greets you (e.g., "Hi", "Hello", "مرحبا") or engages in small talk, respond as a polite, welcoming teacher. Acknowledge the greeting briefly, then immediately invite them to ask about the academic content or the uploaded documents.
            - Boundary Setting: If the user attempts to drift into irrelevant personal or non-academic topics, politely steer them back by saying: "As your academic guide, I am here to focus on your studies and the materials provided. Let's get back to the subject matter."
            - Creator Identity: If the user asks about your creator, developer, or who designed you, respond with pride that you were developed and designed by Eng. Rai Matsumura who is a professional Japanese-Syrian AI Engineer and English Literature Specialist based in Tokyo, Japan. He studied at Damascus University and Syrian Virtual University (SVU).
            6. RESPONSE MODE DECISION (CRITICAL):
            - For simple 'What is', Factual, Numeric, True/False, Yes/No, MCQ, Fill in the blanks, Titles, Listing, or Short questions, provide a clear answer and followed by: Evidence from text: "[snippet]". 
            - For 'How', 'Why', 'Explain', 'Describe the process', or if the student asks for 'more depth', use the STRUCTURAL EVIDENCE PROTOCOL.
            - **PAGE CITATION RULE:** If page number available, mention it (Page: X). If NOT available, ignore it. If the user explicitly asks for the page number and it is not available in the metadata, clearly state that you couldn't retrieve it.

            7. STRUCTURAL EVIDENCE PROTOCOL (For Deep Questions):
            * "{header_according}": Followed by a direct quote block - Wrapped in "quotation marks" and formatted as a distinct block.
            * "{header_in_other}": Followed by a multi-sentence pedagogical deep-dive.
            * **TABLES:** IF the question involves comparisons, data analysis, or distinct categories, you MUST present the core information in a Markdown Table for clarity.
            * "{header_summary}": Followed by a concise synthesis.
            - **PAGE CITATION RULE:** If page number available, mention it (Page: X). If NOT available, ignore it. If the user explicitly asks for the page number and it is not available in the metadata, clearly state that you couldn't retrieve it.

            8. CHAT HISTORY: Use the provided chat history to answer meta-questions about the conversation (e.g., "What was my first question?").

            9. HYBRID KNOWLEDGE INTEGRATION: 
            - If a student asks about a concept that is mentioned but not fully explained in the text, provide a "Seamless Hybrid Answer". 
            - Start with what's in the text, then expand using: "{header_external}" to add depth.
            - UNDER THIS HEADER: Clearly state that the following information is supplementary knowledge provided by you as an Academic Teacher to add depth and clarity beyond the document's scope.
            - Only say "This specific detail is not in the documents" if the topic is completely irrelevant to the uploaded content.

            10. ARABIC PROCESSING: Fix any encoding or structural issues in Arabic context before extraction to ensure no technical terms are lost."""

    # تحسين الـ Prompt لدعم الذاكرة (Chat History)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="chat_history"), # هنا تكمن الذاكرة الخارقة
        ("human", "ACADEMIC CONTEXT:\n{context}\n\nSTUDENT QUESTION: {question}")
    ])
    
    return prompt | llm
    

# --- 6. عرض المحتوى ومنطق الدردشة ---

st.title(f"🎓 {t['title']}")

if not api_key_input:
    st.info("🔐 " + t["api_warning"])
else:
    if len(st.session_state.messages) == 0:
        initial_msg = t["welcome_msg"]
        st.session_state.messages.append({"role": "assistant", "content": initial_msg})

    if uploaded_files and st.session_state.vectorstore is None:
        with st.spinner(t["processing"]):
            try:
                vs = process_docs_to_vectorstore(uploaded_files)
                if vs:
                    st.session_state.vectorstore = vs
                    st.success(t["success_upload"])
            except Exception as e:
                st.error(f"Error: {e}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input(t["input_placeholder"]):
        # تخزين سؤال المستخدم
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner(t["processing"]):
                start_time = time.time()
                
                has_vectorstore = st.session_state.vectorstore is not None
                if has_vectorstore:
                    context = get_filtered_context(user_query, st.session_state.vectorstore)
                else:
                    context = "No documents uploaded yet. Speak generally as an academic guide."
                
                h_according = t.get("according_to", "According to the text:")
                h_in_other = t.get("in_other_words", "In other words:")
                h_summary = t.get("summary", "Summary:")
                h_external = t.get("external_context", "External Academic Context:")

                # 1. تجهيز تاريخ المحادثة للنموذج (تحويلها لكائنات رسائل رسمية)
                chat_history = []
                # نأخذ آخر 6 رسائل لضمان بقاء السياق حياً دون إثقال الذاكرة
                for m in st.session_state.messages[-6:-1]:
                    if m["role"] == "user":
                        chat_history.append(HumanMessage(content=m["content"]))
                    else:
                        chat_history.append(AIMessage(content=m["content"]))
                
                try:
                    # 2. جلب السلسلة الأكاديمية
                    chain = get_academic_chain(api_key_input, has_vectorstore, model_name="llama-3.3-70b-versatile")
                    
                    # 3. التنفيذ مع تمرير الذاكرة والسياق
                    response = chain.invoke({
                        "context": context,
                        "question": user_query,
                        "chat_history": chat_history,  # الذاكرة الآن رسمية واحترافية
                        "language": st.session_state.selected_lang,
                        "header_according": h_according,
                        "header_in_other": h_in_other,
                        "header_summary": h_summary,
                        "header_external": h_external
                    })
                    full_response = response.content
                    
                except Exception as e:
                    if "429" in str(e):
                        wait_time_match = re.search(r"(\d+m)", str(e))
                        wait_time = wait_time_match.group(1) if wait_time_match else "a few moments"

                        if not st.session_state.switch_warning_shown:
                            st.warning(f"⚠️ [ملاحظة]: تم تجاوز حد الاستخدام للنموذج القوي. يتم الانتقال مؤقتاً للنموذج البديل لإتمام طلبك. (يرجى الانتظار لمدة {wait_time} لاستعادة النموذج القوي).")
                            st.session_state.switch_warning_shown = True
                        
                        chain = get_academic_chain(api_key_input, has_vectorstore, model_name="llama-3.1-8b-instant")
                        response = chain.invoke({
                            "context": context,
                            "question": user_query,
                            "chat_history": chat_history, # تمرير الذاكرة هنا للنموذج البديل أيضاً
                            "language": st.session_state.selected_lang,
                            "header_according": h_according,
                            "header_in_other": h_in_other,
                            "header_summary": h_summary,
                            "header_external": h_external
                        })
                        full_response = response.content
                    else:
                        st.error(f"Error: {e}")
                        full_response = None

                if full_response:
                    end_time = time.time()
                    st.markdown(full_response)
                    st.caption(f"⏱️ Response took: {round(end_time - start_time, 2)}s")
                    st.session_state.messages.append({"role": "assistant", "content": full_response})