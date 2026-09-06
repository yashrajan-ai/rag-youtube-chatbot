import os
import re
from typing import Iterable, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Optional fallback for Streamlit Cloud / YouTube transcript API failures.
try:
    import yt_dlp
except ImportError:
    yt_dlp = None


# ============================================================
# APP CONFIG
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="centered",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #888;
            margin-bottom: 30px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SECRETS / ENVIRONMENT
# ============================================================

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Works both locally (.env) and on Streamlit Cloud (st.secrets).
    """
    value = os.getenv(name)

    if value:
        return value.strip()

    try:
        value = st.secrets.get(name)
        if value:
            return str(value).strip()
    except Exception:
        pass

    return default


HF_TOKEN = (
    get_secret("HUGGINGFACEHUB_ACCESS_TOKEN")
    or get_secret("HF_TOKEN")
    or get_secret("HUGGINGFACE_API_TOKEN")
)

HF_MODEL_ID = get_secret(
    "HF_MODEL_ID",
    "meta-llama/Llama-3.1-8B-Instruct",
)

# Optional proxy for YouTubeTranscriptApi.
# Example:
# YOUTUBE_HTTP_PROXY=http://user:password@host:port
YOUTUBE_HTTP_PROXY = get_secret("YOUTUBE_HTTP_PROXY")

# Optional external transcript fallback API.
# If configured, it should accept:
#   GET <URL>?id=<VIDEO_ID>&lang=<LANG>
# and return JSON containing either:
#   {"transcript": "..."}
# or:
#   {"text": "..."}
TRANSCRIPT_API_URL = get_secret("TRANSCRIPT_API_URL")


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():
    defaults = {
        "messages": [],
        "vector_store": None,
        "retriever": None,
        "video_id": "",
        "video_processed": False,
        "transcript_language": "",
        "transcript_source": "",
        "chunk_count": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# VALIDATION HELPERS
# ============================================================

def extract_video_id(value: str) -> str:
    """
    Accepts either a YouTube video ID or a normal YouTube URL.
    """
    value = value.strip()

    if not value:
        raise ValueError("Please enter a YouTube video ID or URL.")

    # Plain 11-character YouTube ID.
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        return value

    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/live/)([A-Za-z0-9_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)

    raise ValueError(
        "Invalid YouTube input. Enter an 11-character video ID "
        "or a valid YouTube URL."
    )


def get_youtube_http_client():
    """
    youtube-transcript-api supports a custom requests Session.
    Keeping this optional makes local and Streamlit Cloud deployments
    work without requiring a proxy.
    """
    if not YOUTUBE_HTTP_PROXY:
        return None

    import requests

    session = requests.Session()
    session.proxies.update(
        {
            "http": YOUTUBE_HTTP_PROXY,
            "https": YOUTUBE_HTTP_PROXY,
        }
    )
    return session


# ============================================================
# HUGGING FACE MODELS
# ============================================================

def require_hf_token():
    if not HF_TOKEN:
        raise RuntimeError(
            "Hugging Face authentication is missing.\n\n"
            "Local: add HUGGINGFACEHUB_ACCESS_TOKEN to .env.\n"
            "Streamlit Cloud: add HUGGINGFACEHUB_ACCESS_TOKEN "
            "under Settings → Secrets."
        )


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """
    Multilingual embeddings are used locally.

    This is more robust on Streamlit Cloud than depending on the
    Hugging Face inference API for every embedding request.
    """
    require_hf_token()

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={
            "token": HF_TOKEN,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    require_hf_token()

    llm = HuggingFaceEndpoint(
        repo_id=HF_MODEL_ID,
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
    )

    return ChatHuggingFace(llm=llm)


# ============================================================
# YOUTUBE TRANSCRIPT HELPERS
# ============================================================

def normalize_transcript_items(items: Iterable) -> str:
    texts = []

    for item in items:
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = getattr(item, "text", "")

        if text:
            texts.append(str(text).strip())

    transcript = " ".join(texts)
    transcript = re.sub(r"\s+", " ", transcript).strip()

    if not transcript:
        raise ValueError("The retrieved transcript is empty.")

    return transcript


def fetch_with_youtube_transcript_api(
    video_id: str,
    preferred_languages: list[str],
) -> Tuple[str, str]:
    """
    Primary transcript method.

    It first tries the requested languages. If an exact transcript is not
    available, it tries YouTube's available transcript list and then tries
    translation where YouTube exposes it.
    """
    http_client = get_youtube_http_client()

    if http_client is None:
        api = YouTubeTranscriptApi()
    else:
        # Supported by recent youtube-transcript-api versions.
        api = YouTubeTranscriptApi(http_client=http_client)

    # First attempt: direct language lookup.
    try:
        fetched = api.fetch(
            video_id,
            languages=preferred_languages,
            preserve_formatting=False,
        )
        return normalize_transcript_items(fetched), (
            getattr(fetched, "language_code", None)
            or preferred_languages[0]
        )
    except Exception:
        pass

    # Second attempt: inspect all available transcripts.
    transcript_list = api.list(video_id)
    available = list(transcript_list)

    if not available:
        raise RuntimeError("YouTube returned no available transcripts.")

    # Prefer requested languages, then English.
    preferred_codes = []
    for code in preferred_languages + ["en"]:
        if code and code not in preferred_codes:
            preferred_codes.append(code)

    # Exact/manual/generated transcript.
    for code in preferred_codes:
        for transcript in available:
            if getattr(transcript, "language_code", None) == code:
                try:
                    fetched = transcript.fetch(
                        preserve_formatting=False
                    )
                    return normalize_transcript_items(fetched), code
                except Exception:
                    continue

    # YouTube may allow translating an available transcript.
    target_language = preferred_codes[0] if preferred_codes else "en"

    for transcript in available:
        translation_languages = getattr(
            transcript,
            "translation_languages",
            [],
        ) or []

        supported_translation_codes = {
            item.get("language_code")
            for item in translation_languages
            if isinstance(item, dict)
        }

        if target_language in supported_translation_codes:
            try:
                translated = transcript.translate(target_language)
                fetched = translated.fetch(
                    preserve_formatting=False
                )
                return normalize_transcript_items(
                    fetched
                ), target_language
            except Exception:
                continue

    # Final YouTubeTranscriptApi fallback: use the first available transcript.
    for transcript in available:
        try:
            fetched = transcript.fetch(
                preserve_formatting=False
            )
            return normalize_transcript_items(fetched), (
                getattr(transcript, "language_code", None)
                or "unknown"
            )
        except Exception:
            continue

    raise RuntimeError(
        "YouTube transcripts were detected, but none could be fetched."
    )


def fetch_with_ytdlp(
    video_id: str,
    preferred_languages: list[str],
) -> Tuple[str, str]:
    """
    Secondary fallback.

    yt-dlp asks YouTube for subtitle tracks without downloading the video.
    It can be useful when youtube-transcript-api fails, although YouTube
    may still block cloud-provider IPs.
    """
    if yt_dlp is None:
        raise RuntimeError(
            "yt-dlp is not installed, so the transcript fallback is unavailable."
        )

    language_list = []
    for code in preferred_languages + ["en", "hi"]:
        if code and code not in language_list:
            language_list.append(code)

    lang_selector = ",".join(language_list)

    options = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": language_list,
        "subtitlesformat": "vtt",
        "noplaylist": True,
    }

    # Extracting subtitle information directly is enough for this fallback.
    url = f"https://www.youtube.com/watch?v={video_id}"

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise RuntimeError(f"yt-dlp could not access YouTube: {exc}") from exc

    subtitles = info.get("subtitles") or {}
    automatic = info.get("automatic_captions") or {}

    tracks = []

    # Prefer manually created subtitles.
    for code in language_list:
        tracks.extend(subtitles.get(code, []))

    # Then auto-generated subtitles.
    for code in language_list:
        tracks.extend(automatic.get(code, []))

    if not tracks:
        # Any available language as a final fallback.
        for source in (subtitles, automatic):
            for code, entries in source.items():
                for entry in entries:
                    tracks.append((code, entry))

    if not tracks:
        raise RuntimeError("yt-dlp found no subtitle tracks for this video.")

    # Use the first VTT URL that yt-dlp exposes.
    # Prefer VTT because it is plain subtitle text.
    selected_code = None
    subtitle_url = None
    subtitle_ext = None
    
    for item in tracks:
        if isinstance(item, tuple):
            selected_code, entry = item
        else:
            selected_code, entry = None, item
    
        if entry.get("ext") == "vtt" and entry.get("url"):
            subtitle_url = entry["url"]
            subtitle_ext = "vtt"
            break
    
    if not subtitle_url:
        raise RuntimeError(
            "No usable VTT subtitle track was returned by yt-dlp."
        )

    import requests

    response = requests.get(
        subtitle_url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    subtitle_text = response.text

    # Basic VTT cleanup.
    subtitle_text = re.sub(
        r"WEBVTT.*?\n\n",
        "",
        subtitle_text,
        flags=re.DOTALL,
    )
    subtitle_text = re.sub(
        r"\d{2}:\d{2}:\d{2}\.\d{3} --> .*?\n",
        "",
        subtitle_text,
    )
    subtitle_text = re.sub(r"<[^>]+>", " ", subtitle_text)
    subtitle_text = re.sub(r"\{[^}]+\}", " ", subtitle_text)
    subtitle_text = re.sub(r"\n+", " ", subtitle_text)
    subtitle_text = re.sub(r"\s+", " ", subtitle_text).strip()

    if not subtitle_text:
        raise RuntimeError("yt-dlp returned an empty subtitle track.")

    return subtitle_text, selected_code or "unknown"


def fetch_from_external_api(
    video_id: str,
    language: str,
) -> Tuple[str, str]:
    """
    Optional third fallback.

    Configure TRANSCRIPT_API_URL in Streamlit secrets if you have a
    transcript microservice. This is intentionally optional.
    """
    if not TRANSCRIPT_API_URL:
        raise RuntimeError("External transcript API is not configured.")

    import requests

    response = requests.get(
        TRANSCRIPT_API_URL,
        params={"id": video_id, "lang": language},
        timeout=30,
        headers={"User-Agent": "YouTube-RAG-Streamlit/1.0"},
    )
    response.raise_for_status()

    data = response.json()

    transcript = data.get("transcript") or data.get("text")

    if isinstance(transcript, list):
        transcript = normalize_transcript_items(transcript)

    if not transcript:
        raise RuntimeError(
            "External transcript API returned no transcript text."
        )

    return str(transcript).strip(), language


def get_transcript(
    video_id: str,
    preferred_languages: list[str],
) -> Tuple[str, str, str]:
    """
    Transcript fallback chain:

        1. youtube-transcript-api
        2. yt-dlp subtitles
        3. optional external transcript API

    Returns:
        transcript, language_code, source
    """
    errors = []

    try:
        transcript, language = fetch_with_youtube_transcript_api(
            video_id,
            preferred_languages,
        )
        return transcript, language, "youtube-transcript-api"
    except Exception as exc:
        errors.append(f"youtube-transcript-api: {exc}")

    try:
        transcript, language = fetch_with_ytdlp(
            video_id,
            preferred_languages,
        )
        return transcript, language, "yt-dlp"
    except Exception as exc:
        errors.append(f"yt-dlp: {exc}")

    try:
        language = preferred_languages[0] if preferred_languages else "en"
        transcript, language = fetch_from_external_api(
            video_id,
            language,
        )
        return transcript, language, "external-transcript-api"
    except Exception as exc:
        errors.append(f"external API: {exc}")

    raise RuntimeError(
        "Unable to retrieve a transcript.\n\n"
        + "\n".join(f"• {error}" for error in errors)
        + "\n\n"
        "If this works locally but fails on Streamlit Cloud, YouTube may "
        "be blocking the cloud provider IP. Configure YOUTUBE_HTTP_PROXY "
        "or an external transcript service in Streamlit Secrets."
    )


# ============================================================
# RAG PIPELINE
# ============================================================

def split_text(transcript: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ],
    )
    return splitter.create_documents([transcript])


def create_vector_store(chunks):
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embeddings)


def create_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 15,
            "lambda_mult": 0.7,
        },
    )


def create_context(documents) -> str:
    if not documents:
        return ""

    return "\n\n---\n\n".join(
        doc.page_content for doc in documents
    )


def create_prompt() -> PromptTemplate:
    return PromptTemplate(
        template="""
You are a helpful AI assistant answering questions about a YouTube video.

IMPORTANT RULES:
1. Answer ONLY from the supplied video context.
2. Do not invent facts that are not in the context.
3. The context may be English, Hindi, or another language.
4. Understand multilingual context before answering.
5. Answer in the same language as the user's question unless the user asks
   for another language.
6. If the answer is not present in the context, say:
   "I couldn't find the answer to that in the video."
7. Keep the answer concise but useful.

VIDEO CONTEXT:
-------------------------
{context}
-------------------------

QUESTION:
{question}

ANSWER:
""",
        input_variables=["context", "question"],
    )


def generate_answer(context: str, question: str) -> str:
    if not context.strip():
        return "I couldn't find the answer to that in the video."

    llm = get_llm()
    prompt = create_prompt()

    formatted_prompt = prompt.format(
        context=context,
        question=question,
    )

    response = llm.invoke(formatted_prompt)

    if hasattr(response, "content"):
        answer = response.content
    else:
        answer = str(response)

    return answer.strip()


def process_video(
    video_id: str,
    preferred_languages: list[str],
):
    transcript, language, source = get_transcript(
        video_id,
        preferred_languages,
    )

    chunks = split_text(transcript)

    if not chunks:
        raise RuntimeError("Transcript could not be split into chunks.")

    vector_store = create_vector_store(chunks)
    retriever = create_retriever(vector_store)

    return (
        vector_store,
        retriever,
        len(chunks),
        len(transcript),
        language,
        source,
    )


def ask_question(question: str) -> str:
    retriever = st.session_state.retriever

    if retriever is None:
        return "Please process a YouTube video first."

    documents = retriever.invoke(question)
    context = create_context(documents)

    return generate_answer(context, question)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown(
        """
        ### How it works

        1. Enter a YouTube video ID or URL
        2. Select preferred transcript languages
        3. Retrieve the transcript
        4. Split it into chunks
        5. Generate multilingual embeddings
        6. Store embeddings in FAISS
        7. Retrieve relevant chunks
        8. Generate the answer with Hugging Face
        """
    )

    st.divider()

    st.subheader("🌐 Transcript Languages")

    language_text = st.text_input(
        "Preferred language codes",
        value="en,hi",
        help=(
            "Comma-separated ISO language codes. "
            "Examples: en, hi, fr, de, es, bn."
        ),
    )

    preferred_languages = [
        lang.strip().lower()
        for lang in language_text.split(",")
        if lang.strip()
    ]

    if not preferred_languages:
        preferred_languages = ["en"]

    st.caption(
        "The app tries these languages first and can fall back to "
        "another available YouTube transcript."
    )

    st.divider()

    st.subheader("🔧 RAG Configuration")
    st.write("Chunk size: **1000**")
    st.write("Chunk overlap: **200**")
    st.write("Retriever: **FAISS + MMR**")
    st.write(
        "Embedding: **paraphrase-multilingual-MiniLM-L12-v2**"
    )
    st.write(f"LLM: **{HF_MODEL_ID}**")

    st.divider()

    if st.session_state.transcript_source:
        st.info(
            f"Transcript source: {st.session_state.transcript_source}\n\n"
            f"Language: {st.session_state.transcript_language}\n\n"
            f"Chunks: {st.session_state.chunk_count}"
        )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Reset Video", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vector_store = None
        st.session_state.retriever = None
        st.session_state.video_id = ""
        st.session_state.video_processed = False
        st.session_state.transcript_language = ""
        st.session_state.transcript_source = ""
        st.session_state.chunk_count = 0
        st.rerun()


# ============================================================
# MAIN UI
# ============================================================

st.markdown(
    '<div class="main-title">🎥 YouTube RAG Chatbot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Chat with a YouTube video using Retrieval-Augmented Generation"
    "</div>",
    unsafe_allow_html=True,
)

if not HF_TOKEN:
    st.warning(
        "⚠️ Hugging Face token is not configured. "
        "Add HUGGINGFACEHUB_ACCESS_TOKEN to Streamlit Secrets "
        "before processing a video."
    )


# ============================================================
# VIDEO INPUT
# ============================================================

st.subheader("🎬 Select YouTube Video")

video_input = st.text_input(
    "Enter YouTube Video ID or URL",
    value=st.session_state.video_id,
    placeholder="Example: https://www.youtube.com/watch?v=6x4UgHI7qbk",
)


# ============================================================
# PROCESS VIDEO
# ============================================================

process_button = st.button(
    "🚀 Process Video",
    type="primary",
    use_container_width=True,
)

if process_button:
    try:
        video_id = extract_video_id(video_input)

        if not HF_TOKEN:
            raise RuntimeError(
                "Hugging Face token is missing. Configure "
                "HUGGINGFACEHUB_ACCESS_TOKEN in Streamlit Secrets."
            )

        with st.status(
            "Processing YouTube video...",
            expanded=True,
        ) as status:

            st.write("📜 Retrieving transcript...")

            (
                vector_store,
                retriever,
                chunk_count,
                transcript_length,
                language,
                source,
            ) = process_video(
                video_id,
                preferred_languages,
            )

            st.write(
                f"✅ Transcript retrieved: "
                f"{transcript_length:,} characters"
            )
            st.write(f"🌐 Language: `{language}`")
            st.write(f"🔌 Source: `{source}`")

            st.write("🧠 Embeddings created...")
            st.write(f"📦 Created {chunk_count} chunks")

            st.write("🔎 FAISS retriever created...")

            st.session_state.video_id = video_id
            st.session_state.vector_store = vector_store
            st.session_state.retriever = retriever
            st.session_state.video_processed = True
            st.session_state.messages = []
            st.session_state.transcript_language = language
            st.session_state.transcript_source = source
            st.session_state.chunk_count = chunk_count

            status.update(
                label="✅ Video processed successfully!",
                state="complete",
            )

        st.success(
            "Video is ready. Ask questions below."
        )

    except Exception as exc:
        st.error(
            "❌ Could not process this video.\n\n"
            f"{exc}"
        )

        with st.expander("🔍 Troubleshooting"):
            st.markdown(
                """
                **If this works locally but not on Streamlit Cloud:**

                1. Make sure `HUGGINGFACEHUB_ACCESS_TOKEN` is added under
                   **Streamlit Cloud → Settings → Secrets**.
                2. Make sure your Hugging Face account has access to the
                   selected gated model if the model requires it.
                3. YouTube can block cloud-provider IP addresses. The app
                   therefore tries `youtube-transcript-api`, then `yt-dlp`.
                4. If both are blocked, configure `YOUTUBE_HTTP_PROXY` with
                   a suitable proxy or configure `TRANSCRIPT_API_URL` for
                   your own transcript API.
                5. Check the Streamlit deployment logs for the exact
                   exception.
                """
            )


# ============================================================
# VIDEO PREVIEW
# ============================================================

if st.session_state.video_processed:
    st.divider()

    st.subheader("▶️ YouTube Video")

    st.video(
        f"https://www.youtube.com/watch?v="
        f"{st.session_state.video_id}"
    )

    st.success(
        "✅ Video is ready. You can now ask questions."
    )


# ============================================================
# CHAT
# ============================================================

if st.session_state.video_processed:
    st.divider()
    st.subheader("💬 Chat with the Video")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask something about the video..."
    )

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("🔎 Searching the video..."):
                try:
                    answer = ask_question(question)
                    st.markdown(answer)

                except Exception as exc:
                    answer = (
                        "❌ I couldn't generate an answer.\n\n"
                        f"**Error:** {exc}"
                    )
                    st.error(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

else:
    st.info(
        "👆 Enter a YouTube Video ID or URL above and click "
        "**Process Video** to get started."
    )
