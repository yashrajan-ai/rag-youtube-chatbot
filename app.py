import streamlit as st
import  os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="centered"
)

# custom css
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

    .status-box {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def initialize_session_state():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    if "video_id" not in st.session_state:
        st.session_state.video_id = ""

    if "video_processed" not in st.session_state:
        st.session_state.video_processed = False
initialize_session_state()

# ============================================================
# load env variables
# ============================================================
HF_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
if not HF_TOKEN:
    st.error(
        "❌ Hugging Face token not found.\n\n"
        "Please add HUGGINGFACEHUB_ACCESS_TOKEN to your .env file."
    )
    st.stop()

# ============================================================
# embedding model
# ============================================================
@st.cache_resource
def get_embeddings():
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    )
    return embeddings

# ============================================================
# llm
# ============================================================
@st.cache_resource
def get_llm():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
    )
    model = ChatHuggingFace(llm=llm)
    return model
# step 1 : youtube transcript extraction
def get_transcript(video_id):
    try:
        # if you don't care which language, this returns the 'best one
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        transcript = " ".join(chunk.text for chunk in transcript_list)
        if not transcript.strip():
            raise ValueError(
                "Transcript is empty."
            )
        return transcript

    except TranscriptsDisabled:
        print("No captions avaliable for this video.")
    except Exception as e :
        raise Exception(
            f"Could not retrieve transcript:{str(e)}"
        )

# step 2 : text splitting
def split_text(transcript):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    return chunks

# step 3 : create faiss vector store
def create_vector_store(chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# step 4 : complete indexing pipeline
def process_video(video_id):
    # get transcript
    transcript = get_transcript(video_id)
    # split transcript
    chunks = split_text(transcript)
    # create embeddings + faiss
    vector_store = create_vector_store(chunks)

    return vector_store, len(chunks)

# step 5 : Retrieval
def retrieve_documents(retriever, question):
    documents = retriever.invoke(question)
    return documents

# step 6 : Create context
def create_context(documents):
    context = "\n\n".join(doc.page_content for doc in documents)
    return context

# step 7 : augmentation / prompt
def create_prompt():
    prompt = PromptTemplate(
        template="""
        You are an AI assistant that answers questions about a YouTube video.
Use ONLY the information provided in the context below.
If the answer cannot be found in the context, say:
"I couldn't find the answer to that in the video."
Do not make up information.
Keep the answer clear, accurate and easy to understand.
-------------------------
CONTEXT: {context}
-------------------------
-------------------------
QUESTION: {question}
-------------------------
""",
        input_variables=['context', 'question']
    )
    return prompt

# step 8 : generation
def generate_answer(context, question):
    llm = get_llm()
    prompt = create_prompt()
    formatted_prompt = prompt.format(context=context,question=question)
    response = llm.invoke(formatted_prompt)

    # handle langchain AIMessage
    if hasattr(response, "content"):
        return response.content
    return str(response)

# step 9 : complete RAG pipeline

def ask_question(question):
    retriever = st.session_state.retriever
    if retriever is None :
        return "Please process a YouTube video first."
    # retrieval
    documents = retrieve_documents(retriever,question)
    # create context
    context = create_context(documents)

    # augmentation

    answer = generate_answer(context, question)
    return answer



# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.markdown(
        """
        ### How it works

        1. Enter YouTube Video ID
        2. Extract transcript
        3. Split transcript
        4. Generate embeddings
        5. Store embeddings in FAISS
        6. Retrieve relevant chunks
        7. Augment the prompt
        8. Generate answer
        """
    )

    st.divider()

    st.subheader("🔧 RAG Configuration")

    st.write(
        "Chunk size: *1000*"
    )

    st.write(
        "Chunk overlap: *200*"
    )

    st.write(
        "Retriever: *FAISS*"
    )

    st.write(
        "Embedding: *all-MiniLM-L6-v2*"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    if st.button(
        "🔄 Reset Video",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.vector_store = None

        st.session_state.retriever = None

        st.session_state.video_id = ""

        st.session_state.video_processed = False

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎥 YouTube RAG Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Chat with a YouTube video using Retrieval-Augmented Generation'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# VIDEO INPUT
# ============================================================

st.subheader("🎬 Select YouTube Video")

video_id = st.text_input(
    "Enter YouTube Video ID which have English transcripts only",
    value=st.session_state.video_id,
    placeholder="Example: 6x4UgHI7qbk",
    help="Enter only the YouTube Video ID, not the complete URL."
)


# ============================================================
# PROCESS VIDEO BUTTON
# ============================================================

process_button = st.button(
    "🚀 Process Video",
    type="primary",
    use_container_width=True
)


if process_button:

    if not video_id.strip():

        st.warning(
            "⚠️ Please enter a YouTube Video ID."
        )

    else:

        video_id = video_id.strip()

        try:

            with st.status(
                "Processing YouTube video...",
                expanded=True
            ) as status:

                # -------------------------------
                # Transcript
                # -------------------------------

                st.write(
                    "📜 Extracting transcript..."
                )

                transcript = get_transcript(
                    video_id
                )

                st.write(
                    f"✅ Transcript extracted "
                    f"({len(transcript):,} characters)"
                )

                # -------------------------------
                # Chunking
                # -------------------------------

                st.write(
                    "✂️ Splitting transcript..."
                )

                chunks = split_text(
                    transcript
                )

                st.write(
                    f"✅ Created {len(chunks)} chunks"
                )

                # -------------------------------
                # Embeddings
                # -------------------------------

                st.write(
                    "🧠 Creating embeddings..."
                )

                vector_store = create_vector_store(
                    chunks
                )

                st.write(
                    "✅ Embeddings created"
                )

                # -------------------------------
                # Retriever
                # -------------------------------

                st.write(
                    "🔎 Creating retriever..."
                )

                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": 4
                    }
                )

                # -------------------------------
                # Save in session
                # -------------------------------

                st.session_state.video_id = video_id

                st.session_state.vector_store = vector_store

                st.session_state.retriever = retriever

                st.session_state.video_processed = True

                # Clear old conversation
                st.session_state.messages = []

                status.update(
                    label="✅ Video processed successfully!",
                    state="complete"
                )

        except Exception as e:

            st.error(
                f"❌ Error processing video:\n\n{str(e)}"
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
# CHAT SECTION
# ============================================================

if st.session_state.video_processed:

    st.divider()

    st.subheader("💬 Chat with the Video")


    # --------------------------------------------
    # Display previous messages
    # --------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # --------------------------------------------
    # Chat input
    # --------------------------------------------

    question = st.chat_input(
        "Ask something about the video..."
    )


    if question:

        # ----------------------------------------
        # User message
        # ----------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)


        # ----------------------------------------
        # Assistant response
        # ----------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching the video..."
            ):

                try:

                    answer = ask_question(
                        question
                    )

                    st.markdown(answer)

                except Exception as e:

                    answer = (
                        "❌ Something went wrong:\n\n"
                        f"{str(e)}"
                    )

                    st.error(answer)


        # ----------------------------------------
        # Save assistant response
        # ----------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


# ============================================================
# INITIAL MESSAGE
# ============================================================

else:

    st.info(
        "👆 Enter a YouTube Video ID above and "
        "click *Process Video* to get started."
    )
