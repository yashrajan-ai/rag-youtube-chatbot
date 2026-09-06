# 🎥🤖 RAG YouTube Chatbot

> **Chat with YouTube videos using Retrieval-Augmented Generation (RAG)!**

An intelligent **YouTube Video Question-Answering chatbot** that allows users to provide a YouTube video and ask questions about its content.

Instead of sending the entire transcript to an LLM, the application retrieves only the **most relevant pieces of information** and uses them to generate accurate, context-aware answers.

---

## 🌟 Demo

🚀 **Live Application:** Add your Streamlit deployment link here

📂 **GitHub Repository:**  
https://github.com/yashrajan-ai/rag-youtube-chatbot

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎥 YouTube Integration | Extract content from YouTube videos |
| 📝 Transcript Processing | Retrieves and processes video transcripts |
| ✂️ Smart Chunking | Splits transcripts into meaningful chunks |
| 🔎 Semantic Retrieval | Finds the most relevant information |
| 🧠 RAG Pipeline | Combines retrieval with LLM generation |
| 💬 Interactive Chat | Ask questions about the video |
| ⚡ Context-Aware Answers | Answers using retrieved video context |
| 🌐 Streamlit UI | Simple and interactive web interface |

---

# 🧠 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines information retrieval with Large Language Models.

Instead of asking an LLM to answer a question only from its general knowledge, RAG first retrieves relevant information from a knowledge source and provides it to the LLM as context.

### Traditional LLM

```text
User Question
      ↓
     LLM
      ↓
   Answer
```

### RAG

```text
                    ┌───────────────┐
                    │ User Question │
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │    Retriever  │
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │ Relevant Docs │
                    └───────┬───────┘
                            ↓
                 ┌────────────────────┐
                 │       LLM          │
                 │ + Retrieved Context│
                 └──────────┬─────────┘
                            ↓
                    ┌───────────────┐
                    │     Answer    │
                    └───────────────┘
```

---

# 🔄 How This Project Works

```text
          🎥 YouTube Video
                 │
                 ▼
        📝 Get Transcript
                 │
                 ▼
          ✂️ Text Chunking
                 │
                 ▼
        🔢 Generate Embeddings
                 │
                 ▼
          🗄️ Vector Store
                 │
                 ▼
          🔎 Similarity Search
                 │
                 ▼
          📚 Relevant Context
                 │
                 ▼
             🧠 LLM
                 │
                 ▼
          💬 Final Answer
```

---

# 🏗️ Project Architecture

```text
┌───────────────────────────────┐
│          Streamlit UI         │
│                               │
│  YouTube URL + User Question  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│       YouTube Transcript      │
│            Loader             │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│        Text Processing        │
│       Chunking / Cleaning     │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│      Embedding Model          │
│   Text → Vector Representation│
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│         Vector Store          │
│       Similarity Search       │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│           Retriever           │
│      Top Relevant Chunks      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│             LLM               │
│   Context + Question → Answer │
└───────────────┬───────────────┘
                │
                ▼
          💬 Chat Response
```

---

# 🛠️ Tech Stack

### 💻 Programming
- 🐍 Python

### 🤖 AI / LLM
- Large Language Model
- Retrieval-Augmented Generation (RAG)
- Embeddings
- Semantic Search

### 🔗 Frameworks & Libraries
- LangChain
- Streamlit
- YouTube Transcript API
- Vector Database / Vector Store
- Python dotenv

### ☁️ Deployment
- Streamlit

---

# 📂 Project Structure

```text
rag-youtube-chatbot/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/yashrajan-ai/rag-youtube-chatbot.git
cd rag-youtube-chatbot
```

## 2️⃣ Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

Use the API key required by your selected LLM.

⚠️ **Never upload your `.env` file or API keys to GitHub.**

Use `.env.example` instead:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 💬 How to Use

### Step 1️⃣
Enter a **YouTube video URL**.

### Step 2️⃣
The application retrieves the video's transcript.

### Step 3️⃣
The transcript is divided into smaller chunks.

### Step 4️⃣
The chunks are converted into embeddings and stored in a vector store.

### Step 5️⃣
Ask a question about the video.

### Step 6️⃣
The retriever finds the most relevant chunks.

### Step 7️⃣
The LLM uses the retrieved context to generate the answer.

---

# 🧪 Example

### 🎥 Video

```text
https://www.youtube.com/watch?v=example
```

### ❓ Question

```text
What are the main points discussed in this video?
```

### 🤖 Chatbot

```text
The video primarily discusses...

1. ...
2. ...
3. ...
```

---

# 🎯 Why RAG?

A complete YouTube transcript can contain thousands of words.

Sending the entire transcript to an LLM every time can:

❌ Increase token usage  
❌ Increase latency  
❌ Increase cost  
❌ Provide unnecessary context  

RAG solves this by retrieving only the **most relevant information**.

```text
Large Transcript
      │
      ▼
   Chunking
      │
      ▼
  Embeddings
      │
      ▼
Vector Database
      │
      ▼
Relevant Chunks
      │
      ▼
     LLM
      │
      ▼
Better Answer
```

---

# 🚀 Future Improvements

- [ ] 🌍 Support multiple transcript languages
- [ ] 🎙️ Support videos without available transcripts
- [ ] 📚 Chat with multiple YouTube videos
- [ ] 💾 Conversation history
- [ ] 🔎 Improved retrieval
- [ ] 🧠 Hybrid search
- [ ] 📌 Source citations in answers
- [ ] 🎤 Voice-based questions
- [ ] 📄 Export conversation
- [ ] 🌐 Chrome Extension
- [ ] ⚡ Streaming LLM responses

---

# ⚠️ Limitations

Currently, the chatbot may depend on:

- Availability of a YouTube transcript
- Supported transcript languages
- API availability
- LLM/API rate limits
- Quality of the generated transcript

The quality of the final answer also depends on the quality of the retrieved context.

---
```

Recommended `.gitignore`:

```gitignore
.env
venv/
.venv/
__pycache__/
*.pyc
.idea/
.vscode/
```

---

# 📊 RAG Pipeline Summary

| Stage | Purpose |
|---|---|
| 🎥 YouTube | Source of knowledge |
| 📝 Transcript | Extract textual information |
| ✂️ Chunking | Divide text into manageable pieces |
| 🔢 Embeddings | Convert text into vectors |
| 🗄️ Vector Store | Store searchable representations |
| 🔎 Retriever | Find relevant chunks |
| 🧠 LLM | Generate final response |
| 💬 Streamlit | Provide interactive interface |

---

# 🌟 Key Learning Outcomes

This project demonstrates practical knowledge of:

- Retrieval-Augmented Generation
- Large Language Models
- LangChain
- Embeddings
- Vector databases
- Semantic similarity search
- Prompt engineering
- Document processing
- Streamlit application development
- API integration
- AI application deployment

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch:

```bash
git checkout -b feature/new-feature
```

3. Make your changes
4. Commit:

```bash
git commit -m "Add new feature"
```

5. Push:

```bash
git push origin feature/new-feature
```

6. Open a Pull Request

---

# 👨‍💻 Author

## Yash Rajan

🎓 Computer Science & Engineering — AI/ML

💡 Interested in:

- Artificial Intelligence
- Machine Learning
- Generative AI
- Data Science
- Software Development

### 🔗 Connect With Me

- 💻 GitHub: https://github.com/yashrajan-ai
- 🔗 LinkedIn: www.linkedin.com/in/yash-rajan1

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

<div align="center">

### 🎥 Turn YouTube Videos into an Interactive Knowledge Base with RAG 🤖

**Built with Python • LangChain • RAG • LLMs • Streamlit**

⭐ **Star this repository if you like it!** ⭐

</div>
