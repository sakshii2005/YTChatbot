import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import os
from dotenv import load_dotenv
load_dotenv()

# --- Streamlit App Title ---
st.title("📽️ YouTube RAG Chat with HuggingFace")

# --- Input fields ---
video_id = st.text_input("🎬 Enter YouTube Video ID (e.g., Gfr50f6ZBvo)")
question = st.text_input("❓ Enter your question")

# --- If both inputs are provided ---
if video_id and question:
    try:
        if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
            st.error("Missing HUGGINGFACEHUB_API_TOKEN. Add it to Streamlit Secrets or .env.")
            st.stop()

        # Step 1: Fetch transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        st.success("Transcript fetched successfully!")

        with st.expander("📜 View Transcript"):
            st.write(transcript[:3000] + "..." if len(transcript) > 3000 else transcript)

        # Step 2: Split and wrap as documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = splitter.split_text(transcript)
        documents = [Document(page_content=chunk) for chunk in texts]

        # Step 3: Embeddings and Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()

        # Step 4: Prompt Template
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            Context:
            {context}

            Question:
            {question}
            """,
            input_variables=["context", "question"]
        )

        # Step 5: HuggingFace LLM
        hf_llm = HuggingFaceEndpoint(
            repo_id="microsoft/Phi-3-mini-4k-instruct",
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        chat = ChatHuggingFace(llm=hf_llm)

        # Step 6: Format function
        def format_docs(retrieved_docs):
            return "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Step 7: Modular Chain using RunnableParallel
        parallel_chain = RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })

        main_chain = parallel_chain | prompt | chat | StrOutputParser()

        # Step 8: Run the final chain
        response = main_chain.invoke(question)

        st.subheader("💡 Answer")
        st.write(response)

    except TranscriptsDisabled:
        st.error("❌ Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("❌ No transcript found for this video.")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
