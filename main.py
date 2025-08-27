import streamlit as st
import os
from typing import TypedDict, List, Dict
from pprint import pformat

from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from tavily import TavilyClient


# --- 1. 상태 정의 (State Definition) ---
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    is_relevant: bool
    is_hallucination: bool
    search_attempts: int
    regeneration_attempts: int
    error: str
    final_answer: str
    final_sources: List[Dict]


# --- 2. 워크플로우 구성 함수 ---
# Streamlit 앱에서 API 키를 받은 후 워크플로우를 동적으로 생성하기 위해 함수로 묶습니다.
def create_workflow(llm, retriever, tavily_client):
    """LangGraph 워크플로우를 생성하고 컴파일합니다."""

    # --- Grader 설정 ---
    relevance_system = """You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n Provide the binary score as a JSON with a single key 'score' and no premable or explanation."""
    relevance_prompt = ChatPromptTemplate.from_messages(
        [("system", relevance_system), ("human", "question: {question}\n\n document: {document} ")])
    retrieval_grader = relevance_prompt | llm | JsonOutputParser()

    hallucination_system = """You are a grader assessing whether an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation."""
    hallucination_prompt = ChatPromptTemplate.from_messages(
        [("system", hallucination_system), ("human", "documents: {documents}\n\n answer: {generation} ")])
    hallucination_grader = hallucination_prompt | llm | JsonOutputParser()

    # --- 노드 함수 정의 ---
    def retrieve_docs(state: GraphState) -> dict:
        st.session_state.logs.append("---NODE: Docs Retrieval---")
        question = state['question']
        documents = retriever.invoke(question)
        return {"documents": documents, "search_attempts": 0, "regeneration_attempts": 0, "error": None}

    def relevance_checker(state: GraphState) -> dict:
        st.session_state.logs.append("---NODE: Relevance Checker---")
        question = state['question']
        documents = state['documents']
        if not documents:
            st.session_state.logs.append("  -> 관련성 검사 결과: 검색된 문서 없음")
            return {"is_relevant": False}
        is_relevant = False
        for d in documents:
            doc_txt = d.page_content
            score = retrieval_grader.invoke({"question": question, "document": doc_txt})
            grade = score['score']
            if grade == 'yes':
                st.session_state.logs.append(f"  -> 관련성 검사 결과: 관련 문서 발견 (score: {grade})")
                is_relevant = True
                break
            else:
                st.session_state.logs.append(f"  -> 관련성 검사 결과: 관련 없는 문서 (score: {grade})")
        return {"is_relevant": is_relevant}

    def generate_answer(state: GraphState) -> dict:
        st.session_state.logs.append("---NODE: Generate Answer---")
        question = state['question']
        documents = state['documents']
        regeneration_attempts = state['regeneration_attempts']
        system = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise"""
        prompt = ChatPromptTemplate.from_messages(
            [("system", system), ("human", "question: {question}\n\n context: {context} ")])
        rag_chain = prompt | llm | StrOutputParser()
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"generation": generation, "regeneration_attempts": regeneration_attempts + 1}

    def hallucination_checker(state: GraphState) -> dict:
        st.session_state.logs.append("---NODE: Hallucination Checker---")
        documents = state['documents']
        generation = state['generation']
        score = hallucination_grader.invoke({"documents": documents, "generation": generation})
        grade = score['score']
        if grade == 'yes':
            st.session_state.logs.append("  -> 환각 검사 결과: 답변이 문서에 기반함 (score: yes)")
            is_hallucination = False
        else:
            st.session_state.logs.append("  -> 환각 검사 결과: 답변이 문서에 기반하지 않음 (score: no)")
            is_hallucination = True
        return {"is_hallucination": is_hallucination}

    def search_tavily(state: GraphState) -> dict:
        st.session_state.logs.append("---NODE: Search Tavily---")
        question = state['question']
        search_attempts = state['search_attempts']
        if not tavily_client:
            st.session_state.logs.append("  -> Tavily API 키가 설정되지 않아 웹 검색을 건너뜁니다.")
            return {"documents": [], "search_attempts": search_attempts + 1}
        response = tavily_client.search(query=question, max_results=3)
        web_documents = [
            Document(page_content=obj["content"], metadata={"url": obj["url"], "title": obj.get("title", question)}) for
            obj in response['results']]
        return {"documents": web_documents, "search_attempts": search_attempts + 1}

    def prepare_final_answer(state: GraphState) -> dict:
        st.session_state.logs.append("---NODE: Prepare Final Answer---")
        generation = state['generation']
        documents = state['documents']
        sources = []
        for doc in documents:
            url = doc.metadata.get('url') or doc.metadata.get('source')
            title = doc.metadata.get('title', url)
            if url:
                sources.append({"title": str(title), "url": str(url)})
        unique_sources = [dict(t) for t in {tuple(d.items()) for d in sources}]
        return {"final_answer": generation, "final_sources": unique_sources}

    # --- 조건부 엣지 함수 정의 ---
    def decide_after_relevance_check(state: GraphState) -> str:
        st.session_state.logs.append("---DECISION: After Relevance Check---")
        if state["is_relevant"]:
            st.session_state.logs.append("  -> 결정: 관련성 있음, 답변 생성으로 이동")
            return "generate"
        elif state["search_attempts"] < 1:
            st.session_state.logs.append("  -> 결정: 관련성 없음, 웹 검색으로 이동")
            return "search"
        else:
            st.session_state.logs.append("  -> 결정: 웹 검색 후에도 관련성 없음, 워크플로우 종료")
            return "end_not_relevant"

    def decide_after_hallucination_check(state: GraphState) -> str:
        st.session_state.logs.append("---DECISION: After Hallucination Check---")
        if not state["is_hallucination"]:
            st.session_state.logs.append("  -> 결정: 환각 없음, 최종 답변 준비로 이동")
            return "useful"
        elif state["regeneration_attempts"] < 2:
            st.session_state.logs.append("  -> 결정: 환각 있음, 답변 재성성으로 이동")
            return "regenerate"
        else:
            st.session_state.logs.append("  -> 결정: 재성성 후에도 환각 있음, 워크플로우 종료")
            return "end_hallucination"

    # --- 워크플로우 구성 ---
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_docs)
    workflow.add_node("relevance_check", relevance_checker)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("hallucination_check", hallucination_checker)
    workflow.add_node("search", search_tavily)
    workflow.add_node("prepare_final_answer", prepare_final_answer)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "relevance_check")
    workflow.add_edge("search", "relevance_check")
    workflow.add_edge("generate", "hallucination_check")
    workflow.add_edge("prepare_final_answer", END)
    workflow.add_conditional_edges("relevance_check", decide_after_relevance_check,
                                   {"generate": "generate", "search": "search", "end_not_relevant": END})
    workflow.add_conditional_edges("hallucination_check", decide_after_hallucination_check,
                                   {"useful": "prepare_final_answer", "regenerate": "generate",
                                    "end_hallucination": END})

    return workflow.compile()


# --- 3. Streamlit UI 및 실행 로직 ---

# VectorDB 설정은 리소스 소모가 크므로 캐싱합니다.
@st.cache_resource
def setup_retriever():
    """문서를 로드하고 VectorDB를 설정합니다."""
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(docs_list)
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    )
    return vectorstore.as_retriever()


def main():
    st.set_page_config(page_title="LangGraph RAG Demo", page_icon="🔗")
    st.title("LangGraph RAG Demo")

    # 사이드바에 API 키 입력 필드 추가
    with st.sidebar:
        st.header("API Keys")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        tavily_api_key = st.text_input("Tavily API Key", type="password")

    # 사용자 질문 입력
    question = st.text_input("질문을 입력하세요:", placeholder="What are the types of agent memory?")

    if st.button("실행"):
        if not openai_api_key or not tavily_api_key:
            st.error("사이드바에 OpenAI 및 Tavily API 키를 입력해주세요.")
            return

        # API 키를 환경 변수로 설정
        os.environ["OPENAI_API_KEY"] = openai_api_key
        os.environ["TAVILY_API_KEY"] = tavily_api_key

        # 로깅을 위한 세션 상태 초기화
        st.session_state.logs = []

        with st.spinner("워크플로우 실행 중..."):
            # 컴포넌트 초기화
            retriever = setup_retriever()
            llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
            tavily_client = TavilyClient(api_key=tavily_api_key)

            # 워크플로우 생성
            app = create_workflow(llm, retriever, tavily_client)

            # 로그를 표시할 컨테이너
            log_container = st.expander("실행 로그", expanded=True)
            log_placeholder = log_container.empty()

            inputs = {"question": question}
            final_state = None

            # 워크플로우 스트리밍 실행 및 로그 출력
            for output in app.stream(inputs, {"recursion_limit": 15}):
                for key, value in output.items():
                    log_placeholder.text("\n".join(st.session_state.logs))
                final_state = list(output.values())[0]

            # 최종 결과 표시
            st.success("워크플로우 실행 완료!")

            if final_state and final_state.get("final_answer"):
                st.subheader("✅ 최종 답변")
                st.write(final_state["final_answer"])

                st.subheader("📚 출처")
                for source in final_state["final_sources"]:
                    st.markdown(f"- [{source['title']}]({source['url']})")
            else:
                st.error("❌ 워크플로우가 답변 생성 전에 종료되었습니다.")
                if not final_state.get("is_relevant") and final_state.get("search_attempts", 0) > 0:
                    st.info("   이유: 웹 검색 후에도 관련 문서를 찾지 못했습니다. (failed: not relevant)")
                elif final_state.get("is_hallucination") and final_state.get("regeneration_attempts", 0) > 1:
                    st.info("   이유: 답변 재성성 후에도 환각이 감지되었습니다. (failed: hallucination)")


if __name__ == "__main__":
    main()