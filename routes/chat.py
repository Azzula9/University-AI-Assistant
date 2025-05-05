from fastapi import APIRouter
from langchain_core.messages import SystemMessage, HumanMessage
from llm.llm_service import get_llm
from typing import Optional
from vectordb.qdrant_service import VectorDB
from langchain_qdrant import Qdrant
from langgraph.graph import Graph


router = APIRouter(prefix="/chat", tags=["chat"])


GENERAL_SYSTEM_MESSAGE = """You are a university policy assistant. Only answer using the context provided below.

Use this step-by-step format:
1- Summarize the relevant parts of the context related to the question.
2- Identify any specific policy section name or number, if available.
3- Provide a **final, formal, and precise response** based strictly on the context. Do **not** show the intermediary steps in the answer.

The final response should be a full sentence that directly answers the question based on the context. **Do not include the process of steps in your response.**

Only respond based on provided context. Use bullet points where appropriate. If the answer is not found, say: "The answer is not available in the provided context."
"""

EXAM_SYSTEM_MESSAGE = """You are an academic assistant. Only use the context to answer exam/project questions.

Think step-by-step:
1. Identify if the context includes:
   - Deadlines
   - Submission methods
   - Materials
   - Grading criteria
2. Present info clearly using bullet points.

Do not add information not found in the context.
"""

CATEGORY_PROMPT = """
You are a classifier bot.

Classify the following question into ONE of these categories:
1. General Question (about policies/rules)
2. Administrative Request (Recommendation Letter/Make-up Exam)
3. Project/Exam Info (specific deadlines/details)

Only respond with the number 1, 2, or 3 — nothing else.

Question:
<<<
{question}
>>>

Category:"""


def detect_category(question: str) -> int:
    llm = get_llm()
    response = llm.invoke(CATEGORY_PROMPT.format(question=question))
    content = response.content.strip()
    if content not in {"1", "2", "3"}:
        raise ValueError(f"Unexpected category response: {content}")
    return int(content)


def get_workflow(question: str) -> Optional[Graph]:
    q = question.lower()
    if any(keyword in q for keyword in ["recommendation", "recommend", "letter"]):
        from workflows.recommendation import build_recommendation_workflow
        return build_recommendation_workflow()
    elif any(keyword in q for keyword in ["make-up", "makeup", "exam"]):
        from workflows.makeup_exam import build_makeup_exam_workflow
        return build_makeup_exam_workflow()
    return None


def run_direct_qa(question: str, category: int) -> Optional[str]:
    vector_store = Qdrant(
        client=VectorDB().client,
        collection_name="university_knowledge",
        embeddings=VectorDB().embeddings
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(question)  
    context = "\n---\n".join([doc.page_content for doc in docs])

    system_prompt = (
        GENERAL_SYSTEM_MESSAGE if category == 1 else
        EXAM_SYSTEM_MESSAGE if category == 3 else
        None
    )
    if not system_prompt:
        return None

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Context:
<<<
{context}
>>>

Question:
<<<
{question}
>>>

Final Answer:
""")
    ]

    llm = get_llm()
    response = llm.invoke(messages)
    return response.content


@router.post("/")
async def unified_chat_endpoint(question: str):
    try:
        if not question:
            return {"type": "error", "message": "Empty question"}
            
        category = detect_category(question)

        direct_result = run_direct_qa(question, category)
        if direct_result:
            return {"type": "direct_answer", "result": direct_result}

        workflow = get_workflow(question)
        if not workflow:
            return {"type": "unhandled", "message": "No matching workflow"}
            
        compiled = workflow.compile()
        if not compiled:
            return {"type": "error", "message": "Workflow compilation failed"}
            
        result = compiled.invoke({"user_input": question})
        
        if not result:
            return {"type": "error", "message": "Workflow returned empty result"}
            
        if "error" in result:
            return {
                "type": "workflow_error",
                "error": result["error"],
                "details": result.get("raw_response")
            }
            
        return {
            "type": "workflow_result",
            "data": {
                "fields": result.get("extracted_fields", {}),
                "draft": result.get("draft", "No draft generated")
            }
        }

    except Exception as e:
        return {
            "type": "error",
            "message": f"System error: {str(e)}",
            "suggestion": "Check server logs"
        }