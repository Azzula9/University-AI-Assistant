from langgraph.graph import Graph
from langchain_core.messages import HumanMessage
from llm.llm_service import get_llm
import json
from typing import Dict, Any

def build_recommendation_workflow():
    workflow = Graph()

    def extract_fields(state: Dict[str, Any]) -> Dict[str, Any]:
        llm = get_llm()
        try:
            response = llm.invoke([
                HumanMessage(content=f"""Extract the following as JSON:
                Input: {state['user_input']}
                Output format: {{"student_name": "...", "course_name": "...", "prof_name": "..."}}
                """)
            ])
            fields = json.loads(response.content)
            return {"extracted_fields": fields}
        except Exception as e:
            return {"error": f"Field extraction failed: {str(e)}"}

    def generate_draft(state: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in state:
            return state  
        
        try:
            llm = get_llm()
            fields = state["extracted_fields"]
            response = llm.invoke([
                HumanMessage(content=f"""
                Write a formal recommendation letter with:
                - Student: {fields['student_name']}
                - Course: {fields['course_name']}
                - Professor: {fields['prof_name']}
                """)
            ])
            return {"draft": response.content, **state}
        except Exception as e:
            return {"error": f"Draft generation failed: {str(e)}"}

    workflow.add_node("extract_fields", extract_fields)
    workflow.add_node("generate_draft", generate_draft)
    workflow.set_entry_point("extract_fields")
    workflow.add_edge("extract_fields", "generate_draft")
    
    def finalize(state: Dict[str, Any]) -> Dict[str, Any]:
        if not state:
            return {"error": "Empty final state"}
        return state
        
    workflow.add_node("finalize", finalize)
    workflow.add_edge("generate_draft", "finalize")
    
    return workflow