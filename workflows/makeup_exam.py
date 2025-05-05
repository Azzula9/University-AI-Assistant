from langgraph.graph import Graph
from langchain_core.messages import HumanMessage
from llm.llm_service import get_llm

def build_makeup_exam_workflow():
    workflow = Graph()
    
    def extract_fields(state):
        llm = get_llm()
        response = llm.invoke([
            HumanMessage(content=f"Extract: student name, course, and reason from: {state['user_input']}")
        ])
        return {"fields": response.content}
    
    def generate_request(state):
        llm = get_llm()
        prompt = f"""
        Generate make-up exam request with:
        Student: {state['fields']['student_name']}
        Course: {state['fields']['course_name']}
        Reason: {state['fields']['reason']}
        """
        return {"request": llm.invoke([HumanMessage(content=prompt)]).content}
    
    workflow.add_node("extract_fields", extract_fields)
    workflow.add_node("generate_request", generate_request)
    workflow.set_entry_point("extract_fields")
    workflow.add_edge("extract_fields", "generate_request")
    
    return workflow