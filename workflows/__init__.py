from fastapi import APIRouter
from .recommendation import build_recommendation_workflow
from .makeup_exam import build_makeup_exam_workflow

router = APIRouter(prefix="/workflows", tags=["workflows"])

# Recommendation Letter Workflow
@router.post("/recommendation")
async def recommendation_flow(user_input: str):
    workflow = build_recommendation_workflow()
    return workflow.invoke({"user_input": user_input})

# Make-up Exam Workflow (new)
@router.post("/makeup_exam")
async def makeup_exam_flow(user_input: str):
    workflow = build_makeup_exam_workflow()
    return workflow.invoke({"user_input": user_input})