from fastapi import APIRouter, UploadFile, File, HTTPException
import os
from datetime import datetime
from fastapi.responses import JSONResponse
from controllers.pdf_processor import PDFProcessor
from controllers.text_chunker import TextChunker
from vectordb.qdrant_service import VectorDB

router = APIRouter(prefix="/upload", tags=["upload"])

os.makedirs("assets", exist_ok=True)

@router.post("/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"upload_{timestamp}_{file.filename}"
        pdf_path = os.path.join("assets", pdf_filename)
        
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        extracted_text = PDFProcessor.extract_text_from_pdf(pdf_path)
        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract text from PDF"
            )
        
        txt_path = PDFProcessor.save_extracted_text(
            extracted_text,
            file.filename
        )
        
        chunker = TextChunker()
        chunks = chunker.chunk_text(extracted_text)
        chunks_path = chunker.save_chunks(chunks, file.filename)
        
        vector_db = VectorDB()
        vector_db.store_chunks(chunks)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "File processed and stored in Qdrant Cloud",
                "pdf_path": pdf_path,
                "text_path": txt_path,
                "chunks_path": chunks_path,
                "chunks_count": len(chunks),
                "collection": "university_knowledge",
                "filename": file.filename
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing file: {str(e)}"
        )