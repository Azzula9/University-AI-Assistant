import pdfplumber
from typing import Optional
import os
from datetime import datetime

class PDFProcessor:
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> Optional[str]:
        """
        Extracts all text from a PDF file
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as a single string, or None if extraction fails
        """
        try:
            full_text = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:  # Only append if text was extracted
                        full_text.append(text)
            return "\n".join(full_text) if full_text else None
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return None

    @staticmethod
    def save_extracted_text(text: str, original_filename: str) -> str:
        """
        Saves extracted text to a file in assets folder
        
        Args:
            text: Extracted text content
            original_filename: Original PDF filename for naming
            
        Returns:
            Path to the saved text file
        """
        try:
            os.makedirs("assets/extracted_text", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(original_filename)[0]
            txt_filename = f"extracted_{timestamp}_{base_name}.txt"
            txt_path = os.path.join("assets/extracted_text", txt_filename)
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
                
            return txt_path
        except Exception as e:
            print(f"Error saving extracted text: {e}")
            raise