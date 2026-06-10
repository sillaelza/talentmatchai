import pdfplumber
from docx import Document
from typing import Optional
import os


class ResumeParser:
    """Parser for extracting text from resume files (PDF, DOCX, TXT)."""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> Optional[str]:
        """
        Extract text from a PDF file using pdfplumber.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as string, or None if extraction fails
        """
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip() if text else None
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {e}")
            return None
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> Optional[str]:
        """
        Extract text from a DOCX file using python-docx.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Extracted text as string, or None if extraction fails
        """
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip() if text else None
        except Exception as e:
            print(f"Error extracting text from DOCX {file_path}: {e}")
            return None
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> Optional[str]:
        """
        Extract text from a TXT file.
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            Extracted text as string, or None if extraction fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            return text.strip() if text else None
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                return text.strip() if text else None
            except Exception as e:
                print(f"Error extracting text from TXT {file_path}: {e}")
                return None
        except Exception as e:
            print(f"Error extracting text from TXT {file_path}: {e}")
            return None
    
    @staticmethod
    def parse_resume(file_path: str) -> Optional[str]:
        """
        Parse a resume file and extract text based on file extension.
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            Extracted text as string, or None if extraction fails
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return ResumeParser.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            return ResumeParser.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            return ResumeParser.extract_text_from_txt(file_path)
        else:
            print(f"Unsupported file format: {file_extension}")
            return None


if __name__ == "__main__":
    # Example usage
    parser = ResumeParser()
    
    # Test with a file (uncomment to test)
    # test_file = "path/to/your/resume.pdf"
    # text = parser.parse_resume(test_file)
    # if text:
    #     print(f"Successfully extracted {len(text)} characters from {test_file}")
    #     print(text[:500])  # Print first 500 characters
