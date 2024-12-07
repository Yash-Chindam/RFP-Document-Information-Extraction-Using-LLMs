import os
import json
import requests
import streamlit as st
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pypdf import PdfReader
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Retrieve the Hugging Face API Key from the environment variable
hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")


class DocumentExtractor:
    def __init__(self, hf_token):
        """
        Initialize the document extractor with Hugging Face Inference API
        
        :param hf_token: Hugging Face API token
        """
        self.hf_token = hf_token
        self.inference_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B-Instruct"
        self.headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        
    def extract_pdf_text(self, pdf_file):
        """
        Extract text from a PDF file
        
        :param pdf_file: Path or file-like object of PDF
        :return: Extracted text
        """
        text = ""
        try:
            reader = PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
        return text

    def extract_html_text(self, html_file):
        """
        Extract text from an HTML file
        
        :param html_file: Path or file-like object of HTML
        :return: Extracted text
        """
        text = ""
        try:
            soup = BeautifulSoup(html_file, 'html.parser')
            text = soup.get_text()
        except Exception as e:
            st.error(f"Error reading HTML: {e}")
        return text

    def chunk_text(self, text, max_tokens=4096):
        """
        Chunk text into smaller segments
        
        :param text: Input text
        :param max_tokens: Maximum tokens per chunk
        :return: List of text chunks
        """
        return [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]

    def extract_information(self, text_chunks):
        """
        Extract structured information using Hugging Face Inference API
        
        :param text_chunks: List of text chunks
        :return: Combined extracted information
        """
        fields = [
            "Bid Number", "Title", "Due Date", "Bid Submission Type", 
            "Term of Bid", "Pre Bid Meeting", "Installation", 
            "Bid Bond Requirement", "Delivery Date", "Payment Terms",
            "Additional Documentation", "MFG Registration", 
            "Contract/Cooperative", "Model Number", "Part Number", 
            "Product", "Contact Info", "Company Name", 
            "Bid Summary", "Product Specification"
        ]
        
        combined_response = ""
        for chunk in text_chunks:
            prompt = f"""
            Extract the following fields from the document:
            {', '.join(fields)}
            
            Document Text:
            {chunk}
            
            Provide structured information with each field followed by its value. 
            Format your response with each field on a new line, like:
            Field1: Value1
            Field2: Value2
            """
            
            try:
                # Use requests to directly call the Hugging Face Inference API
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 500,
                        "temperature": 0.2,
                        "return_full_text": False
                    }
                }
                
                response = requests.post(
                    self.inference_url, 
                    headers=self.headers, 
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()[0]['generated_text']
                    combined_response += result + "\n"
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                
            except Exception as e:
                st.error(f"Inference API error: {e}")
        
        return combined_response

    def extract_fields_from_response(self, response):
        """
        Parse the model's response into a structured dictionary
        
        :param response: Model's text response
        :return: Structured data dictionary
        """
        structured_data = {}
        for field in [
            "Bid Number", "Title", "Due Date", "Bid Submission Type", 
            "Term of Bid", "Pre Bid Meeting", "Installation", 
            "Bid Bond Requirement", "Delivery Date", "Payment Terms",
            "Additional Documentation", "MFG Registration", 
            "Contract/Cooperative", "Model Number", "Part Number", 
            "Product", "Contact Info", "Company Name", 
            "Bid Summary", "Product Specification"
        ]:
            # Simple extraction logic
            start_index = response.find(field + ":")
            if start_index != -1:
                start_index += len(field) + 1
                end_index = response.find("\n", start_index)
                value = response[start_index:end_index].strip() if end_index != -1 else response[start_index:].strip()
                structured_data[field] = value
        
        return structured_data

# Streamlit App
def streamlit_app(hf_token):
    """
    Create a Streamlit web application for document extraction
    
    :param hf_token: Hugging Face API token
    """
    st.title("📄 Document Information Extractor")
    
    # Initialize extractor
    extractor = DocumentExtractor(hf_token)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF or HTML files", 
        type=['pdf', 'html'], 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        all_extracted_data = []
        
        for uploaded_file in uploaded_files:
            # Determine file type
            if uploaded_file.name.endswith('.pdf'):
                text = extractor.extract_pdf_text(uploaded_file)
            elif uploaded_file.name.endswith('.html'):
                text = extractor.extract_html_text(uploaded_file)
            else:
                st.error("Unsupported file type")
                continue
            
            # Extract text chunks
            text_chunks = extractor.chunk_text(text)
            
            # Extract information
            extracted_info = extractor.extract_information(text_chunks)
            
            # Parse structured data
            structured_data = extractor.extract_fields_from_response(extracted_info)
            
            # Store extracted data
            all_extracted_data.append({
                "document": uploaded_file.name,
                "data": structured_data
            })
        
        # Display extracted information
        for doc_data in all_extracted_data:
            st.subheader(f"Document: {doc_data['document']}")
            st.json(doc_data['data'])
        
        # Option to download JSON
        json_str = json.dumps(all_extracted_data, indent=4)
        st.download_button(
            label="Download Extracted Data as JSON",
            data=json_str,
            file_name="extracted_document_data.json",
            mime="application/json"
        )

# Main execution
def main():
    # Get Hugging Face token from environment variable
    HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')
    
    if not HUGGING_FACE_API_KEY:
        st.error("Please set the HUGGING_FACE_API_KEY environment variable")
        return
    
    # Run Streamlit app
    streamlit_app(HUGGING_FACE_API_KEY)

if __name__ == "__main__":
    main()

