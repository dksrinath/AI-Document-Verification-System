import os
import json
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv
from loguru import logger
import google.generativeai as genai

from utils import setup_logging, time_it
from verifier import DataExtractor, VerificationEngine


class DocumentVerificationSystem:

    def __init__(self):
        setup_logging()
        logger.info("Initializing Document Verification System")
        load_dotenv()

        self.vision_client = self._initialize_vision_client()
        self.llm_client, self.llm_provider = self._initialize_llm_client()
        self.data_extractor = DataExtractor(self.vision_client, self.llm_client, self.llm_provider)
        self.verification_engine = VerificationEngine()
        
        self.data_dir = Path("data")
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("System initialization complete")

    def _initialize_vision_client(self):
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set in .env file")
            
            genai.configure(api_key=api_key)
            client = genai.GenerativeModel('gemini-2.5-pro')
            logger.info("Google Gemini Vision client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Vision client: {e}")
            raise

    def _initialize_llm_client(self):
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        try:
            if provider == "gemini":
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not set in .env file")
                
                genai.configure(api_key=api_key)
                client = genai.GenerativeModel('gemini-2.5-pro')
                logger.info("Google Gemini LLM client initialized successfully")
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
            
            return client, provider
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise

    @time_it
    def process_person_documents(self, person_dir: Path) -> Optional[Dict]:
        person_id = person_dir.name
        logger.info(f"Processing documents for {person_id}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf'}
        image_files = [f for f in person_dir.iterdir() if f.suffix.lower() in image_extensions]

        if not image_files:
            logger.warning(f"No document images found for {person_id}")
            return None

        extracted_data = {}
        for idx, image_file in enumerate(image_files, 1):
            logger.info(f"Processing {image_file.name}")
            doc_key = f"document_{idx}"
            doc_type = self._infer_document_type(image_file.name)
            
            raw_text = self.data_extractor.extract_text_from_image(str(image_file))

            if raw_text:
                structured = self.data_extractor.structure_text_with_llm(raw_text, doc_type)
                if structured:
                    extracted_data[doc_key] = structured
                else:
                    logger.warning(f"Failed to structure data from {image_file.name}")
            else:
                logger.warning(f"No text extracted from {image_file.name}")

        if not extracted_data:
            logger.error(f"No data extracted for {person_id}")
            return None

        verification_results = self.verification_engine.run_all_verifications(extracted_data)

        return {
            "person_id": person_id,
            "extracted_data": extracted_data,
            "verification_results": verification_results,
            "overall_status": verification_results["overall_status"]
        }

    def _infer_document_type(self, filename: str) -> str:
        fn = filename.lower()
        if 'aadhaar' in fn or 'aadhar' in fn:
            return 'Government ID'
        elif 'pan' in fn:
            return 'PAN Card'
        elif 'bank' in fn or 'statement' in fn:
            return 'Bank Statement'
        elif 'employment' in fn or 'letter' in fn:
            return 'Employment Letter'
        elif 'government' in fn or 'govt' in fn or 'id' in fn:
            return 'Government ID'
        return 'Unknown Document'

    @time_it
    def run(self):
        logger.info("Starting Document Verification System")
        
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            return

        person_dirs = [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('P')]
        
        if not person_dirs:
            logger.warning("No person directories found in data folder")
            return

        logger.info(f"Found {len(person_dirs)} person directories to process")
        all_results = []
        
        for person_dir in sorted(person_dirs):
            try:
                result = self.process_person_documents(person_dir)
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {person_dir.name}: {e}")
                all_results.append({
                    "person_id": person_dir.name,
                    "error": str(e),
                    "overall_status": "ERROR"
                })

        output_file = self.output_dir / "verification_results.json"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results successfully written to {output_file}")

            total = len(all_results)
            verified = sum(1 for r in all_results if r.get("overall_status") == "VERIFIED")
            failed = sum(1 for r in all_results if r.get("overall_status") == "FAILED")
            errors = sum(1 for r in all_results if r.get("overall_status") == "ERROR")
            
            logger.info(f"Processing Summary: Total={total}, Verified={verified}, Failed={failed}, Errors={errors}")
        except Exception as e:
            logger.error(f"Failed to write results file: {e}")
            raise


def main():
    try:
        system = DocumentVerificationSystem()
        system.run()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.critical(f"Critical error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
