import json
import re
import time
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path
from loguru import logger
from PIL import Image
from difflib import SequenceMatcher

from utils import time_it


class DataExtractor:

    def __init__(self, vision_client, llm_client, llm_provider="gemini"):
        self.vision_client = vision_client
        self.llm_client = llm_client
        self.llm_provider = llm_provider
        self.request_times = []
        self.max_requests_per_minute = 25
        logger.info(f"DataExtractor initialized with {llm_provider} provider")
    
    def _wait_for_rate_limit(self):
        now = datetime.now()
        self.request_times = [t for t in self.request_times if (now - t).total_seconds() < 60]
        
        if len(self.request_times) >= self.max_requests_per_minute:
            oldest = self.request_times[0]
            wait_time = 60 - (now - oldest).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit protection: waiting {wait_time:.1f} seconds")
                time.sleep(wait_time + 1)
        
        self.request_times.append(now)
    
    @time_it
    def extract_text_from_image(self, image_path: str) -> Optional[str]:
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                
                if not Path(image_path).exists():
                    logger.error(f"Image file not found: {image_path}")
                    return None
                
                img = Image.open(image_path)
                prompt = "Extract all visible text from this document image. Return only the extracted text without any explanations or formatting."
                
                response = self.vision_client.generate_content([prompt, img])
                
                if response and response.text:
                    extracted_text = response.text.strip()
                    logger.info(f"Successfully extracted {len(extracted_text)} characters from {Path(image_path).name}")
                    return extracted_text
                else:
                    logger.warning(f"No text returned from vision API for {Path(image_path).name}")
                    return None
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit, retrying in 35 seconds (attempt {attempt + 1}/{max_retries})")
                        time.sleep(35)
                        continue
                    else:
                        logger.error(f"Rate limit exceeded after {max_retries} retries for {image_path}")
                else:
                    logger.error(f"Failed to extract text from {image_path}: {e}")
                return None
        
        return None
    
    @time_it
    def structure_text_with_llm(self, raw_text: str, doc_type: str) -> Optional[Dict]:
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                prompt = self._create_extraction_prompt(raw_text, doc_type)
                
                if self.llm_provider == "gemini":
                    response = self.llm_client.generate_content(prompt)
                    response_text = response.text
                else:
                    logger.error(f"Unsupported LLM provider: {self.llm_provider}")
                    return None
                
                structured_data = self._parse_llm_response(response_text)
                if structured_data:
                    logger.info(f"Successfully structured {doc_type} data")
                    return structured_data
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit during structuring, retrying (attempt {attempt + 1}/{max_retries})")
                        time.sleep(35)
                        continue
                    else:
                        logger.error("Rate limit exceeded during text structuring")
                else:
                    logger.error(f"Failed to structure text: {e}")
                return None
        
        return None
    
    def _create_extraction_prompt(self, raw_text: str, doc_type: str) -> str:
        prompt = """You are extracting structured data from a {doc_type} document.

Extract the following fields from the text and return ONLY a valid JSON object:
{{
  "full_name": "",
  "father_name": "",
  "date_of_birth": "",
  "address": "",
  "phone_number": "",
  "email": "",
  "aadhaar_number": "",
  "pan_number": "",
  "employee_id": "",
  "account_number": ""
}}

Important Rules:
- Return ONLY the JSON object, no markdown formatting, no code blocks, no explanations
- Use empty string "" for fields not found in the document
- Extract dates in DD-MM-YYYY or DD/MM/YYYY format if present
- For Employment Letters: SKIP the company/HR phone number at the top. Extract ONLY the employee's personal phone if listed in the employee details section.
- For Bank Statements and Government IDs: Extract the individual's personal phone number
- Extract Aadhaar as 12 digits (remove spaces and dashes)
- Extract PAN as 10 characters (5 letters + 4 digits + 1 letter)
- Extract complete address including house number, street, city, state, and pincode

Document Text:
{raw_text}

Return only the JSON:"""
        
        return prompt.replace("{doc_type}", doc_type).replace("{raw_text}", raw_text)
    
    def _parse_llm_response(self, response_text: str) -> Optional[Dict]:
        try:
            text = response_text.strip()
            
            if text.startswith("```"):
                lines = text.split("\n")
                json_lines = [line for line in lines if "```" not in line and "json" not in line.lower()]
                text = "\n".join(json_lines).strip()
            
            parsed_data = json.loads(text)
            return parsed_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response text was: {response_text[:200]}")
            return None


class VerificationEngine:

    def __init__(self):
        self.address_similarity_threshold = 0.70
        logger.info("VerificationEngine initialized with strict validation rules")

    @time_it
    def run_all_verifications(self, extracted_data: Dict[str, Dict]) -> Dict:
        logger.info("Starting verification checks")
        
        docs = list(extracted_data.values())
        
        results = {
            "rule_1_name_match": self._verify_name_match(docs),
            "rule_2_dob_match": self._verify_dob_match(docs),
            "rule_3_address_match": self._verify_address_match(docs),
            "rule_4_phone_match": self._verify_phone_match(docs),
            "rule_5_father_name_match": self._verify_father_name_match(docs),
            "rule_6_pan_format": self._verify_pan_format(docs),
            "rule_7_aadhaar_format": self._verify_aadhaar_format(docs)
        }

        all_passed = all(r["status"] == "PASS" for r in results.values())
        results["overall_status"] = "VERIFIED" if all_passed else "FAILED"
        
        logger.info(f"Verification complete. Overall Status: {results['overall_status']}")
        return results

    def _verify_name_match(self, docs: List[Dict]) -> Dict:
        names = [d.get("full_name") for d in docs if d.get("full_name")]
        
        if len(names) < 2:
            return {"status": "PASS", "reason": "Insufficient data"}
        
        normalized_names = [self._normalize_name(n) for n in names]
        unique_names = set(normalized_names)
        
        if len(unique_names) == 1:
            return {"status": "PASS"}
        else:
            return {"status": "FAIL", "reason": f"Name mismatch found: {list(unique_names)}"}

    def _verify_dob_match(self, docs: List[Dict]) -> Dict:
        dobs = [d.get("date_of_birth") for d in docs if d.get("date_of_birth")]
        
        if len(dobs) < 2:
            return {"status": "PASS", "reason": "Insufficient data"}
        
        normalized_dobs = [self._normalize_date(d) for d in dobs]
        unique_dobs = set(normalized_dobs)
        
        if len(unique_dobs) == 1:
            return {"status": "PASS"}
        else:
            return {"status": "FAIL", "reason": f"Date of birth mismatch: {list(unique_dobs)}"}

    def _verify_address_match(self, docs: List[Dict]) -> Dict:
        addresses = [d.get("address") for d in docs if d.get("address")]
        
        if len(addresses) < 2:
            return {"status": "PASS", "reason": "Insufficient data"}
        
        normalized_addrs = [self._normalize_address(a) for a in addresses]
        
        base_addr = normalized_addrs[0]
        for addr in normalized_addrs[1:]:
            similarity = SequenceMatcher(None, base_addr, addr).ratio()
            if similarity < self.address_similarity_threshold:
                return {"status": "FAIL", "reason": f"Address similarity {similarity:.2f} below threshold {self.address_similarity_threshold}"}
        
        return {"status": "PASS"}

    def _verify_phone_match(self, docs: List[Dict]) -> Dict:
        phones = [d.get("phone_number") for d in docs if d.get("phone_number")]
        
        if len(phones) < 2:
            logger.info("Phone verification skipped: Employment letters don't contain personal phone numbers")
            return {"status": "PASS", "reason": "Insufficient data (employment letters have company HR phone only)"}
        
        normalized_phones = [self._normalize_phone(p) for p in phones]
        unique_phones = set(normalized_phones)
        
        if len(unique_phones) == 1:
            return {"status": "PASS"}
        else:
            return {"status": "FAIL", "reason": f"Phone number mismatch: {list(unique_phones)}"}

    def _verify_father_name_match(self, docs: List[Dict]) -> Dict:
        father_names = [d.get("father_name") for d in docs if d.get("father_name")]
        
        if len(father_names) < 2:
            return {"status": "PASS", "reason": "Insufficient data"}
        
        normalized_names = [self._normalize_name(n) for n in father_names]
        unique_names = set(normalized_names)
        
        if len(unique_names) == 1:
            return {"status": "PASS"}
        else:
            return {"status": "FAIL", "reason": f"Father's name mismatch: {list(unique_names)}"}

    def _verify_pan_format(self, docs: List[Dict]) -> Dict:
        pans = [d.get("pan_number") for d in docs if d.get("pan_number")]
        
        if not pans:
            return {"status": "PASS", "reason": "No PAN number found"}
        
        pan = pans[0].strip().upper()
        pan_pattern = r'^[A-Z]{5}[0-9]{4}[A-Z]$'
        
        if re.match(pan_pattern, pan):
            return {"status": "PASS"}
        else:
            return {"status": "FAIL", "reason": f"Invalid PAN format: {pan}. Expected format: ABCDE1234F"}

    def _verify_aadhaar_format(self, docs: List[Dict]) -> Dict:
        aadhaars = [d.get("aadhaar_number") for d in docs if d.get("aadhaar_number")]
        
        if not aadhaars:
            return {"status": "PASS", "reason": "No Aadhaar number found"}
        
        aadhaar = re.sub(r'[\s\-]', '', aadhaars[0])
        
        if re.match(r'^\d{12}$', aadhaar):
            return {"status": "PASS"}
        else:
            return {"status": "FAIL", "reason": f"Invalid Aadhaar format: {aadhaar}. Expected 12 digits"}

    def _normalize_name(self, name: str) -> str:
        if not name:
            return ""
        name = name.strip().lower()
        name = re.sub(r'\s+', ' ', name)
        return name

    def _normalize_date(self, date_str: str) -> str:
        if not date_str:
            return ""
        date_clean = re.sub(r'[\s\-/]', '', date_str)
        return date_clean

    def _normalize_phone(self, phone: str) -> str:
        if not phone:
            return ""
        digits = re.sub(r'\D', '', phone)
        if len(digits) > 10:
            digits = digits[-10:]
        return digits

    def _normalize_address(self, address: str) -> str:
        if not address:
            return ""
        
        addr = address.lower()
        addr = re.sub(r'\bhouse no\.?\s*', 'h no ', addr)
        addr = re.sub(r'\bh\.?no\.?\s*', 'h no ', addr)
        addr = re.sub(r'\bpin:?\s*', 'pin ', addr)
        addr = re.sub(r'\bpin code:?\s*', 'pin ', addr)
        addr = re.sub(r'[^\w\s]', ' ', addr)
        addr = re.sub(r'\s+', ' ', addr).strip()
        return addr