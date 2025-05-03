import os
import json
import re
import requests
import pandas as pd
import time
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler("b2b_lead_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import TavilyClient from the tavily package (make sure to install it)
from tavily import TavilyClient

# API keys from environment variables (or replace with your keys)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

app = Flask(__name__)

class B2BLeadGenerator:
    def __init__(self, tavily_api_key: str, groq_api_key: str) -> None:
        """
        Initialize the lead generator with the provided API keys.
        """
        self.tavily_api_key = tavily_api_key
        self.groq_api_key = groq_api_key
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        self.groq_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.session = requests.Session()  # Persistent session for API calls
        self.request_count = 0
        self.last_request_time = time.time()

    def _call_groq_api(self, prompt: str, max_tokens: int = 800, max_retries: int = 3) -> str:
        """
        Call the Groq API with a given prompt using a retry mechanism.
        """
        elapsed = time.time() - self.last_request_time
        if elapsed < 1 and self.request_count > 5:
            time.sleep(1 - elapsed)
        
        url = self.groq_endpoint
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1
        }
        
        for attempt in range(max_retries):
            try:
                response = self.session.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"].strip()
                logger.info(f"Groq API response received on attempt {attempt + 1}.")
                self.request_count += 1
                self.last_request_time = time.time()
                return content
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt
                logger.warning(f"Groq API call attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s.")
                time.sleep(wait_time)
        logger.error("Groq API call failed after all retries.")
        return ""

    def _extract_json(self, content: str) -> str:
        """
        Extract a JSON array from the provided string using regex.
        """
        content = re.sub(r"```(?:json)?", "", content)
        content = re.sub(r"```", "", content)
        
        match = re.search(r'(\[.*\])', content, re.DOTALL | re.MULTILINE)
        if match:
            json_str = match.group(1)
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        
        match = re.search(r'(\{.*\})', content, re.DOTALL | re.MULTILINE)
        if match:
            json_str = f"[{match.group(1)}]"
            try:
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                pass
        
        logger.warning("Could not extract valid JSON from response.")
        return ""

    def collect_data(self, queries: dict, max_results: int = 8):
        """
        Collect company data from multiple sources using TavilyClient.
        """
        results_data = []
        
        for platform, query in queries.items():
            try:
                logger.info(f"Searching on {platform} with query: {query}")
                search_response = self.tavily_client.search(
                    query=query, 
                    search_depth="advanced", 
                    max_results=max_results
                )
                for result in search_response.get("results", []):
                    result["platform"] = platform
                    results_data.append(result)
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error searching on {platform}: {e}")
        
        return results_data

    def extract_leads_info(self, search_results):
        """
        Extract structured lead information from search results using the Groq API.
        """
        if not search_results:
            logger.warning("No search results provided for lead extraction.")
            return []
        
        results_by_domain = {}
        for result in search_results:
            url = result.get("url", "")
            if not url:
                continue
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if domain_match:
                domain = domain_match.group(1)
                results_by_domain.setdefault(domain, []).append(result)
        
        all_leads = []
        batch_size = 3
        domain_items = list(results_by_domain.items())
        for i in range(0, len(domain_items), batch_size):
            batch = domain_items[i:i+batch_size]
            batch_results = []
            for domain, results in batch:
                batch_results.extend(results[:2])
            
            context = json.dumps(batch_results, indent=2)
            prompt = (
                "You are a B2B lead generation expert specializing in finding accurate contact information. "
                "Based solely on the search results below, extract company information as a valid JSON array. "
                "Each company should appear only once in your results. For each company, provide these fields:\n\n"
                "1. \"Company Name\" (REQUIRED): The full official name of the company\n"
                "2. \"URL\" (REQUIRED): The main company website URL\n"
                "3. \"LinkedIn Page\" (REQUIRED): The complete LinkedIn company page URL if available\n"
                "4. \"Contact Information\" (REQUIRED): Any available phone numbers, email addresses, or contact page links\n"
                "5. \"Industry\": The business sector or industry classification\n"
                "6. \"Key Decision Maker\": Name and position of a key decision maker if available\n"
                "7. \"Company Size\": Approximate number of employees or size classification\n"
                "8. \"Location\": Main office location or headquarters\n"
                "9. \"Key Products/Services\": Brief description of main offerings\n\n"
                "IMPORTANT RULES:\n"
                "- If any REQUIRED field is not available, make your best estimate but never leave it blank\n"
                "- Prioritize finding DIRECT contact information\n"
                "- Every entry MUST have at least one piece of actionable contact information\n"
                "- Format phone numbers with country code\n"
                "- If the LinkedIn page is not mentioned, use the company name to construct a likely URL\n"
                "- Return only the JSON array with no other text\n\n"
                "Search Results:\n" + context
            )
            
            content = self._call_groq_api(prompt, max_tokens=1200)
            if not content:
                logger.error("No content received from Groq API for lead extraction.")
                continue
                
            json_str = self._extract_json(content)
            if not json_str:
                logger.error("Could not extract JSON array from Groq response for lead extraction.")
                continue
                
            try:
                batch_leads = json.loads(json_str)
                batch_leads = [lead for lead in batch_leads if lead.get("Contact Information")]
                all_leads.extend(batch_leads)
                logger.info(f"Successfully extracted {len(batch_leads)} leads from batch")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error in lead extraction: {e}")
        
        seen_companies = set()
        unique_leads = []
        for lead in all_leads:
            company = lead.get("Company Name", "").strip().lower()
            if company and company not in seen_companies:
                seen_companies.add(company)
                unique_leads.append(lead)
        
        return unique_leads

    def verify_contact_info(self, leads):
        """
        Verify and improve contact information quality.
        """
        if not leads:
            return []
            
        verified_leads = []
        for lead in leads:
            contact_info = lead.get("Contact Information", "")
            confidence_score = 0
            
            if re.search(r'[\w\.-]+@[\w\.-]+\.\w+', contact_info):
                confidence_score += 3
            if re.search(r'(?:\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', contact_info):
                confidence_score += 3
            if "contact" in contact_info.lower() and "http" in contact_info.lower():
                confidence_score += 1
                
            linkedin_url = lead.get("LinkedIn Page", "")
            if not linkedin_url or "linkedin.com" not in linkedin_url:
                company_name = lead.get("Company Name", "").strip()
                if company_name:
                    formatted_name = re.sub(r'[^\w\s-]', '', company_name).lower()
                    formatted_name = re.sub(r'\s+', '-', formatted_name)
                    linkedin_url = f"https://www.linkedin.com/company/{formatted_name}"
                    lead["LinkedIn Page"] = linkedin_url
                    
            company_url = lead.get("URL", "")
            if company_url and not company_url.startswith(("http://", "https://")):
                company_url = f"https://{company_url}"
                lead["URL"] = company_url
                
            if company_url:
                domain_match = re.search(r'https?://(?:www\.)?([^/]+)', company_url)
                lead["Domain"] = domain_match.group(1) if domain_match else "Unknown"
            else:
                lead["Domain"] = "Unknown"
            
            lead["Contact Confidence"] = "High" if confidence_score >= 3 else "Medium" if confidence_score > 0 else "Low"
            lead["Last Verified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            verified_leads.append(lead)
            
        return verified_leads

    def format_for_display(self, leads):
        """
        Format leads for optimal display.
        """
        display_leads = []
        for lead in leads:
            display_lead = {
                "Company Name": lead.get("Company Name", "Unknown"),
                "Industry": lead.get("Industry", "Unknown"),
                "Contact Details": lead.get("Contact Information", "Not available"),
                "Company Website": lead.get("URL", "Not available"),
                "Domain": lead.get("Domain", "Unknown"),
                "LinkedIn": lead.get("LinkedIn Page", "Not available"),
                "Key Contact": lead.get("Key Decision Maker", "Unknown"),
                "Location": lead.get("Location", "Unknown"),
                "Company Size": lead.get("Company Size", "Unknown"),
                "Services": lead.get("Key Products/Services", "Unknown"),
                "Confidence": lead.get("Contact Confidence", "Low")
            }
            if "Personalized Outreach Approach" in lead:
                display_lead["Outreach Strategy"] = lead["Personalized Outreach Approach"]
            if "Best Contact Method" in lead:
                display_lead["Best Contact Method"] = lead["Best Contact Method"]
            if "Pain Points" in lead:
                display_lead["Pain Points"] = lead["Pain Points"]
            if "Conversation Starters" in lead:
                display_lead["Conversation Starters"] = lead["Conversation Starters"]
            display_leads.append(display_lead)
        return display_leads

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_leads():
    """
    Expects a JSON payload with:
      - industry: string
      - platforms: list of strings (e.g., ["facebook", "linkedin", "reddit"])
      - resultsCount: integer (number of results per platform)
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid input"}), 400

    industry = data.get("industry")
    platforms = data.get("platforms")
    results_count = data.get("resultsCount", 8)

    if not industry or not platforms:
        return jsonify({"error": "Missing industry or platforms"}), 400

    # Construct queries based on selected platforms
    queries = {}
    for platform in platforms:
        if platform.lower() == "facebook":
            queries["Facebook"] = f"{industry} B2B companies site:facebook.com"
        elif platform.lower() == "linkedin":
            queries["LinkedIn"] = f"{industry} B2B companies site:linkedin.com/company"
        elif platform.lower() == "reddit":
            queries["Reddit"] = f"{industry} B2B companies site:reddit.com"
    
    lead_gen = B2BLeadGenerator(tavily_api_key=TAVILY_API_KEY, groq_api_key=GROQ_API_KEY)
    search_results = lead_gen.collect_data(queries, max_results=int(results_count))
    if not search_results:
        return jsonify({"error": "No data collected. Please check your API keys and network connection."}), 500

    leads = lead_gen.extract_leads_info(search_results)
    if not leads:
        return jsonify({"error": "No leads with contact details extracted."}), 500

    leads = lead_gen.verify_contact_info(leads)
    display_leads = lead_gen.format_for_display(leads)
    
    return jsonify(display_leads)

if __name__ == "__main__":
    # Run the Flask app in debug mode for development
    app.run(debug=True)
