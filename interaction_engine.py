"""
Drug-Drug Interaction Engine - Standalone Module
Clean. Testable. Production-Ready.
"""

import json
import os
import re
import requests
import zipfile
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
import google.generativeai as genai
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# ============================================================================
# DATA MODELS
# ============================================================================

class InteractionRequest(BaseModel):
    """Request model for checking drug interactions"""
    current_med: str
    new_drug: str


class AIAnalysis(BaseModel):
    """AI analysis result - smart fallback (Gemini → Groq)"""
    severity: Optional[str] = None
    mechanism: Optional[str] = None
    risk: Optional[str] = None
    recommended_action: Optional[str] = None
    patient_explanation: Optional[str] = None
    analysis_available: bool = True


class InteractionResponse(BaseModel):
    """Response model for drug interactions"""
    interaction_found: bool
    ai_analysis: Optional[AIAnalysis] = None
    raw_interaction_text: Optional[str] = None
    message: Optional[str] = None


# ============================================================================
# DATABASE LOADING
# ============================================================================

def load_drug_database(filename: str = "drugbank_comprehensive.json") -> List[Dict]:
    """Load drug database from JSON file or .zip if JSON doesn't exist"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    
    # If .json doesn't exist, try to decompress from .zip
    if not os.path.exists(filepath):
        zip_filepath = filepath + ".zip"
        if os.path.exists(zip_filepath):
            print(f"Decompressing {zip_filepath}...")
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(__file__))
            print(f"✅ Database decompressed")
    
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
            return data.get("drugs", [])
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return []


# Lazy-load database on first use (to save memory on startup)
_DRUG_DB_CACHE = None

def get_drug_database() -> List[Dict]:
    """Get drug database, loading on first use (lazy loading to save startup memory)"""
    global _DRUG_DB_CACHE
    if _DRUG_DB_CACHE is None:
        print("Loading drug database...")
        _DRUG_DB_CACHE = load_drug_database()
        print(f"✅ Loaded {len(_DRUG_DB_CACHE)} drugs")
    return _DRUG_DB_CACHE

# ============================================================================
# NORMALIZATION & LOOKUP LOGIC
# ============================================================================

def normalize(name: str) -> str:
    """Normalize drug name for comparison"""
    return name.strip().lower().replace(" ", "")


def find_drug_in_db(drug_name: str, db: Optional[List[Dict]] = None) -> Optional[Dict]:
    """Find drug in database with regex fuzzy matching"""
    if db is None:
        db = get_drug_database()
    
    normalized_input = normalize(drug_name)
    
    # First try: exact match after normalization
    for drug in db:
        names_to_check = [drug.get("drugName", ""), drug.get("genericName", "")]
        for name in names_to_check:
            if normalize(name) == normalized_input:
                return drug
    
    # Second try: regex partial match (case-insensitive)
    try:
        pattern = re.compile(f"^{re.escape(drug_name)}$", re.IGNORECASE)
        for drug in db:
            names_to_check = [drug.get("drugName", ""), drug.get("genericName", "")]
            for name in names_to_check:
                if pattern.search(name):
                    return drug
    except:
        pass
    
    # Third try: contains match
    for drug in db:
        names_to_check = [drug.get("drugName", ""), drug.get("genericName", "")]
        for name in names_to_check:
            if normalized_input in normalize(name):
                return drug
    
    return None


def find_interaction(current_med: str, new_drug: str) -> Optional[str]:
    """
    Find interaction between two drugs
    Returns interaction description if found
    """
    # Search in database
    drug_a = find_drug_in_db(current_med)
    
    if not drug_a:
        return None
    
    # Check interactions in drug_a
    interactions = drug_a.get("interactions", [])
    
    # If interactions is empty, try drugInteractions
    if not interactions:
        interactions = drug_a.get("drugInteractions", [])
    
    # Handle both string and list formats
    if isinstance(interactions, str):
        # If it's a string containing drug info, parse it
        if "N/A" in interactions or interactions.strip() == "N/A":
            return None
        # Return raw string for Gemini to process
        return interactions if len(interactions) > 10 else None
    
    # Handle list format (DrugBank style with objects)
    if isinstance(interactions, list):
        for interaction in interactions:
            if isinstance(interaction, dict):
                # Check both 'drug' and 'name' field names
                interaction_drug = interaction.get("drug") or interaction.get("name", "")
                if normalize(interaction_drug) == normalize(new_drug):
                    # Return the interaction info with description
                    description = interaction.get("description") or interaction.get("effect", "")
                    severity = interaction.get("severity", "")
                    if severity:
                        return f"{description} (Severity: {severity})"
                    return description if description else "Interaction found"
            elif isinstance(interaction, str):
                if normalize(new_drug) in normalize(interaction):
                    return interaction
    
    return None


# ============================================================================
# AI ANALYSIS INTEGRATION
# ============================================================================

def create_analysis_prompt(interaction_text: str) -> str:
    """Create the clinical analysis prompt for AI"""
    return f"""You are a clinical pharmacology assistant.

Your task is to analyze a drug-drug interaction description and convert it into structured clinical information for a patient safety application.

IMPORTANT RULES:
- Use ONLY the information provided in the interaction description.
- Do NOT invent new medical facts.
- Do NOT add conditions or risks not mentioned or clearly implied.
- If severity is not explicitly stated, estimate conservatively based on risk language.
- Keep explanations concise and medically accurate.
- Return output STRICTLY in valid JSON format.
- Do not include extra text outside JSON.

Classify severity using one of:
LOW
MODERATE
HIGH
CRITICAL

Provide output in this exact JSON structure:

{{
  "severity": "",
  "mechanism": "",
  "risk": "",
  "recommended_action": "",
  "patient_explanation": ""
}}

Definitions:
- severity: clinical seriousness of the interaction
- mechanism: how the interaction happens (pharmacological reason)
- risk: potential clinical outcome
- recommended_action: what should be done (monitor, avoid, adjust dose, etc.)
- patient_explanation: 2-3 simple sentences explaining in plain language what this means

Interaction Description:
\"\"\"
{interaction_text}
\"\"\"

Return ONLY valid JSON."""


def analyze_with_gemini(interaction_text: str) -> Optional[AIAnalysis]:
    """Analyze interaction using Gemini API - returns None if fails (for fallback)"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        prompt = create_analysis_prompt(interaction_text)
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            structured = json.loads(json_str)
            
            return AIAnalysis(
                severity=structured.get("severity", "UNKNOWN"),
                mechanism=structured.get("mechanism", ""),
                risk=structured.get("risk", ""),
                recommended_action=structured.get("recommended_action", ""),
                patient_explanation=structured.get("patient_explanation", "")
            )
    except Exception as e:
        print(f"Gemini analysis unavailable: {type(e).__name__}")
        return None


def analyze_with_groq(interaction_text: str) -> Optional[AIAnalysis]:
    """Analyze interaction using Groq API - fallback option"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = create_analysis_prompt(interaction_text)
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "llama-3.3-70b-versatile",
            "temperature": 0.3
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                structured = json.loads(json_str)
                
                return AIAnalysis(
                    severity=structured.get("severity", "UNKNOWN"),
                    mechanism=structured.get("mechanism", ""),
                    risk=structured.get("risk", ""),
                    recommended_action=structured.get("recommended_action", ""),
                    patient_explanation=structured.get("patient_explanation", "")
                )
        else:
            print(f"Groq API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Groq analysis unavailable: {type(e).__name__}")
        return None


def get_ai_analysis(interaction_text: str) -> AIAnalysis:
    """
    Drug interaction analysis using Groq API ONLY
    (Gemini is reserved for text extraction)
    """
    # Use Groq for drug interaction analysis
    result = analyze_with_groq(interaction_text)
    if result is not None:
        return result
    
    # If Groq fails - return unavailable
    return AIAnalysis(
        severity="UNKNOWN",
        mechanism="AI analysis temporarily unavailable",
        risk="Please consult with a healthcare professional",
        recommended_action="Refer to database information below",
        patient_explanation="We couldn't reach our analysis service. Please review the database information.",
        analysis_available=False
    )


# ============================================================================
# MAIN LOGIC
# ============================================================================

def check_drug_interaction(current_med: str, new_drug: str) -> InteractionResponse:
    """
    Main interaction checker
    1. Normalize input
    2. Find drugs in database
    3. Check for interactions
    4. Dual AI analysis (Gemini + Grok)
    5. Return result
    """
    
    # Edge case: same drug entered twice
    if normalize(current_med) == normalize(new_drug):
        return InteractionResponse(
            interaction_found=False,
            message="You entered the same medication."
        )
    
    # Edge case: empty input
    if not current_med.strip() or not new_drug.strip():
        return InteractionResponse(
            interaction_found=False,
            message="Invalid input: medication names cannot be empty."
        )
    
    # Find interaction
    interaction_text = find_interaction(current_med, new_drug)
    
    # Edge case: drug not found
    if not interaction_text:
        # Verify if drugs exist in database
        db = get_drug_database()
        drug_a = find_drug_in_db(current_med, db)
        drug_b = find_drug_in_db(new_drug, db)
        
        if not drug_a:
            return InteractionResponse(
                interaction_found=False,
                message=f"'{current_med}' not found in database."
            )
        if not drug_b:
            return InteractionResponse(
                interaction_found=False,
                message=f"'{new_drug}' not found in database."
            )
        
        # Both drugs found but no interaction
        return InteractionResponse(
            interaction_found=False,
            message=f"No known interaction found between '{current_med}' and '{new_drug}' in database."
        )
    
    # Analyze with smart fallback (Gemini → Groq)
    ai_analysis = get_ai_analysis(interaction_text)
    
    return InteractionResponse(
        interaction_found=True,
        ai_analysis=ai_analysis,
        raw_interaction_text=interaction_text
    )


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Drug-Drug Interaction Engine",
    description="Standalone Module for checking drug interactions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Redirect to interaction checker page"""
    return RedirectResponse(url="/interaction-checker.html")


@app.get("/interaction-checker.html")
async def get_interaction_checker():
    """Serve the interaction checker HTML file"""
    html_file = os.path.join(os.path.dirname(__file__), "public", "interaction-checker.html")
    if os.path.exists(html_file):
        with open(html_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return {"error": "File not found"}


@app.post("/check-interaction", response_model=InteractionResponse)
async def check_interaction(request: InteractionRequest):
    """
    Check for drug-drug interactions
    
    POST /check-interaction
    Body: {"current_med": "Warfarin", "new_drug": "Aspirin"}
    
    Returns: Structured interaction result with severity, mechanism, risk, action
    """
    result = check_drug_interaction(request.current_med, request.new_drug)
    return result


@app.get("/drugs/search")
async def search_drugs(q: str):
    """Search for drugs by name (for autocomplete)"""
    if len(q) < 2:
        return {"results": []}
    
    q_normalized = normalize(q)
    results = []
    db = get_drug_database()
    
    for drug in db[:100]:  # Limit to first 100
        names = [drug.get("drugName", ""), drug.get("genericName", "")]
        names = [n for n in names if n]
        
        for name in names:
            if q_normalized in normalize(name):
                results.append({
                    "name": name,
                    "generic": drug.get("genericName", ""),
                    "type": drug.get("pharmClass", "N/A")[:50]
                })
                break
        
        if len(results) >= 10:
            break
    
    return {"results": results}


@app.get("/drug-names")
async def get_all_drug_names():
    """Get all available drug names and generic names for autocomplete"""
    drug_names = []
    seen = set()
    db = get_drug_database()
    
    for drug in db:
        drug_name = drug.get("drugName", "").strip()
        generic_name = drug.get("genericName", "").strip()
        
        # Add drug name
        if drug_name and drug_name not in seen:
            drug_names.append(drug_name)
            seen.add(drug_name)
        
        # Add generic name if different
        if generic_name and generic_name != drug_name and generic_name not in seen:
            drug_names.append(generic_name)
            seen.add(generic_name)
    
    return {"drug_names": sorted(drug_names)}


@app.get("/drug-data")
async def get_drug_data(name: str):
    """Retrieve complete drug data from database by name"""
    if not name or len(name) < 1:
        raise HTTPException(status_code=400, detail="Drug name parameter required")
    
    # Search for the drug
    drug = find_drug_in_db(name, DRUG_DB)
    
    if not drug:
        raise HTTPException(status_code=404, detail=f"Drug '{name}' not found in database")
    
    return drug


@app.get("/drugs-visual-info")
async def get_drugs_visual_info(drug1: str, drug2: str):
    """Get visual information for two drugs to display side-by-side"""
    drug_a = find_drug_in_db(drug1)
    drug_b = find_drug_in_db(drug2)
    
    if not drug_a:
        raise HTTPException(status_code=404, detail=f"Drug '{drug1}' not found")
    if not drug_b:
        raise HTTPException(status_code=404, detail=f"Drug '{drug2}' not found")
    
    return {
        "drug1": {
            "name": drug_a.get("drugName", ""),
            "generic": drug_a.get("genericName", ""),
            "class": drug_a.get("pharmClass", ""),
            "indication": drug_a.get("indication", "")[:200] if drug_a.get("indication") else "",
            "side_effects": drug_a.get("sideEffects", "")[:200] if drug_a.get("sideEffects") else "",
            "manufacturer": drug_a.get("manufacturer", ""),
            "interaction_count": drug_a.get("interactionCount", 0)
        },
        "drug2": {
            "name": drug_b.get("drugName", ""),
            "generic": drug_b.get("genericName", ""),
            "class": drug_b.get("pharmClass", ""),
            "indication": drug_b.get("indication", "")[:200] if drug_b.get("indication") else "",
            "side_effects": drug_b.get("sideEffects", "")[:200] if drug_b.get("sideEffects") else "",
            "manufacturer": drug_b.get("manufacturer", ""),
            "interaction_count": drug_b.get("interactionCount", 0)
        }
    }


@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    return {
        "database": "drugbank_comprehensive.json",
        "total_drugs": len(get_drug_database()),
        "unique_drug_names": len(set([d.get("drugName", "") for d in get_drug_database()]))
    }


@app.get("/text-extractor.html")
async def get_text_extractor():
    """Serve the text extractor HTML file"""
    html_file = os.path.join(os.path.dirname(__file__), "public", "text-extractor.html")
    if os.path.exists(html_file):
        with open(html_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return {"error": "File not found"}


@app.post("/extract-text")
async def extract_text(image: UploadFile = File(...)):
    """Extract text from uploaded image using Gemini Vision API"""
    try:
        print(f"Received image: {image.filename}, content_type: {image.content_type}")
        
        # Read image file
        contents = await image.read()
        print(f"Original image size: {len(contents)} bytes")
        
        # Open and resize image if too large
        img = Image.open(io.BytesIO(contents))
        print(f"Image opened: {img.format} {img.size}")
        
        # Resize if larger than 2048px on any side
        max_size = 2048
        if img.width > max_size or img.height > max_size:
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            print(f"Resized to: {img.size}")
        
        # Convert to RGB if needed
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        
        # Use Gemini 2.5 Flash model (has available quota)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Generate text from image
        print("Sending to Gemini (2.5-flash)...")
        response = model.generate_content([
            "Extract all text from this image. Return only the text content, nothing else.",
            img
        ])
        
        extracted_text = response.text.strip()
        print(f"✅ Extracted {len(extracted_text)} characters")
        
        return {
            "success": True,
            "text": extracted_text if extracted_text else "No text detected in image"
        }
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {type(e).__name__}: {error_msg}")
        
        # User-friendly error messages
        if "DeadlineExceeded" in str(type(e).__name__) or "504" in error_msg:
            return {
                "success": False,
                "error": "Request timed out. Try uploading a smaller image."
            }
        elif "quota" in error_msg.lower():
            return {
                "success": False,
                "error": "API quota exceeded. Please try again later."
            }
        else:
            return {
                "success": False,
                "error": f"Text extraction failed: {error_msg[:100]}"
            }


@app.post("/parse-medicine-text")
async def parse_medicine_text(request: dict):
    """Parse extracted text into structured JSON using Groq"""
    try:
        text = request.get("text", "")
        if not text:
            return {"success": False, "error": "No text provided"}
        
        print(f"Parsing medicine text ({len(text)} chars)...")
        
        # Use Groq to parse into JSON
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        prompt = f"""You are a medical text parser. Extract information from the following text (from a medicine strip/package) and return it as a JSON object with these fields:

- "names": array of medicine/drug names found
- "dosage": array of dosage information (e.g., "500mg", "10ml")
- "caution_notes": array of warnings, precautions, or side effects
- "manufacturer": name of manufacturer if found
- "composition": active ingredients if mentioned

Text to parse:
{text}

Return ONLY valid JSON, no other text."""

        data = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            parsed_json = result["choices"][0]["message"]["content"].strip()
            
            # Try to extract JSON if wrapped in markdown
            if "```json" in parsed_json:
                parsed_json = parsed_json.split("```json")[1].split("```")[0].strip()
            elif "```" in parsed_json:
                parsed_json = parsed_json.split("```")[1].split("```")[0].strip()
            
            print(f"✅ Parsed successfully")
            
            return {
                "success": True,
                "parsed_data": parsed_json
            }
        else:
            print(f"❌ Groq API error: {response.status_code}")
            return {
                "success": False,
                "error": f"Parsing failed: {response.status_code}"
            }
    
    except Exception as e:
        print(f"❌ Parse error: {str(e)}")
        return {
            "success": False,
            "error": f"Parsing failed: {str(e)[:100]}"
        }


@app.post("/search-drugs-batch")
async def search_drugs_batch(request: dict):
    """Search for multiple drug names in database with fuzzy matching"""
    try:
        drug_names = request.get("names", [])
        if not drug_names:
            return {"success": False, "results": []}
        
        print(f"Searching for {len(drug_names)} drugs: {drug_names}")
        
        results = []
        db = get_drug_database()
        for name in drug_names:
            # First try exact match
            drug = find_drug_in_db(name, db)
            
            if drug:
                results.append({
                    "found": True,
                    "match_type": "exact",
                    "search_name": name,
                    "drug_name": drug.get("drugName", ""),
                    "generic_name": drug.get("genericName", ""),
                    "pharm_class": drug.get("pharmClass", "N/A")[:100],
                    "description": drug.get("description", "")[:200],
                    "interactions_count": len(drug.get("interactingDrugs", []))
                })
            else:
                # Try fuzzy/prefix matching
                fuzzy_matches = find_fuzzy_drug_matches(name, db)
                
                if fuzzy_matches:
                    results.append({
                        "found": False,
                        "match_type": "fuzzy",
                        "search_name": name,
                        "suggestions": fuzzy_matches
                    })
                else:
                    results.append({
                        "found": False,
                        "match_type": "none",
                        "search_name": name,
                        "suggestions": []
                    })
        
        print(f"✅ Found {sum(1 for r in results if r.get('found'))} exact matches")
        
        return {
            "success": True,
            "results": results
        }
    
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def find_fuzzy_drug_matches(query: str, drug_db: Optional[list] = None, limit: int = 5) -> list:
    """Find drugs matching by prefix and similarity"""
    if drug_db is None:
        drug_db = get_drug_database()
    query_norm = normalize(query).lower()
    query_words = query_norm.split()
    
    matches = []
    
    for drug in get_drug_database():
        drug_name = drug.get("drugName", "").lower()
        generic_name = drug.get("genericName", "").lower()
        
        names_to_check = [drug_name, generic_name]
        
        for check_name in names_to_check:
            if not check_name:
                continue
            
            check_norm = normalize(check_name).lower()
            
            # Calculate similarity score
            score = 0
            
            # Prefix matching (highest priority)
            if check_norm.startswith(query_norm):
                score += 100
            elif query_norm.startswith(check_norm[:3]):
                score += 80
            
            # Word-based matching
            check_words = check_norm.split()
            for q_word in query_words:
                for c_word in check_words:
                    if c_word.startswith(q_word):
                        score += 50
                    elif q_word in c_word or c_word in q_word:
                        score += 30
            
            # Character-based similarity (levenshtein-like)
            if score == 0:
                common_chars = sum(1 for c in query_norm if c in check_norm)
                score = (common_chars / max(len(query_norm), len(check_norm))) * 50
            
            if score > 0:
                matches.append({
                    "score": score,
                    "drug_name": drug.get("drugName", ""),
                    "generic_name": drug.get("genericName", ""),
                    "pharm_class": drug.get("pharmClass", "N/A")[:100],
                    "description": drug.get("description", "")[:200],
                    "interactions_count": len(drug.get("interactingDrugs", []))
                })
    
    # Remove duplicates and sort by score
    seen = set()
    unique_matches = []
    for m in sorted(matches, key=lambda x: x["score"], reverse=True):
        drug_key = m["drug_name"].lower()
        if drug_key not in seen:
            seen.add(drug_key)
            unique_matches.append(m)
            if len(unique_matches) >= limit:
                break
    
    print(f"  Found {len(unique_matches)} fuzzy matches for '{query}'")
    return unique_matches


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
