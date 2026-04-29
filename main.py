import pdfplumber
import uuid
import io
import re
from keywords import LEGAL_KEYWORDS
from fastapi import Body
import requests
import spacy
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import os
from dotenv import load_dotenv
import hashlib

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Path, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


import google.generativeai as genai


MAX_FILE_SIZE = 5 * 1024 * 1024  
TRIVIAL_WORDS = {"law", "agreement", "will", "document", "party", "shall"}
SUPPORTED_FILE_TYPES = ['.pdf']

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("⚠️ spaCy model not found. Run: python -m spacy download en_core_web_sm")


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not found. Some features will be disabled.")
    GEMINI_ENABLED = False
else:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_ENABLED = True



# -------------------------
# Data Models
# -------------------------
@dataclass
class DocumentClause:
    id: str
    text: str
    category: str
    keywords: List[str]
    score: int
    score_reasons: List[str]
    page: int
    position: str
    entities: Dict
    risk_level: Optional[str] = None
    risk_reasoning: Optional[str] = None

@dataclass
class DocumentRecord:
    id: str
    filename: str
    file_hash: str
    upload_timestamp: str
    size_kb: float
    total_pages: int
    status: str
    metadata: Dict
    clauses: List[DocumentClause]
    summaries: Dict
    last_updated: str


class DocumentUpdateRequest(BaseModel):
    filename: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None

class DocumentPatchRequest(BaseModel):
    reviewed: Optional[bool] = None
    archived: Optional[bool] = None
    priority: Optional[str] = None

class ClauseSearchParams(BaseModel):
    document_id: str
    category: Optional[str] = None
    query: Optional[str] = None
    min_score: Optional[int] = 5
    limit: Optional[int] = 50

document_storage: Dict[str, DocumentRecord] = {}

# -------------------------
# FastAPI App
# -------------------------
app = FastAPI(
    title="📄 Legal Document Analyzer API",
    description="Complete PDF analysis with clause parsing, risk assessment, and document management",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)




class DocumentStructureParser:
    clause_patterns = [
        r"^\s*(\d+\.\d+(?:\.\d+)*)\s+(.+)",      
        r"^\s*\(([a-z])\)\s+(.+)",             
        r"^\s*([A-Z]+)\.\s+(.+)",              
        r"^\s*Article\s+(\d+)\s*[:\-]?\s*(.+)", 
        r"^\s*Section\s+(\d+)\s*[:\-]?\s*(.+)", # Section 1:
    ]
    
    heading_patterns = [
        r"^\s*([A-Z\s]{3,})\s*$",
        r"^\s*\*\*(.+)\*\*\s*$",
        r"^\s*_{2,}(.+)_{2,}\s*$",
    ]

    def parse_document_structure(self, text: str) -> List[Dict]:
        lines = text.split('\n')
        clauses, current_clause = [], None
        clause_counter = 1

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            heading = self._match_heading(line_stripped)
            if heading:
                if current_clause:
                    clauses.append(current_clause)
                current_clause = {
                    "id": str(uuid.uuid4()),
                    "clause_number": clause_counter,
                    "text": heading,
                    "type": "heading",
                    "line_start": i,
                    "line_end": i
                }
                clause_counter += 1
                continue

            clause_start = self._match_clause_start(line_stripped)
            if clause_start:
                if current_clause:
                    clauses.append(current_clause)
                current_clause = {
                    "id": str(uuid.uuid4()),
                    "clause_number": clause_counter,
                    "text": clause_start[1],
                    "type": "clause",
                    "line_start": i,
                    "line_end": i
                }
                clause_counter += 1
                continue

            if current_clause and current_clause["type"] == "clause":
                current_clause["text"] += " " + line_stripped
                current_clause["line_end"] = i

        if current_clause:
            clauses.append(current_clause)
        return clauses

    def _match_heading(self, line: str) -> Optional[str]:
        for pattern in self.heading_patterns:
            match = re.match(pattern, line)
            if match:
                return match.group(1).strip()
        if len(line) < 50 and line.isupper() and len(line.split()) > 1:
            return line
        return None

    def _match_clause_start(self, line: str) -> Optional[Tuple[str, str]]:
        for pattern in self.clause_patterns:
            match = re.match(pattern, line)
            if match:
                return match.group(1), match.group(2)
        return None

# -------------------------
# Smart Keyword Detector
# -------------------------
import re
import spacy
from typing import List, Dict, Tuple, Optional

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

class SmartKeywordDetector:
    def __init__(self):
        self.parser = DocumentStructureParser()

    def analyze_document_with_pages(
        self,
        text_pages: List[Tuple[int, str]],
        category_filter: Optional[str] = None
    ) -> Dict:
        detected_clauses, global_entities = [], {
            "parties": set(),
            "dates": set(),
            "money": set(),
            "jurisdictions": set(),
            "laws": set(),
            "numbers": set(),
        }

        for page_num, text in text_pages:
            clauses = self.parser.parse_document_structure(text)
            for clause in clauses:
                analyzed = self._analyze_clause(clause, category_filter, page_num)
                for c in analyzed:
                    c.page = page_num
                    detected_clauses.append(c)
                    # merge entities
                    for k, v in c.entities.items():
                        global_entities[k].update(v)

        detected_clauses.sort(key=lambda x: x.score, reverse=True)
        return {
            "structured_clauses": detected_clauses,
            "global_entities": {k: list(v) for k, v in global_entities.items()},
            "document_summary": self._generate_summary(detected_clauses),
        }

    def _analyze_clause(
        self,
        clause: Dict,
        category_filter: Optional[str],
        page_num: int
    ) -> List["DocumentClause"]:
        results = []
        for cat, keywords in LEGAL_KEYWORDS.items():
            if category_filter and cat.lower() != category_filter.lower():
                continue
            found = [
                kw for kw in keywords
                if kw.lower() not in TRIVIAL_WORDS
                and self._build_pattern(kw).search(clause["text"])
            ]
            if found:
                score, reasons = self._score_clause(clause["text"], found, clause["type"])
                entities = self._extract_entities(clause["text"])
                results.append(DocumentClause(
                    id=clause["id"],
                    text=clause["text"][:500] + "..." if len(clause["text"]) > 500 else clause["text"],
                    category=cat,
                    keywords=found,
                    score=score,
                    score_reasons=reasons,
                    page=page_num,
                    position="heading" if clause["type"] == "heading" else "clause_start",
                    entities=entities
                ))
        return results

    def _extract_entities(self, text: str) -> Dict:
        doc = nlp(re.sub(r"\s+", " ", text))
        parties, dates, money, locations, laws, numbers = [], [], [], [], [], []

        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON"] and len(ent.text.strip()) > 2:
                parties.append(ent.text.strip())
            elif ent.label_ == "DATE":
                dates.append(ent.text.strip())
            elif ent.label_ == "MONEY":
                money.append(ent.text.strip())
            elif ent.label_ == "GPE":
                locations.append(ent.text.strip())
            elif ent.label_ == "LAW":
                laws.append(ent.text.strip())
            elif ent.label_ == "CARDINAL":
                numbers.append(ent.text.strip())

        return {
            "parties": list(set(parties)),
            "dates": list(set(dates)),
            "money": list(set(money)),
            "jurisdictions": list(set(locations)),
            "laws": list(set(laws)),
            "numbers": list(set(numbers)),
        }

    def _build_pattern(self, keyword: str) -> re.Pattern:
        if " " in keyword:
            return re.compile(r"\b" + r"\s*[-\s]\s*".join(keyword.split()) + r"\b", re.I)
        return re.compile(rf"\b{keyword}(?:s|ing|ed)?\b", re.I)

    def _score_clause(
        self,
        text: str,
        keywords: List[str],
        clause_type: str
    ) -> Tuple[int, List[str]]:
        score, reasons = 1, [f"Contains {len(keywords)} relevant keyword(s)"]

        if clause_type == "heading":
            score += 5
            reasons.append("Found in heading")

        total_keywords = sum(len(re.findall(rf"\b{re.escape(kw)}\b", text, re.I)) for kw in keywords)
        if total_keywords > len(keywords):
            score += 3
            reasons.append("Multiple keyword instances")

        cluster_bonus = self._keyword_clusters(text, keywords)
        if cluster_bonus > 0:
            score += cluster_bonus
            reasons.append("Clustered legal terms")

        if re.search(r"(section|clause|article|paragraph)\s+\d+", text, re.I):
            score += 2
            reasons.append("Legal structure reference")

        formal_terms = ["hereby", "whereas", "notwithstanding", "pursuant to"]
        formal_count = sum(1 for f in formal_terms if f in text.lower())
        score += min(formal_count, 3)
        if formal_count > 0:
            reasons.append(f"{formal_count} formal term(s)")

        return min(score, 10), reasons

    def _keyword_clusters(self, text: str, keywords: List[str]) -> int:
        clusters = {
            "termination": ["termination", "breach", "default", "cure period", "notice"],
            "liability": ["liability", "damages", "indemnification", "hold harmless", "limitation"],
            "payment": ["payment", "invoice", "late fees", "interest", "penalty"],
        }
        bonus = 0
        for terms in clusters.values():
            match = sum(1 for term in terms if any(term.lower() in kw.lower() for kw in keywords))
            if match >= 2:
                bonus += min(match, 3)
        return bonus

    def _generate_summary(self, clauses: List["DocumentClause"]) -> Dict:
        if not clauses:
            return {"overview": "No significant legal clauses detected"}

        by_cat = {}
        for c in clauses:
            by_cat.setdefault(c.category, []).append(c)

        summaries = {}
        for cat, cls in by_cat.items():
            high = [c for c in cls if c.score >= 7]
            summaries[cat] = {
                "total_clauses": len(cls),
                "high_importance": len(high),
                "top_keywords": list(set([kw for c in cls for kw in c.keywords]))[:5],
                "max_score": max(c.score for c in cls)
            }

        return {
            "overview": f"{len(clauses)} clauses across {len(by_cat)} categories",
            "categories": summaries,
            "critical_findings": [
                {
                    "clause_id": c.id,
                    "category": c.category,
                    "score": c.score,
                    "preview": c.text[:100] + "..." if len(c.text) > 100 else c.text
                }
                for c in clauses if c.score >= 8
            ][:5]
        }




    
        
# -------------------------
# Extractive summary
# -------------------------     

def make_extractive_summary(selected: List[DocumentClause], llm_error: Optional[str] = None) -> str:
    
    if not selected:
        return "No clauses available to summarize."

    # Group by category
    by_cat: Dict[str, List[DocumentClause]] = {}
    for c in selected:
        by_cat.setdefault(c.category, []).append(c)

    lines = []
    if llm_error:
        lines.append(f"(LLM unavailable, produced extractive summary) Reason: {llm_error}\n")

    lines.append("Executive Summary (Extractive)")
    lines.append(f"- Total input clauses: {len(selected)}")
    lines.append(f"- Categories covered: {', '.join(sorted(by_cat.keys()))}\n")

    # Per-category bullets
    for cat, items in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        top_kw = []
        for c in items:
            for k in c.keywords:
                if k not in top_kw:
                    top_kw.append(k)
        top_kw = top_kw[:6]
        lines.append(f"{cat}:")
        lines.append(f"  - Clauses: {len(items)} | Top keywords: {', '.join(top_kw) if top_kw else '—'}")
        # Include up to 2 short previews
        for c in items[:2]:
            preview = c.text.strip().replace("\n", " ")
            preview = (preview[:220] + "...") if len(preview) > 220 else preview
            lines.append(f"  - Sample: {preview}")
        lines.append("")

    # Quick risks heuristic
    high_risk_terms = ["terminate", "breach", "default", "penalty", "damages", "liability", "indemnify"]
    risk_hits = [c for c in selected if any(t in c.text.lower() for t in high_risk_terms)]
    lines.append("Key Risks (heuristic):")
    if risk_hits:
        for c in risk_hits[:5]:
            lines.append(f"  - [{c.category}] score {c.score}, page {c.page}: {', '.join([k for k in c.keywords[:5]]) or '—'}")
    else:
        lines.append("  - No high-risk indicators detected in selected clauses.")
    lines.append("")

    # Action items heuristic
    lines.append("Action Items:")
    lines.append("  - Review termination/indemnity language for exposure caps and carve-outs.")
    lines.append("  - Confirm payment terms (amounts, due dates, late fees, interest).")
    lines.append("  - Validate confidentiality scope and IP ownership/usage rights.")
    lines.append("  - Check dispute resolution forum, governing law, and escalation steps.")
    lines.append("  - Fill any missing definitions or ambiguous obligations.")

    return "\n".join(lines)

        
        

# -------------------------
# LLM Services
# -------------------------
def generate_gemini_summary(
    clauses: List[DocumentClause],
    min_high_score: int = 7,
    min_items: int = 5,
    max_items: int = 12
) -> Tuple[str, Dict]:
   
    sorted_clauses = sorted(clauses, key=lambda c: c.score, reverse=True)
    critical = [c for c in sorted_clauses if c.score >= min_high_score]
    fallback_used = False
    selected: List[DocumentClause] = critical[:max_items]

    if len(selected) < min_items:
        selected = sorted_clauses[:max_items]
        fallback_used = True

    if not selected:
        return ("No clauses available to summarize.", {
            "llm_used": False,
            "fallback_used": True,
            "selected_count": 0,
            "clause_selection": "none",
            "selected_ids": []
        })

    selected_ids = [c.id for c in selected]

    def _clause_line(c: DocumentClause, idx: int) -> str:
        kw = ", ".join(c.keywords[:6]) if c.keywords else "—"
        ent_parties = ", ".join(c.entities.get("parties", [])[:3]) if c.entities else ""
        ent_money = ", ".join(c.entities.get("money", [])[:2]) if c.entities else ""
        ents = "; ".join(filter(None, [ent_parties, ent_money]))
        preview = (c.text[:300] + "...") if len(c.text) > 300 else c.text
        return (
            f"[{idx}] Category: {c.category} | Score: {c.score} | Page: {c.page} | "
            f"Keywords: {kw} | Entities: {ents}\n"
            f"Text: {preview}\n"
        )

    prompt_blocks = "\n\n".join(_clause_line(c, i+1) for i, c in enumerate(selected))
    system_prompt = (
        "You are a senior legal analyst. Create an executive, concise summary for business stakeholders. "
        "Goals: (1) Key obligations for each side, (2) High-risk terms (termination, liability, indemnity, penalties), "
        "(3) Payment terms, (4) Confidentiality/IP, (5) Dispute resolution and governing law, (6) Missing or vague terms. "
        "Write bullets grouped by theme, then a short 'Action Items' list (max 5 items). Avoid quoting long text; be specific."
    )
    user_prompt = f"{system_prompt}\n\nCLAUSES:\n{prompt_blocks}\n\nNow provide:\n- Executive Summary\n- Key Risks\n- Payment & Financial Terms\n- Confidentiality/IP\n- Dispute Resolution & Governing Law\n- Action Items (max 5)\n"

    if not GEMINI_ENABLED:
        return (make_extractive_summary(selected), {
            "llm_used": False,
            "fallback_used": fallback_used,
            "selected_count": len(selected),
            "clause_selection": "critical>=7" if not fallback_used else "top_by_score",
            "selected_ids": selected_ids
        })

    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(user_prompt)
        text = (response.text or "").strip()
        if not text:
            raise ValueError("Empty response from LLM")
        return (text, {
            "llm_used": True,
            "fallback_used": fallback_used,
            "selected_count": len(selected),
            "clause_selection": "critical>=7" if not fallback_used else "top_by_score",
            "selected_ids": selected_ids
        })
    except Exception as e:
        return (make_extractive_summary(selected, llm_error=str(e)), {
            "llm_used": False,
            "fallback_used": True,
            "selected_count": len(selected),
            "clause_selection": "top_by_score",
            "selected_ids": selected_ids,
            "llm_error": str(e),
        })



def analyze_clause_risk(clause: DocumentClause) -> Tuple[str, str]:
    """Analyze risk level of a clause using simple heuristics"""
    text_lower = clause.text.lower()
    
    # High risk indicators
    high_risk_terms = ["terminate", "breach", "default", "penalty", "damages", "liability", "indemnify"]
    medium_risk_terms = ["payment", "deadline", "notice", "compliance", "obligation"]
    
    high_count = sum(1 for term in high_risk_terms if term in text_lower)
    medium_count = sum(1 for term in medium_risk_terms if term in text_lower)
    
    if high_count >= 2 or clause.score >= 8:
        return "Red", f"High risk due to {high_count} critical terms and importance score {clause.score}"
    elif high_count >= 1 or medium_count >= 2 or clause.score >= 6:
        return "Yellow", f"Medium risk with {high_count} high-risk and {medium_count} medium-risk terms"
    else:
        return "Green", "Low risk - routine clause with standard terms"

# -------------------------
# Utility Functions
# -------------------------
def generate_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def extract_text_from_pdf(content: bytes) -> List[Tuple[int, str]]:
    text_pages = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text_pages.append((i, page.extract_text() or ""))
    return text_pages

# -------------------------
# API Endpoints
# -------------------------

@app.get("/health")
async def health_check():
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "spacy_model": "en_core_web_sm loaded",
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "supported_formats": SUPPORTED_FILE_TYPES,
        "documents_in_storage": len(document_storage)
    }
    
  
    if GEMINI_ENABLED:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content("Test connection")
            health_status["gemini_api"] = "Connected"
        except Exception as e:
            health_status["gemini_api"] = f"Error: {str(e)}"
            health_status["status"] = "degraded"
    else:
        health_status["gemini_api"] = "Disabled - API key not configured"
    
    return health_status


@app.post("/extract")
async def extract_pdf(
    file_url: str = Body(..., embed=True, description="Publicly accessible PDF URL"),
    category: str = Query(None, description="Filter by category"),
    min_score: int = Query(5, description="Minimum clause importance score"),
    max_results: int = Query(50, description="Maximum clauses to return")
):
   

    
    def is_pdf_url(url: str) -> bool:
        """Check if URL is likely a PDF file"""
        url_lower = url.lower()
        
      
        if url_lower.endswith(".pdf"):
            return True
            
       
        if "utfs.io" in url_lower:
            return True
            
        
        if any(domain in url_lower for domain in ["amazonaws.com", "googleusercontent.com", "dropbox.com", "onedrive.com"]):
            return True
            
        return False
    
    if not is_pdf_url(file_url):
        raise HTTPException(
            status_code=400, 
            detail="URL does not appear to be a PDF file. Supported: .pdf extension or recognized cloud storage URLs."
        )

  
    try:
        print(f"Downloading PDF from: {file_url}")
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        contents = response.content
        print(f"Downloaded {len(contents)} bytes")
        
     
        if not contents.startswith(b'%PDF'):
            raise HTTPException(
                status_code=400,
                detail="Downloaded file is not a valid PDF (missing PDF header)"
            )
            
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=400, detail="Timeout while downloading PDF file")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max {MAX_FILE_SIZE/(1024*1024):.1f} MB allowed."
        )

  
    document_id = str(uuid.uuid4())
    file_hash = generate_file_hash(contents)


    for existing_doc in document_storage.values():
        if existing_doc.file_hash == file_hash:
            return {"message": "Duplicate file detected", "existing_document_id": existing_doc.id}

  
    try:
        text_pages = extract_text_from_pdf(contents)
        print(f"Extracted text from {len(text_pages)} pages")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")


    detector = SmartKeywordDetector()
    analysis = detector.analyze_document_with_pages(text_pages, category)


    filtered_clauses = [c for c in analysis["structured_clauses"] if c.score >= min_score][:max_results]

    document_record = DocumentRecord(
        id=document_id,
        filename=file_url.split("/")[-1] + ".pdf",
        file_hash=file_hash,
        upload_timestamp=datetime.now(timezone.utc).isoformat(),
        size_kb=round(len(contents) / 1024, 2),
        total_pages=len(text_pages),
        status="processed",
        metadata={"category_filter": category, "min_score": min_score, "source_url": file_url},
        clauses=filtered_clauses,
        summaries={},
        last_updated=datetime.now(timezone.utc).isoformat()
    )

    document_storage[document_id] = document_record

 
    serializable_clauses = [
        {
            "clause_id": c.id,
            "text": c.text,
            "category": c.category,
            "keywords_found": c.keywords,
            "importance_score": c.score,
            "score_explanation": c.score_reasons,
            "page_number": c.page,
            "position_type": c.position,
            "extracted_entities": c.entities
        }
        for c in filtered_clauses
    ]

    print(f"Successfully processed document {document_id} with {len(serializable_clauses)} clauses")

    return {
        "document_id": document_id,
        "filename": document_record.filename,
        "page_count": len(text_pages),
        "document_metadata": {
            "size_kb": document_record.size_kb,
            "upload_timestamp": document_record.upload_timestamp,
            "total_pages": document_record.total_pages
        },
        "key_findings": {
            "total_clauses_analyzed": len(analysis["structured_clauses"]),
            "returned_clauses": len(filtered_clauses),
            "high_importance_clauses": len([c for c in filtered_clauses if c.score >= 8]),
            "categories_detected": len(set(c.category for c in filtered_clauses)),
            "critical_entities": analysis["global_entities"]
        },
        "clauses": serializable_clauses
    }

@app.post("/summarize")
async def summarize_document(
    document_id: Optional[str] = Query(None, description="Document ID to summarize"),
    file_url: Optional[str] = Query(None, description="PDF URL to process directly"),
    file: Optional[UploadFile] = File(None, description="PDF file to summarize directly"),
    min_high_score: int = Query(7, description="Score threshold for 'critical' clauses before fallback applies"),
    min_items: int = Query(5, description="Minimum number of clauses to feed the summarizer"),
    max_items: int = Query(12, description="Maximum number of clauses to feed the summarizer")
):
   
    clauses = []
    
    # Collect clauses from various sources
    if document_id:
        if document_id not in document_storage:
            raise HTTPException(status_code=404, detail="Document not found")
        doc_record = document_storage[document_id]
        clauses = doc_record.clauses
        if not clauses:
            raise HTTPException(status_code=400, detail="No clauses available to summarize for this document. Try /extract with a lower min_score.")
    
    elif file_url:
        # Process file URL directly
        print(f"Processing file URL for summary: {file_url}")
        try:
            # Use extract endpoint internally
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()
            contents = response.content
            
            if not contents.startswith(b'%PDF'):
                raise HTTPException(status_code=400, detail="File is not a valid PDF")
            
            # Generate temporary document ID
            temp_document_id = str(uuid.uuid4())
            file_hash = generate_file_hash(contents)
            
            # Extract text and analyze
            text_pages = extract_text_from_pdf(contents)
            detector = SmartKeywordDetector()
            analysis = detector.analyze_document_with_pages(text_pages)
            clauses = analysis["structured_clauses"]
            
           
            document_storage[temp_document_id] = DocumentRecord(
                id=temp_document_id,
                filename=file_url.split("/")[-1] + ".pdf",
                file_hash=file_hash,
                upload_timestamp=datetime.now(timezone.utc).isoformat(),
                size_kb=round(len(contents) / 1024, 2),
                total_pages=len(text_pages),
                status="processed",
                metadata={"source": "summarize_direct", "source_url": file_url},
                clauses=clauses[:max_items*2],
                summaries={},
                last_updated=datetime.now(timezone.utc).isoformat()
            )
            
            document_id = temp_document_id
            print(f"Created temporary document {document_id} for URL processing")
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process file URL: {str(e)}")
    
    elif file:
       
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        contents = await file.read()
        text_pages = extract_text_from_pdf(contents)
        detector = SmartKeywordDetector()
        analysis = detector.analyze_document_with_pages(text_pages)
        clauses = analysis["structured_clauses"]
        document_id = "temp_" + str(uuid.uuid4())[:8]
        

        document_storage[document_id] = DocumentRecord(
            id=document_id,
            filename=file.filename,
            file_hash=generate_file_hash(contents),
            upload_timestamp=datetime.now(timezone.utc).isoformat(),
            size_kb=round(len(contents) / 1024, 2),
            total_pages=len(text_pages),
            status="processed",
            metadata={"source": "summarize_upload"},
            clauses=clauses[:max_items*2],
            summaries={},
            last_updated=datetime.now(timezone.utc).isoformat()
        )
    
    else:
        raise HTTPException(status_code=400, detail="Either document_id, file_url, or file must be provided")

  
    executive_summary, summary_meta = generate_gemini_summary(
        clauses,
        min_high_score=min_high_score,
        min_items=min_items,
        max_items=max_items
    )

  
    top_for_clauses = sorted(clauses, key=lambda c: c.score, reverse=True)[:10]
    clause_summaries = []
    for c in top_for_clauses:
        clause_summaries.append({
            "clause_id": c.id,
            "category": c.category,
            "importance": c.score,
            "key_points": c.keywords[:8],
            "summary": (c.text[:400] + "...") if len(c.text) > 400 else c.text,
            "page": c.page,
        })


    if document_id in document_storage:
        document_storage[document_id].summaries = {
            "executive_summary": executive_summary,
            "clause_summaries": clause_summaries,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "meta": summary_meta
        }
        document_storage[document_id].last_updated = datetime.now(timezone.utc).isoformat()

    return {
        "document_id": document_id,
        "executive_summary": executive_summary,
        "clause_summaries": clause_summaries,
        "summary_metadata": {
            "total_clauses": len(clauses),
            "summarized_clauses": len(clause_summaries),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **summary_meta
        }
    }

@app.get("/clauses")
async def search_clauses(
    document_id: str = Query(..., description="Document ID"),
    category: Optional[str] = Query(None, description="Filter by category"),
    query: Optional[str] = Query(None, description="Semantic search query"),
    min_score: int = Query(5, description="Minimum importance score"),
    limit: int = Query(50, description="Maximum results")
):
    
    
    if document_id not in document_storage:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_record = document_storage[document_id]
    clauses = doc_record.clauses
    

    if category:
        clauses = [c for c in clauses if c.category.lower() == category.lower()]
    
    
    clauses = [c for c in clauses if c.score >= min_score]
    
   
    if query:
        query_lower = query.lower()
        clauses = [c for c in clauses if query_lower in c.text.lower()]
    
  
    clauses = clauses[:limit]
  
    results = [
        {
            "clause_id": c.id,
            "text": c.text,
            "category": c.category,
            "importance_score": c.score,
            "keywords": c.keywords,
            "page_number": c.page,
            "entities": c.entities
        }
        for c in clauses
    ]
    
    return {
        "document_id": document_id,
        "search_params": {
            "category": category,
            "query": query,
            "min_score": min_score,
            "limit": limit
        },
        "results_count": len(results),
        "clauses": results
    }

@app.post("/risk")
async def analyze_risks(
    document_id: str = Query(..., description="Document ID to analyze"),
    clause_ids: Optional[List[str]] = Query(None, description="Specific clause IDs to analyze")
):
 
    
    if document_id not in document_storage:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_record = document_storage[document_id]
    clauses = doc_record.clauses
    
  
    if clause_ids:
        clauses = [c for c in clauses if c.id in clause_ids]
    
  
    risk_analysis = []
    for clause in clauses:
        risk_level, reasoning = analyze_clause_risk(clause)
        
       
        clause.risk_level = risk_level
        clause.risk_reasoning = reasoning
        
        risk_analysis.append({
            "clause_id": clause.id,
            "category": clause.category,
            "risk_level": risk_level,
            "reasoning": reasoning,
            "importance_score": clause.score,
            "preview": clause.text[:150] + "..." if len(clause.text) > 150 else clause.text
        })
    
   
    document_storage[document_id].last_updated = datetime.now(timezone.utc).isoformat()
    
  
    risk_summary = {
        "red_count": len([r for r in risk_analysis if r["risk_level"] == "Red"]),
        "yellow_count": len([r for r in risk_analysis if r["risk_level"] == "Yellow"]),
        "green_count": len([r for r in risk_analysis if r["risk_level"] == "Green"])
    }
    
    return {
        "document_id": document_id,
        "risk_summary": risk_summary,
        "risk_analysis": risk_analysis,
        "analyzed_at": datetime.now(timezone.utc).isoformat()
    }

@app.get("/documents")
async def list_documents():
    
    
    documents = []
    for doc_id, doc_record in document_storage.items():
        documents.append({
            "document_id": doc_record.id,
            "filename": doc_record.filename,
            "upload_timestamp": doc_record.upload_timestamp,
            "size_kb": doc_record.size_kb,
            "total_pages": doc_record.total_pages,
            "status": doc_record.status,
            "clause_count": len(doc_record.clauses),
            "last_updated": doc_record.last_updated,
            "has_summaries": bool(doc_record.summaries)
        })
    
    return {
        "total_documents": len(documents),
        "documents": sorted(documents, key=lambda x: x["upload_timestamp"], reverse=True)
    }

@app.get("/documents/{document_id}")
async def get_document_details(document_id: str = Path(..., description="Document ID")):
   
    
    if document_id not in document_storage:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_record = document_storage[document_id]
    
 
    clauses_data = []
    for clause in doc_record.clauses:
        clause_dict = asdict(clause)
        clauses_data.append(clause_dict)
    
    return {
        "document_id": doc_record.id,
        "filename": doc_record.filename,
        "metadata": {
            "upload_timestamp": doc_record.upload_timestamp,
            "size_kb": doc_record.size_kb,
            "total_pages": doc_record.total_pages,
            "status": doc_record.status,
            "last_updated": doc_record.last_updated,
            "file_hash": doc_record.file_hash
        },
        "clauses": clauses_data,
        "summaries": doc_record.summaries,
        "analysis_metadata": doc_record.metadata
    }

@app.put("/documents/{document_id}")
async def update_document(
    document_id: str = Path(..., description="Document ID"),
    update_data: DocumentUpdateRequest = None
):
  
    
    if document_id not in document_storage:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_record = document_storage[document_id]
    
    if update_data:
        if update_data.filename:
            doc_record.filename = update_data.filename
        
        if update_data.tags:
            doc_record.metadata["tags"] = update_data.tags
        
        if update_data.notes:
            doc_record.metadata["notes"] = update_data.notes
    
    doc_record.last_updated = datetime.now(timezone.utc).isoformat()
    
    return {
        "document_id": document_id,
        "message": "Document updated successfully",
        "updated_fields": {
            "filename": update_data.filename if update_data and update_data.filename else None,
            "tags": update_data.tags if update_data and update_data.tags else None,
            "notes": update_data.notes if update_data and update_data.notes else None
        },
        "last_updated": doc_record.last_updated
    }

@app.patch("/documents/{document_id}")
async def patch_document(
    document_id: str = Path(..., description="Document ID"),
    patch_data: DocumentPatchRequest = None
):
   
    
    if document_id not in document_storage:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_record = document_storage[document_id]
    
    if patch_data:
        if patch_data.reviewed is not None:
            doc_record.metadata["reviewed"] = patch_data.reviewed
        
        if patch_data.archived is not None:
            doc_record.metadata["archived"] = patch_data.archived
            if patch_data.archived:
                doc_record.status = "archived"
            else:
                doc_record.status = "processed"
        
        if patch_data.priority:
            doc_record.metadata["priority"] = patch_data.priority
    
    doc_record.last_updated = datetime.now(timezone.utc).isoformat()
    
    return {
        "document_id": document_id,
        "message": "Document patched successfully",
        "updated_status": doc_record.status,
        "metadata": doc_record.metadata,
        "last_updated": doc_record.last_updated
    }

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str = Path(..., description="Document ID")):
   
    
    if document_id not in document_storage:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_record = document_storage[document_id]
    filename = doc_record.filename
    
    # Delete from storage
    del document_storage[document_id]
    
    return {
        "message": f"Document '{filename}' deleted successfully",
        "document_id": document_id,
        "deleted_at": datetime.now(timezone.utc).isoformat(),
        "remaining_documents": len(document_storage)
    }

@app.get("/")
async def root():
  
    return {
        "message": "📄 Legal Document Analyzer API is running",
        "version": "5.0.0",
        "features": [
            "PDF text extraction and clause parsing",
            "Smart keyword detection with scoring",
            "AI-powered document summarization",
            "Risk analysis and classification",
            "Document management and search",
            "Comprehensive entity extraction"
        ],
        "endpoints": {
            "health": "/health",
            "extract": "POST /extract",
            "summarize": "POST /summarize", 
            "search_clauses": "GET /clauses",
            "risk_analysis": "POST /risk",
            "list_documents": "GET /documents",
            "document_details": "GET /documents/{id}",
            "update_document": "PUT /documents/{id}",
            "patch_document": "PATCH /documents/{id}",
            "delete_document": "DELETE /documents/{id}"
        },
        "documentation": "/docs",
        "total_documents": len(document_storage)
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)