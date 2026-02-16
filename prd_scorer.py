#!/usr/bin/env python3
"""
PRD Scorer - A web application to evaluate Product Requirements Documents
Uses Claude AI to intelligently score PRDs out of 10 based on key quality parameters.
"""

import os
import re
import json
import tempfile
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

# Document parsing libraries
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

# OpenRouter API (OpenAI-compatible)
try:
    import openai
    OPENROUTER_SUPPORT = True
except ImportError:
    OPENROUTER_SUPPORT = False

# Get the directory where this script is located
basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__, template_folder=os.path.join(basedir, 'templates'))
CORS(app)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'md'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    if not PDF_SUPPORT:
        return None, "PDF support not available. Install PyPDF2."
    
    try:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text.strip(), None
    except Exception as e:
        return None, f"Error reading PDF: {str(e)}"


def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    if not DOCX_SUPPORT:
        return None, "DOCX support not available. Install python-docx."
    
    try:
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text.strip(), None
    except Exception as e:
        return None, f"Error reading DOCX: {str(e)}"


def extract_text_from_txt(file_path):
    """Extract text from a TXT/MD file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip(), None
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read().strip(), None
        except Exception as e:
            return None, f"Error reading file: {str(e)}"
    except Exception as e:
        return None, f"Error reading file: {str(e)}"


def extract_text(file_path, filename):
    """Extract text from uploaded file based on extension."""
    ext = filename.rsplit('.', 1)[1].lower()
    
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ('docx', 'doc'):
        return extract_text_from_docx(file_path)
    else:  # txt, md
        return extract_text_from_txt(file_path)


class OpenRouterPRDScorer:
    """
    Uses OpenRouter's free AI models to intelligently evaluate PRDs on 10 key dimensions:
    1. Problem Statement - Is the problem clearly defined with context?
    2. Target Users - Are user personas and segments well-defined?
    3. Success Metrics - Are KPIs specific, measurable, and realistic?
    4. Requirements Clarity - Are requirements specific and unambiguous?
    5. Scope Definition - Is scope clearly bounded with in/out of scope items?
    6. Technical Considerations - Are technical constraints and architecture addressed?
    7. Timeline/Milestones - Are deliverables and dates realistic?
    8. Risks & Dependencies - Are risks identified with mitigations?
    9. Acceptance Criteria - Is definition of done clear for each feature?
    10. Overall Completeness - Is the document well-structured and comprehensive?
    """
    
    # Free models available on OpenRouter (ordered by capability)
    FREE_MODELS = [
        "google/gemma-3-27b-it:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
        "qwen/qwen3-14b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek/deepseek-r1-0528:free",
        "google/gemma-3-4b-it:free",
    ]
    
    SYSTEM_PROMPT = """You are an expert Product Manager and PRD reviewer. Your task is to analyze Product Requirements Documents and provide structured scoring and feedback.

You must respond with valid JSON only, no additional text, thinking, or markdown formatting. Output the JSON directly."""

    ANALYSIS_PROMPT = """Analyze the following Product Requirements Document (PRD) and evaluate it on 10 key dimensions.

For each dimension, provide:
1. A score from 0-10 (where 10 is excellent)
2. Specific, actionable feedback explaining the score

## Scoring Criteria

**1. Problem Statement (problem_statement)**
- 9-10: Crystal clear problem with compelling context, data, and business impact
- 7-8: Well-defined problem with good context
- 5-6: Problem stated but lacks depth or context
- 3-4: Vague problem definition
- 0-2: Missing or unclear problem statement

**2. Target Users (target_users)**
- 9-10: Detailed personas with behaviors, needs, and pain points
- 7-8: Clear user segments with good descriptions
- 5-6: Users mentioned but not well-characterized
- 3-4: Vague user references
- 0-2: No user definition

**3. Success Metrics (success_metrics)**
- 9-10: SMART metrics with baselines and targets
- 7-8: Clear KPIs with measurable targets
- 5-6: Some metrics but vague or incomplete
- 3-4: Generic success statements
- 0-2: No success criteria

**4. Requirements Clarity (requirements_clarity)**
- 9-10: Detailed, unambiguous requirements with user stories and acceptance criteria
- 7-8: Clear requirements, mostly specific
- 5-6: Requirements present but some ambiguity
- 3-4: Vague or incomplete requirements
- 0-2: Missing requirements

**5. Scope Definition (scope_definition)**
- 9-10: Explicit in-scope/out-of-scope with clear MVP definition
- 7-8: Good scope boundaries defined
- 5-6: Scope mentioned but boundaries unclear
- 3-4: Vague scope
- 0-2: No scope definition

**6. Technical Considerations (technical_considerations)**
- 9-10: Thorough technical requirements, architecture, constraints, and dependencies
- 7-8: Good technical coverage
- 5-6: Basic technical mentions
- 3-4: Minimal technical consideration
- 0-2: No technical details

**7. Timeline & Milestones (timeline)**
- 9-10: Detailed roadmap with phases, milestones, and dependencies
- 7-8: Clear timeline with major milestones
- 5-6: General timeline mentioned
- 3-4: Vague timing references
- 0-2: No timeline

**8. Risks & Dependencies (risks_dependencies)**
- 9-10: Comprehensive risk analysis with mitigations and contingencies
- 7-8: Key risks identified with mitigations
- 5-6: Some risks mentioned
- 3-4: Minimal risk consideration
- 0-2: No risk analysis

**9. Acceptance Criteria (acceptance_criteria)**
- 9-10: Clear, testable acceptance criteria for all features
- 7-8: Good acceptance criteria for most features
- 5-6: Some acceptance criteria present
- 3-4: Vague completion criteria
- 0-2: No acceptance criteria

**10. Overall Completeness (completeness)**
- 9-10: Comprehensive, well-organized, ready for development
- 7-8: Good coverage with minor gaps
- 5-6: Adequate but needs work
- 3-4: Significant gaps
- 0-2: Incomplete document

## Response Format

You MUST respond with ONLY this JSON structure, nothing else:
{
  "scores": {
    "problem_statement": <number 0-10>,
    "target_users": <number 0-10>,
    "success_metrics": <number 0-10>,
    "requirements_clarity": <number 0-10>,
    "scope_definition": <number 0-10>,
    "technical_considerations": <number 0-10>,
    "timeline": <number 0-10>,
    "risks_dependencies": <number 0-10>,
    "acceptance_criteria": <number 0-10>,
    "completeness": <number 0-10>
  },
  "feedback": {
    "problem_statement": "<specific feedback>",
    "target_users": "<specific feedback>",
    "success_metrics": "<specific feedback>",
    "requirements_clarity": "<specific feedback>",
    "scope_definition": "<specific feedback>",
    "technical_considerations": "<specific feedback>",
    "timeline": "<specific feedback>",
    "risks_dependencies": "<specific feedback>",
    "acceptance_criteria": "<specific feedback>",
    "completeness": "<specific feedback>"
  },
  "assessment": "<2-3 sentence overall assessment of the PRD>",
  "top_strengths": ["<strength 1>", "<strength 2>"],
  "critical_improvements": ["<improvement 1>", "<improvement 2>", "<improvement 3>"]
}

## PRD to Analyze

"""

    def __init__(self):
        self.client = None
        self.current_model = None
        api_key = os.getenv('OPENROUTER_API_KEY')
        if OPENROUTER_SUPPORT and api_key:
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
    
    def analyze_with_openrouter(self, text):
        """Use OpenRouter to analyze the PRD with automatic model fallback."""
        if not self.client:
            return None, "OpenRouter API not configured. Set OPENROUTER_API_KEY environment variable."
        
        # Truncate very long PRDs to fit context window
        max_chars = 80000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Document truncated due to length...]"
        
        # Try each free model until one works
        last_error = None
        for model in self.FREE_MODELS:
            try:
                print(f"  Trying model: {model}...")
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": self.ANALYSIS_PROMPT + text
                        }
                    ],
                    temperature=0.3,
                    max_tokens=2500,
                    extra_headers={
                        "HTTP-Referer": "https://prd-scorer.local",
                        "X-Title": "PRD Scorer"
                    }
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Extract JSON from response (handle markdown code blocks, thinking tags, etc.)
                # Remove any <think> or similar tags that some models add
                if "<think>" in response_text:
                    response_text = response_text.split("</think>")[-1].strip()
                
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    parts = response_text.split("```")
                    for part in parts:
                        part = part.strip()
                        if part.startswith("{"):
                            response_text = part
                            break
                
                # Find JSON object in response
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                if start_idx != -1 and end_idx > start_idx:
                    response_text = response_text[start_idx:end_idx]
                
                result = json.loads(response_text)
                self.current_model = model  # Remember which model worked
                print(f"  ✓ Success with {model}")
                return result, None
                
            except openai.APIError as e:
                last_error = str(e)
                print(f"  ✗ {model} failed: {last_error[:50]}...")
                continue  # Try next model
            except json.JSONDecodeError as e:
                last_error = f"Failed to parse response: {str(e)}"
                print(f"  ✗ {model} returned invalid JSON")
                continue  # Try next model
            except Exception as e:
                last_error = str(e)
                continue  # Try next model
        
        return None, f"All free models are currently rate-limited. Please try again in a few minutes. Last error: {last_error}"
    
    def analyze(self, text):
        """Perform comprehensive PRD analysis using OpenRouter."""
        if not text or len(text.strip()) < 50:
            return {
                'error': 'PRD text is too short or empty. Please provide a complete document.',
                'total_score': 0
            }
        
        word_count = len(text.split())
        
        # Try OpenRouter analysis
        result, error = self.analyze_with_openrouter(text)
        
        if error:
            return {
                'error': error,
                'total_score': 0
            }
        
        # Calculate total score
        scores = result['scores']
        total_score = round(sum(scores.values()) / len(scores), 1)
        
        # Determine grade
        if total_score >= 8.5:
            grade = 'A'
        elif total_score >= 7.0:
            grade = 'B'
        elif total_score >= 5.5:
            grade = 'C'
        elif total_score >= 4.0:
            grade = 'D'
        else:
            grade = 'F'
        
        # Get top improvements (lowest scoring areas)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        top_improvements = [
            {
                'area': s[0].replace('_', ' ').title(),
                'score': s[1],
                'feedback': result['feedback'][s[0]]
            }
            for s in sorted_scores[:3]
        ]
        
        # Get model name for display
        model_name = self.current_model or self.FREE_MODELS[0]
        model_display = model_name.split("/")[-1].replace(":free", "").replace("-instruct", "").replace("-it", "")
        
        return {
            'total_score': total_score,
            'grade': grade,
            'assessment': result['assessment'],
            'word_count': word_count,
            'scores': scores,
            'feedback': result['feedback'],
            'top_improvements': top_improvements,
            'top_strengths': result.get('top_strengths', []),
            'critical_improvements': result.get('critical_improvements', []),
            'text_preview': text[:500] + '...' if len(text) > 500 else text,
            'analysis_method': f'{model_display} (Free via OpenRouter)'
        }


scorer = OpenRouterPRDScorer()


@app.route('/')
def index():
    """Render main page."""
    return render_template('prd_scorer.html')


@app.route('/score', methods=['POST'])
def score_prd():
    """Score a PRD from uploaded file or text."""
    
    # Check if file was uploaded
    if 'file' in request.files:
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save file temporarily and extract text
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{filename.rsplit(".", 1)[1]}') as tmp:
            file.save(tmp.name)
            text, error = extract_text(tmp.name, filename)
            os.unlink(tmp.name)  # Clean up temp file
        
        if error:
            return jsonify({'error': error}), 400
        
        if not text:
            return jsonify({'error': 'Could not extract text from file'}), 400
    
    # Check if text was provided directly
    elif request.is_json:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
    
    else:
        return jsonify({'error': 'No file or text provided'}), 400
    
    # Score the PRD
    try:
        result = scorer.analyze(text)
        if 'error' in result and result.get('total_score', 0) == 0:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    api_configured = bool(os.getenv('OPENROUTER_API_KEY'))
    return jsonify({
        'status': 'healthy',
        'pdf_support': PDF_SUPPORT,
        'docx_support': DOCX_SUPPORT,
        'openrouter_support': OPENROUTER_SUPPORT,
        'api_configured': api_configured
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8081))
    debug = os.getenv('FLASK_ENV') != 'production'
    api_configured = bool(os.getenv('OPENROUTER_API_KEY'))
    
    print("\n" + "=" * 50)
    print("  📋 PRD Scorer (Free AI via OpenRouter)")
    print(f"  Open http://localhost:{port} in your browser")
    print("=" * 50)
    print(f"  PDF Support: {'✓' if PDF_SUPPORT else '✗ (install PyPDF2)'}")
    print(f"  DOCX Support: {'✓' if DOCX_SUPPORT else '✗ (install python-docx)'}")
    print(f"  OpenRouter API: {'✓ Configured' if api_configured else '✗ Set OPENROUTER_API_KEY'}")
    print(f"  Model: Llama 3.3 70B (Free)")
    print("=" * 50 + "\n")
    
    if not api_configured:
        print("  ⚠️  To enable AI analysis:")
        print("     1. Sign up at https://openrouter.ai (free)")
        print("     2. Get your API key from https://openrouter.ai/keys")
        print("     3. export OPENROUTER_API_KEY=your-api-key")
        print("     or add it to your .env file\n")
    
    app.run(debug=debug, host='0.0.0.0', port=port)
