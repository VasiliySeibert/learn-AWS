# learn-AWS Project Documentation

## Overview
AWS certification study tool with badge-based learning and progress tracking.

## Key Scripts

### learning.py
Gradio-based study application with Learning and Exam modes.

**How to run:**
```bash
source .venv/bin/activate
python learning.py LearningMaterials/clf-c02.json
```

**Features:**
- Badge-based organization (questions grouped by Domain, Technology, AWS Service category)
- Learning mode: immediate feedback, can peek at reference material
- Exam mode: no peeking, scores revealed at end
- **Human-in-the-loop grading**: User assigns scores based on LLM analysis
- Progress tracking per badge (review count, last reviewed, exam scores)

### create_learningMaterial.py
Generates learning materials from exam guide task statements using LLM.

**How to run:**
```bash
source .venv/bin/activate
python create_learningMaterial.py Input/<input_folder>
```

### lm_studio_utils.py
Utility for calling local LM Studio API (Nemotron model on localhost:1234).

## Folder Structure

```
learn-AWS/
├── Input/                      # Exam guide content (task statements, technologies, services)
├── LearningMaterials/          # Generated learning materials JSON files
│   └── clf-c02.json            # AWS Cloud Practitioner learning materials
├── ExamMode/                   # Exam results and session files
│   └── clf-c02_session.json    # Badge progress tracking (auto-generated)
├── .venv/                      # Python virtual environment
└── .claude/                    # Claude Code configuration
```

## Session File System

### Location & Naming
Session files are stored in `ExamMode/` with naming derived from the input file:
- Input: `LearningMaterials/clf-c02.json`
- Session: `ExamMode/clf-c02_session.json`

### When Created
- **First run**: `load_or_create_session()` creates the session file if it doesn't exist
- **Subsequent runs**: Loads existing file to preserve progress

### What's Tracked (per badge)
```json
{
  "review_count": 3,           // How many times badge was completed
  "last_reviewed": "2025-...", // ISO timestamp of last completion
  "exam_scores": [             // History of exam attempts
    {"date": "...", "score": 28, "max_score": 40, "percentage": 70.0}
  ],
  "best_score_percentage": 70.0
}
```

### When Updated
- After completing a badge (Learning or Exam mode)
- `update_badge_metadata()` increments review_count and appends scores
- `save_session()` persists changes to disk

## Badge System

Questions are organized into badges by topic:
- **Domain badges**: 1 per domain (Domain 1, Domain 2, etc.)
- **Technology badges**: Split into 2 parts (~15-16 questions each)
- **AWS Service badges**: 1 per service category (Analytics, Compute, Security, etc.)

Category headers and PDF artifacts are filtered out during question loading.

## Human-in-the-Loop Grading System

The grading system uses a two-phase approach for reliable scoring:

### Phase 1: Answer Collection
1. User answers questions (peek available in Learning mode)
2. LLM provides structured analysis for each answer (not scores)
3. Answers are stored with analysis for later grading

### Phase 2: Grading Review
After all questions are answered, user enters grading phase:
1. Each answer is displayed with LLM analysis
2. User assigns score (0-10) using slider
3. Reference material available for verification
4. Navigate back/forward between answers

### LLM Analysis Structure
The LLM provides structured feedback (no scoring):
```json
{
  "mentioned": ["key points the user covered correctly"],
  "missing": ["important topics not addressed"],
  "incorrect": ["factually wrong statements"]
}
```

### Why Human-in-the-Loop?
- **Reliability**: LLM scoring was inconsistent and regex parsing fragile
- **Trust**: User has final say on point assignment
- **Learning**: Reviewing answers reinforces understanding

### Answer Data Structure
```json
{
  "question": "...",
  "category": "Domain 1",
  "user_answer": "...",
  "learning_material": "...",
  "analysis": {
    "mentioned": [...],
    "missing": [...],
    "incorrect": [...]
  },
  "score": 7,
  "feedback": "...",
  "graded_by": "human"
}
```

### Key Functions
- `analyze_answer()`: Gets structured LLM analysis (no scoring)
- `format_analysis_display()`: Formats analysis as markdown
- `enter_grading_phase()`: Transitions from answering to grading
- `handle_grade_submit()`: Saves user score, advances to next
- `finish_grading()`: Updates badge metadata, shows summary

## Prerequisites

### 1. Python Virtual Environment

The project uses a `.venv` virtual environment. **Always activate it before running scripts.**

**If .venv exists:**
```bash
source .venv/bin/activate
```

**If .venv doesn't exist (first-time setup):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. LM Studio

LM Studio must be running locally for LLM-powered feedback:
- Download from: https://lmstudio.ai/
- Model: `nvidia/llama-3.1-nemotron-nano-8b-v1`
- Server: `http://localhost:1234`
- Start the local server before running learning.py

### 3. Dependencies (requirements.txt)

Core dependencies:
- `gradio` - Web UI framework
- `openai` - LM Studio API client (OpenAI-compatible)
- `requests` - HTTP requests
- `tqdm` - Progress bars (for create_learningMaterial.py)
