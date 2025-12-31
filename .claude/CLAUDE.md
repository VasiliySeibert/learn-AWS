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

⚠️ **IMPORTANT:** LM Studio must be manually started before running the application!

LM Studio must be running locally for LLM-powered feedback:
- Download from: https://lmstudio.ai/
- Model: `nvidia/llama-3.1-nemotron-nano-8b-v1`
- Server: `http://localhost:1234` (or your Windows IP on WSL)

**Steps to start:**
1. Open LM Studio application
2. Load the Nemotron model
3. Go to "Local Server" or "Developer" tab
4. Click "Start Server"
5. Verify server is running before launching learning.py

**On WSL2:**
- LM Studio runs as Windows application (not in WSL)
- Must be configured to listen on `0.0.0.0:1234` (not just `127.0.0.1`)
- Application will connect via Windows host IP

### 3. Dependencies (requirements.txt)

Core dependencies:
- `gradio` - Web UI framework
- `openai` - LM Studio API client (OpenAI-compatible)
- `requests` - HTTP requests
- `tqdm` - Progress bars (for create_learningMaterial.py)

## Cross-Platform Networking

The application automatically detects the environment and configures networking appropriately.

### Automatic Environment Detection

The `platform_utils.py` module detects whether the application is running in:
- **WSL2** (Windows Subsystem for Linux)
- **macOS** (native)
- **Linux** (native)

### WSL2 Configuration

When running in WSL2:
- **Gradio UI**: Binds to `0.0.0.0`, making it accessible from Windows browser at `http://localhost:7860`
- **LM Studio**: Connects to Windows host at `http://{windows_host_ip}:1234/v1`
  - Windows host IP is automatically detected from `/etc/resolv.conf` (typically `10.255.255.254`)
- **Detection**: Automatic via `/proc/version` check

**Requirements:**
- Run LM Studio as a native Windows application (not inside WSL)
- Access the Gradio UI from Windows browser
- No manual configuration needed

### macOS/Linux Configuration

When running on macOS or native Linux:
- **Gradio UI**: Binds to `127.0.0.1` (localhost only)
- **LM Studio**: Connects to `http://localhost:1234/v1`
- **Detection**: Automatic
- **Behavior**: Unchanged from previous version

### Manual Configuration Overrides

You can override the automatic detection using environment variables or CLI arguments:

#### Environment Variables

```bash
# Force specific LM Studio URL
export LM_STUDIO_URL="http://192.168.1.100:1234/v1"

# Force specific Gradio binding
export GRADIO_SERVER_NAME="0.0.0.0"

# Enable debug output for networking decisions
export DEBUG_NETWORKING="1"

# Then run the application
python learning.py LearningMaterials/clf-c02.json
```

#### Command Line Arguments

```bash
# Override Gradio server binding
python learning.py LearningMaterials/clf-c02.json --server-name 0.0.0.0

# Combine with other options
python learning.py LearningMaterials/clf-c02.json --port 8080 --server-name 0.0.0.0
```

### Troubleshooting

#### WSL: Cannot access Gradio UI from Windows browser

1. **Check Windows host IP detection:**
   ```bash
   cat /etc/resolv.conf
   # Look for the nameserver IP
   ```

2. **Verify Gradio binding:**
   ```bash
   python learning.py LearningMaterials/clf-c02.json
   # Look for: "Binding Gradio to: 0.0.0.0:7860"
   ```

3. **Test platform detection:**
   ```bash
   python3 platform_utils.py
   # Should show WSL: True and Windows Host IP
   ```

4. **Manual override if needed:**
   ```bash
   export GRADIO_SERVER_NAME="0.0.0.0"
   python learning.py LearningMaterials/clf-c02.json
   ```

#### WSL: LM Studio connection fails

1. **Verify LM Studio is running on Windows:**
   - Open LM Studio on Windows (not in WSL)
   - Start the local server
   - Check it's listening on port 1234

2. **Check Windows host IP:**
   ```bash
   cat /etc/resolv.conf | grep nameserver
   # Note the IP address
   ```

3. **Test platform detection:**
   ```bash
   python3 platform_utils.py
   # Verify LM Studio URL uses Windows host IP
   ```

4. **Manual override if automatic detection fails:**
   ```bash
   # Replace with your Windows host IP
   export LM_STUDIO_URL="http://10.255.255.254:1234/v1"
   python learning.py LearningMaterials/clf-c02.json
   ```

5. **Check Windows Firewall:**
   - Windows Firewall might block WSL connections
   - Allow incoming connections on port 1234 for LM Studio

#### Debug Mode

Enable detailed networking logs:

```bash
export DEBUG_NETWORKING="1"
python learning.py LearningMaterials/clf-c02.json
```

This will print:
- Detected platform (WSL/macOS/Linux)
- Windows host IP (if WSL)
- LM Studio URL being used
- Gradio server_name being used

### Technical Details

**WSL2 Networking:**
- WSL2 uses a virtualized network adapter
- Windows host is accessible via the nameserver IP in `/etc/resolv.conf`
- Services in WSL binding to `127.0.0.1` are not accessible from Windows
- Services binding to `0.0.0.0` are accessible from Windows via `localhost`

**Platform Detection Method:**
- Check `/proc/version` for "microsoft" string
- Parse nameserver IP from `/etc/resolv.conf`
- Graceful fallback to localhost if detection fails
