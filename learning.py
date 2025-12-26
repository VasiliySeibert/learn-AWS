"""
Docstring für learning. Do not remove this specification.

HOW TO RUN:
    1. Start LM Studio with the Nemotron model (nvidia/llama-3.1-nemotron-nano-8b-v1) on localhost:1234
    2. Activate venv and run: source .venv/bin/activate && python learning.py LearningMaterials/clf-c02.json
    3. Open http://localhost:7860 in your browser

This script expects as an input a path to a .json from the LearningMaterials folder. The json will look like this:


{
  "Domains": {
    "Domain 1": [
      {
        "task-statement": "Task Statement 1.1: Define the benefits of ...",
        "learning-material": "# **Learning Material – 1.1 Define the Benefits of the AWS Cloud**  ..."
      },
      ...
    ],
    "Domain 2": [
      ...
    ],
    "Domain 3": [
      ...
    ],
    "Domain 4": [
      ...
  },
  "Technologies": [
    {
      "technology": "APIs",
      "learning-material": "## 📚 Quick‑Start Guide to APIs ..."
    },
    ...
  ],
  "AWS-Services": [
    {
      "aws-service": "Analytics",
      "learning-material": "## AWS Analytics – Quick‑Reference Learning Material  ..."
    },
    ...
  ]
}

The purpose of this script is to help me study. It will iterate through the task statements and phrase them as a question. Then it will wait for my input (I will try to answer the question). After confirmation, it will give me feedback.
This is not a command line tool. I expect some kind of GUI (e.g. a separate text editor or in the browser).
Since this is for learning, there should be a peek button, which shows me the learning material related to the question.

In default mode the script should go through all domains and their tasks in the order that is specified in the JSON file.
Im random mode the script should randomly select tasks from all domains and technologies and services.

I can specify how many questions I want to answer in one session.

for giving feedback the script should use lm studio (check out lm_studio_utils.py).

After answering all the questions the script should give me a summary of my performance (e.g. how many questions I answered correctly, which topics I struggled with etc.).
For determining correctness use lm studio as well, formulate a prompt so that the LM acts like a teacher grading my answers.

There should also be an exam mode, where I can specify a number of questions and then have to answer them without any peeking at the learning material. At the end of the exam I get a grade and feedback on my answers and overall performance.

the data about my performance should be stored in the ExamMode Folder in a .json file, so that I can track my progress over time.
"""

import json
import os
import argparse
import random
from datetime import datetime
from dataclasses import dataclass, field
import gradio as gr
from lm_studio_utils import prompt_nemotron


# =============================================================================
# Data Loading
# =============================================================================

def load_learning_materials(path: str) -> dict:
    """Load learning materials JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_question_pool(data: dict) -> list[dict]:
    """
    Build a flat list of questions from the learning materials.
    Each question is a dict with: question_id, type, category, subcategory, question, learning_material
    """
    questions = []
    question_id = 0

    # Process Domains
    for domain_name, tasks in data.get("Domains", {}).items():
        for task in tasks:
            if "learning-material" in task:
                questions.append({
                    "question_id": question_id,
                    "type": "domain",
                    "category": domain_name,
                    "subcategory": None,
                    "question": task["task-statement"],
                    "learning_material": task["learning-material"]
                })
                question_id += 1

    # Process Technologies
    for tech in data.get("Technologies", []):
        if isinstance(tech, dict) and "learning-material" in tech:
            questions.append({
                "question_id": question_id,
                "type": "technology",
                "category": "Technologies",
                "subcategory": None,
                "question": f"Explain the following AWS technology: {tech['technology']}",
                "learning_material": tech["learning-material"]
            })
            question_id += 1

    # Process AWS Services - track current category for grouping
    # Known category headers in AWS services list (from AWS documentation)
    known_categories = {
        "Analytics", "Application Integration", "Business Applications",
        "Cloud Financial Management", "Compute", "Containers",
        "Customer Engagement", "Database", "Developer Tools",
        "End User Computing", "Frontend Web and Mobile",
        "Internet of Things (IoT)", "Machine Learning",
        "Management and Governance", "Migration and Transfer",
        "Networking and Content Delivery", "Security, Identity, and Compliance",
        "Serverless", "Storage"
    }
    # Artifacts from PDF parsing that should be skipped entirely
    skip_entries = {
        "In-Scope AWS Services", "AWS Certification Exam Guides"
    }
    current_service_category = "General"
    for service in data.get("AWS-Services", []):
        if isinstance(service, dict):
            service_name = service.get("aws-service", "")
            # Skip PDF artifacts
            if any(service_name.startswith(skip) for skip in skip_entries):
                continue
            # Detect category headers by known names only
            if service_name in known_categories:
                current_service_category = service_name
                continue
            questions.append({
                "question_id": question_id,
                "type": "service",
                "category": "AWS Services",
                "subcategory": current_service_category,
                "question": f"Explain the following AWS service: {service_name}",
                "learning_material": service["learning-material"]
            })
            question_id += 1

    return questions


# =============================================================================
# Badge System
# =============================================================================
#
# The badge system organizes questions into manageable study units:
# - Questions are grouped by topic (Domains, Technologies, AWS Service categories)
# - Each badge tracks: review_count, last_reviewed, exam_scores, best_score_percentage
# - Progress is persisted in a session file (ExamMode/<input_name>_session.json)
#
# Session File Lifecycle:
# - Created on first run via load_or_create_session() if it doesn't exist
# - Named based on input file: clf-c02.json -> clf-c02_session.json
# - Updated after completing each badge (review_count++, scores appended)
# - Reloaded when returning to badge selector to reflect latest progress
#
# =============================================================================

def generate_badge_definitions(questions: list[dict]) -> list[dict]:
    """
    Generate badge definitions by grouping questions by topic.

    Grouping strategy:
    - 1 badge per Domain (Domain 1, Domain 2, etc.)
    - Technologies split into 2 badges (~15-16 questions each)
    - AWS Services grouped by subcategory (Analytics, Compute, etc.)

    Each badge contains metadata initialized to zero/empty for tracking progress.
    """
    badges = []

    # Group questions by type and category
    domains = {}
    technologies = []
    services_by_category = {}

    for q in questions:
        if q["type"] == "domain":
            cat = q["category"]
            if cat not in domains:
                domains[cat] = []
            domains[cat].append(q["question_id"])
        elif q["type"] == "technology":
            technologies.append(q["question_id"])
        elif q["type"] == "service":
            subcat = q["subcategory"] or "General"
            if subcat not in services_by_category:
                services_by_category[subcat] = []
            services_by_category[subcat].append(q["question_id"])

    # Create Domain badges
    for domain_name, question_ids in domains.items():
        badge_id = domain_name.lower().replace(" ", "_")
        badges.append({
            "id": badge_id,
            "name": domain_name,
            "type": "domain",
            "question_ids": question_ids,
            "review_count": 0,
            "last_reviewed": None,
            "exam_scores": [],
            "best_score_percentage": None
        })

    # Create Technology badges (split if > 16)
    if technologies:
        mid = len(technologies) // 2
        badges.append({
            "id": "technologies_1",
            "name": "Technologies (Part 1)",
            "type": "technology",
            "question_ids": technologies[:mid],
            "review_count": 0,
            "last_reviewed": None,
            "exam_scores": [],
            "best_score_percentage": None
        })
        badges.append({
            "id": "technologies_2",
            "name": "Technologies (Part 2)",
            "type": "technology",
            "question_ids": technologies[mid:],
            "review_count": 0,
            "last_reviewed": None,
            "exam_scores": [],
            "best_score_percentage": None
        })

    # Create AWS Services badges by category
    for category, question_ids in services_by_category.items():
        badge_id = f"services_{category.lower().replace(' ', '_').replace('-', '_')}"
        badges.append({
            "id": badge_id,
            "name": f"AWS Services: {category}",
            "type": "service",
            "question_ids": question_ids,
            "review_count": 0,
            "last_reviewed": None,
            "exam_scores": [],
            "best_score_percentage": None
        })

    return badges


def get_session_file_path(data_path: str) -> str:
    """
    Derive session file path from data file path.
    Example: LearningMaterials/clf-c02.json -> ExamMode/clf-c02_session.json
    """
    base_name = os.path.basename(data_path).replace(".json", "")
    return os.path.join("ExamMode", f"{base_name}_session.json")


def load_or_create_session(data_path: str, questions: list[dict]) -> tuple[dict, str]:
    """
    Load existing session file or create new one with badge definitions.

    This is the main entry point for session management:
    - If session file exists: load and return it (preserves progress)
    - If not: create new session with fresh badge definitions

    Args:
        data_path: Path to the learning materials JSON (e.g., "LearningMaterials/clf-c02.json")
        questions: List of question dicts from build_question_pool()

    Returns:
        Tuple of (session_data dict, session_path string)

    Session file is stored in ExamMode/ folder with naming:
        LearningMaterials/clf-c02.json -> ExamMode/clf-c02_session.json
    """
    session_path = get_session_file_path(data_path)

    # Load existing session if available (preserves all progress data)
    if os.path.exists(session_path):
        with open(session_path, "r", encoding="utf-8") as f:
            session_data = json.load(f)
        return session_data, session_path

    # Create new session file with fresh badge definitions
    os.makedirs(os.path.dirname(session_path), exist_ok=True)
    session_data = {
        "source_file": data_path,
        "created_at": datetime.now().isoformat(),
        "badges": generate_badge_definitions(questions)
    }

    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

    return session_data, session_path


def save_session(session_data: dict, session_path: str) -> None:
    """Save session data to file."""
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)


def update_badge_metadata(session_data: dict, badge_id: str,
                          mode: str, score: int = None, max_score: int = None) -> dict:
    """
    Update badge metadata after completion.
    - Increment review_count
    - Update last_reviewed timestamp
    - Append exam score (if exam mode)
    """
    for badge in session_data["badges"]:
        if badge["id"] == badge_id:
            badge["review_count"] += 1
            badge["last_reviewed"] = datetime.now().isoformat()

            if mode == "exam" and score is not None and max_score is not None:
                percentage = round((score / max_score) * 100, 1) if max_score > 0 else 0
                badge["exam_scores"].append({
                    "date": datetime.now().isoformat(),
                    "score": score,
                    "max_score": max_score,
                    "percentage": percentage
                })
                # Update best score
                if badge["best_score_percentage"] is None or percentage > badge["best_score_percentage"]:
                    badge["best_score_percentage"] = percentage
            break

    return session_data


def format_badge_display(badge: dict) -> str:
    """
    Format badge info for display in selector.
    Example: "Domain 1 | 4 questions | Reviewed 3x | Best: 87%"
    """
    parts = [badge["name"], f"{len(badge['question_ids'])} questions"]

    if badge["review_count"] > 0:
        parts.append(f"Reviewed {badge['review_count']}x")

    if badge["best_score_percentage"] is not None:
        parts.append(f"Best: {badge['best_score_percentage']}%")

    if badge["last_reviewed"]:
        try:
            last = datetime.fromisoformat(badge["last_reviewed"])
            days_ago = (datetime.now() - last).days
            if days_ago == 0:
                parts.append("Today")
            elif days_ago == 1:
                parts.append("Yesterday")
            else:
                parts.append(f"{days_ago} days ago")
        except:
            pass

    return " | ".join(parts)


def get_badge_questions(badge: dict, all_questions: list[dict]) -> list[dict]:
    """Extract questions for a specific badge using question_ids."""
    question_map = {q["question_id"]: q for q in all_questions}
    return [question_map[qid] for qid in badge["question_ids"] if qid in question_map]


# =============================================================================
# LLM Feedback Functions
# =============================================================================

FEEDBACK_PROMPT = """You are a helpful AWS instructor providing feedback to a student.

QUESTION: {question}

REFERENCE MATERIAL (the correct/complete answer):
{learning_material}

STUDENT'S ANSWER:
{user_answer}

Please provide constructive feedback:
1. Acknowledge what the student got right
2. Point out any gaps or inaccuracies
3. Suggest areas for improvement
4. Be encouraging but honest

Keep your feedback concise (2-3 paragraphs max)."""

GRADING_PROMPT = """You are grading an AWS certification exam answer.

QUESTION: {question}

REFERENCE MATERIAL (the correct/complete answer):
{learning_material}

STUDENT'S ANSWER:
{user_answer}

Score this answer from 0-10 based on:
- Accuracy (correct information)
- Completeness (covers key points)
- Understanding (shows comprehension)

You MUST respond with ONLY a valid JSON object in this exact format, no other text:
{{"score": <number 0-10>, "feedback": "<brief feedback>"}}"""


# Human-in-the-loop grading: LLM provides structured analysis, user assigns score
ANALYSIS_PROMPT = """You are an AWS instructor analyzing a student's answer.

QUESTION: {question}

REFERENCE MATERIAL (the correct/complete answer):
{learning_material}

STUDENT'S ANSWER:
{user_answer}

Analyze the student's answer and provide a structured breakdown. Do NOT assign a score.

You MUST respond with ONLY a valid JSON object in this exact format:
{{
  "mentioned": ["key point 1 covered by student", "key point 2 covered", ...],
  "missing": ["important point not mentioned", "another gap", ...],
  "incorrect": ["factually wrong statement 1", "misconception 2", ...]
}}

Guidelines:
- "mentioned": List specific concepts/facts the student correctly addressed
- "missing": List important topics from the reference material not covered
- "incorrect": List any factually wrong statements (empty array if none)
- Be specific and concise in each point
- Use empty arrays [] if nothing applies to that category"""


def get_feedback(question: str, user_answer: str, learning_material: str) -> str:
    """Get LLM feedback for learning mode."""
    if not user_answer.strip():
        return "Please provide an answer before submitting."

    prompt = FEEDBACK_PROMPT.format(
        question=question,
        learning_material=learning_material[:3000],  # Truncate to avoid token limits
        user_answer=user_answer
    )

    try:
        response = prompt_nemotron(prompt, include_reasoning=False)
        return response
    except Exception as e:
        return f"Error getting feedback: {e}"


def grade_answer(question: str, user_answer: str, learning_material: str) -> dict:
    """Grade an answer (0-10) for exam mode."""
    if not user_answer.strip():
        return {"score": 0, "feedback": "No answer provided."}

    prompt = GRADING_PROMPT.format(
        question=question,
        learning_material=learning_material[:3000],
        user_answer=user_answer
    )

    try:
        response = prompt_nemotron(prompt, include_reasoning=False)
        # Try to parse JSON from response
        import re
        json_match = re.search(r'\{[^{}]*"score"[^{}]*"feedback"[^{}]*\}', response)
        if json_match:
            result = json.loads(json_match.group())
            return {"score": int(result.get("score", 0)), "feedback": result.get("feedback", "")}
        else:
            # Fallback: try to parse entire response
            result = json.loads(response)
            return {"score": int(result.get("score", 0)), "feedback": result.get("feedback", "")}
    except Exception as e:
        return {"score": 5, "feedback": f"Could not parse grade, defaulting to 5/10. Error: {e}"}


def analyze_answer(question: str, user_answer: str, learning_material: str) -> dict:
    """Get structured LLM analysis (no scoring) for human grading."""
    if not user_answer.strip():
        return {
            "mentioned": [],
            "missing": ["No answer provided"],
            "incorrect": []
        }

    prompt = ANALYSIS_PROMPT.format(
        question=question,
        learning_material=learning_material[:3000],
        user_answer=user_answer
    )

    try:
        response = prompt_nemotron(prompt, include_reasoning=False)
        # Try to parse JSON from response
        import re
        # Match JSON object containing the expected fields
        json_match = re.search(r'\{[^{}]*"mentioned"[^{}]*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "mentioned": result.get("mentioned", []),
                "missing": result.get("missing", []),
                "incorrect": result.get("incorrect", [])
            }
        else:
            # Fallback: try to parse entire response
            result = json.loads(response)
            return {
                "mentioned": result.get("mentioned", []),
                "missing": result.get("missing", []),
                "incorrect": result.get("incorrect", [])
            }
    except Exception as e:
        return {
            "mentioned": [],
            "missing": [f"Analysis error: {e}"],
            "incorrect": []
        }


def format_analysis_display(analysis: dict) -> str:
    """Format the LLM analysis for display in grading UI."""
    output = "## LLM Analysis\n\n"

    if analysis.get("mentioned"):
        output += "### ✓ Mentioned (covered by your answer)\n"
        for point in analysis["mentioned"]:
            output += f"- {point}\n"
        output += "\n"

    if analysis.get("missing"):
        output += "### ○ Missing (not covered)\n"
        for point in analysis["missing"]:
            output += f"- {point}\n"
        output += "\n"

    if analysis.get("incorrect"):
        output += "### ✗ Incorrect (factual errors)\n"
        for point in analysis["incorrect"]:
            output += f"- {point}\n"
        output += "\n"

    if not any([analysis.get("mentioned"), analysis.get("missing"), analysis.get("incorrect")]):
        output += "*No analysis available*\n"

    return output


# =============================================================================
# Session State
# =============================================================================

@dataclass
class StudySession:
    """
    Manages the state of a study/exam session.

    Session Flow:
    1. Answering phase (phase="answering"): User answers questions
    2. Grading phase (phase="grading"): User reviews answers and assigns scores
    3. Summary: Results displayed and saved

    Answer Structure (stored in self.answers):
    - question, category, user_answer: The question and response
    - learning_material: Reference material for grading
    - analysis: LLM structured analysis {mentioned, missing, incorrect}
    - score: User-assigned score (0-10), None until graded
    - graded_by: "human" for user-graded, "auto" for skipped questions
    """
    questions: list = field(default_factory=list)
    mode: str = "learning"  # "learning" or "exam"
    current_index: int = 0
    answers: list = field(default_factory=list)  # list of result dicts
    num_questions: int = 10
    # Badge tracking
    current_badge: dict | None = None
    session_data: dict | None = None
    session_path: str | None = None
    # Grading phase: after answering all questions, user reviews and grades each
    phase: str = "answering"  # "answering" or "grading"
    grading_index: int = 0    # Current answer being graded (0-indexed)

    def current_question(self) -> dict | None:
        if 0 <= self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    def is_complete(self) -> bool:
        return self.current_index >= self.num_questions or self.current_index >= len(self.questions)

    def badge_progress_text(self) -> str:
        """Return 'Badge: Domain 1 | Question 3/4' format."""
        if self.current_badge:
            return f"**Badge: {self.current_badge['name']}** | Question {self.current_index + 1}/{self.num_questions}"
        return f"**Question {self.current_index + 1}/{self.num_questions}**"

    def grading_progress_text(self) -> str:
        """Return 'Grading 3/10' format for grading phase."""
        return f"**Grading {self.grading_index + 1}/{len(self.answers)}**"

    def current_grading_answer(self) -> dict | None:
        """Get the current answer being graded."""
        if 0 <= self.grading_index < len(self.answers):
            return self.answers[self.grading_index]
        return None

    def total_score(self) -> int:
        return sum(a.get("score") or 0 for a in self.answers)

    def max_score(self) -> int:
        return len(self.answers) * 10

    def category_performance(self) -> dict:
        """Calculate average score per category."""
        categories = {}
        for answer in self.answers:
            cat = answer.get("category", "Unknown")
            if cat not in categories:
                categories[cat] = {"total": 0, "count": 0}
            categories[cat]["total"] += answer.get("score") or 0
            categories[cat]["count"] += 1

        return {
            cat: {"avg_score": round(data["total"] / data["count"], 1), "count": data["count"]}
            for cat, data in categories.items()
        }


# =============================================================================
# Results Persistence
# =============================================================================

def save_results(session: StudySession, output_dir: str = "ExamMode") -> str:
    """Save session results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{session.mode}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    result_data = {
        "timestamp": datetime.now().isoformat(),
        "mode": session.mode,
        "grading_type": "human-in-the-loop",
        "total_score": session.total_score(),
        "max_score": session.max_score(),
        "questions_answered": len(session.answers),
        "results": session.answers,
        "category_performance": session.category_performance()
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    return filepath


# =============================================================================
# Gradio UI
# =============================================================================

def create_app(data_path: str):
    """Create and return the Gradio app."""

    # Load data
    data = load_learning_materials(data_path)
    all_questions = build_question_pool(data)

    if not all_questions:
        raise ValueError("No questions found in the learning materials. Make sure the JSON has learning-material fields.")

    # Load or create session file with badges
    session_data, session_path = load_or_create_session(data_path, all_questions)

    def get_badge_choices():
        """Get formatted badge choices for dropdown."""
        return [(format_badge_display(b), b["id"]) for b in session_data["badges"]]

    def show_badge_info(badge_id: str):
        """Show detailed info for selected badge."""
        if not badge_id:
            return ""
        for badge in session_data["badges"]:
            if badge["id"] == badge_id:
                info = f"### {badge['name']}\n\n"
                info += f"- **Questions:** {len(badge['question_ids'])}\n"
                info += f"- **Reviews:** {badge['review_count']}\n"
                if badge["best_score_percentage"] is not None:
                    info += f"- **Best Score:** {badge['best_score_percentage']}%\n"
                if badge["exam_scores"]:
                    info += f"- **Exam Attempts:** {len(badge['exam_scores'])}\n"
                return info
        return ""

    def start_badge_session(mode: str, randomize: bool, badge_id: str):
        """Initialize a new study session for a specific badge."""
        if not badge_id:
            return (
                None, "", "", "", "", "",
                gr.update(visible=True),   # badge_selector_section
                gr.update(visible=False),  # question_section
                gr.update(visible=False),  # grading_section
                gr.update(visible=False),  # summary_section
                gr.update(), gr.update(), gr.update()
            )

        # Find the badge
        badge = None
        badge_index = 0
        for i, b in enumerate(session_data["badges"]):
            if b["id"] == badge_id:
                badge = b
                badge_index = i
                break

        if not badge:
            return (
                None, "", "", "", "", "",
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(), gr.update(), gr.update()
            )

        # Get questions for this badge
        questions = get_badge_questions(badge, all_questions)
        if randomize:
            random.shuffle(questions)

        session = StudySession(
            questions=questions,
            mode=mode.lower(),
            num_questions=len(questions),
            current_badge=badge,
            session_data=session_data,
            session_path=session_path
        )

        q = session.current_question()
        progress = session.badge_progress_text()
        question_text = f"### {q['question']}"
        peek_visible = mode.lower() == "learning"

        return (
            session,  # session_state
            progress,  # progress_text
            question_text,  # question_display
            "",  # answer_input
            "",  # feedback_display
            "",  # peek_display
            gr.update(visible=False),  # badge_selector_section
            gr.update(visible=True),   # question_section
            gr.update(visible=False),  # grading_section
            gr.update(visible=False),  # summary_section
            gr.update(visible=peek_visible),  # peek_btn
            gr.update(visible=True),  # submit_btn
            gr.update(visible=False),  # next_btn
        )

    def submit_answer(session: StudySession, answer: str):
        """Handle answer submission."""
        if session is None:
            return session, "Please start a session first.", "", gr.update(), gr.update()

        q = session.current_question()
        if q is None:
            return session, "No more questions.", "", gr.update(), gr.update()

        # Get structured analysis for human grading (both modes)
        analysis = analyze_answer(q["question"], answer, q["learning_material"])

        # Get feedback text based on mode
        if session.mode == "learning":
            feedback = get_feedback(q["question"], answer, q["learning_material"])
            feedback_text = f"**Answer recorded.**\n\n{feedback}"
        else:  # exam mode
            feedback = ""
            feedback_text = "**Answer recorded.** (You'll grade all answers at the end)"

        # Record the answer with analysis (score=None, will be set during grading)
        session.answers.append({
            "question": q["question"],
            "category": q["category"],
            "user_answer": answer,
            "learning_material": q["learning_material"],  # Store for grading reference
            "analysis": analysis,  # LLM structured analysis
            "score": None,  # Will be set by user during grading phase
            "feedback": feedback,
            "graded_by": None  # Will be set to "human" during grading
        })

        return (
            session,
            feedback_text,
            gr.update(visible=False),  # hide submit
            gr.update(visible=True),   # show next
        )

    def next_question(session: StudySession):
        """Move to the next question or enter grading phase."""
        if session is None:
            return (session, "", "", "", "", gr.update(), gr.update(), gr.update(), "",
                    gr.update(), "", "", "", "", "", 0)

        session.current_index += 1

        if session.is_complete():
            # Enter grading phase instead of showing summary directly
            return enter_grading_phase(session)

        q = session.current_question()
        progress = session.badge_progress_text()
        question_text = f"### {q['question']}"

        return (
            session,
            progress,
            question_text,
            "",  # clear answer
            "",  # clear feedback
            gr.update(visible=True),   # question_section
            gr.update(visible=False),  # grading_section
            gr.update(visible=False),  # summary_section
            gr.update(visible=False),  # next_btn (will show after submit)
            "",  # summary_display
            # Grading section outputs (empty when not in grading phase)
            "", "", "", "", "", 0
        )

    def peek_material(session: StudySession):
        """Show the learning material for current question."""
        if session is None or session.current_question() is None:
            return "No question loaded."
        return session.current_question()["learning_material"]

    def skip_question(session: StudySession):
        """Skip current question without answering."""
        if session is None:
            return (session, "", "", "", "", gr.update(), gr.update(), gr.update(), "",
                    gr.update(), "", "", "", "", "", 0)

        q = session.current_question()
        if q:
            session.answers.append({
                "question": q["question"],
                "category": q["category"],
                "user_answer": "(skipped)",
                "learning_material": q["learning_material"],
                "analysis": {"mentioned": [], "missing": ["Question was skipped"], "incorrect": []},
                "score": 0,  # Skipped = 0 points
                "feedback": "Question was skipped.",
                "graded_by": "auto"  # Auto-graded as 0
            })

        return next_question(session)

    # =========================================================================
    # Grading Phase Functions
    # =========================================================================

    def load_grading_item(session: StudySession):
        """Load the current answer for grading display. Returns UI component values."""
        if session is None or session.grading_index >= len(session.answers):
            return None

        answer = session.answers[session.grading_index]
        progress = session.grading_progress_text()
        question = f"### {answer['question']}"
        user_answer = answer.get("user_answer", "(no answer)")
        analysis = format_analysis_display(answer.get("analysis", {}))
        reference = answer.get("learning_material", "")
        current_score = answer.get("score") if answer.get("score") is not None else 0

        return {
            "progress": progress,
            "question": question,
            "user_answer": user_answer,
            "analysis": analysis,
            "reference": reference,
            "score": current_score
        }

    def enter_grading_phase(session: StudySession):
        """Transition from answering to grading phase."""
        session.phase = "grading"
        session.grading_index = 0

        grading_data = load_grading_item(session)
        if grading_data is None:
            # No answers to grade, go straight to summary
            return finish_grading(session)

        return (
            session,
            grading_data["progress"],
            grading_data["question"],
            grading_data["user_answer"],
            grading_data["analysis"],
            grading_data["reference"],
            grading_data["score"],
            gr.update(visible=False),  # question_section
            gr.update(visible=True),   # grading_section
            gr.update(visible=False),  # summary_section
        )

    def handle_grade_submit(session: StudySession, score: int):
        """Save user's score for current answer and advance to next or summary."""
        if session is None:
            return (session,) + ("",) * 6 + (gr.update(),) * 3 + ("",)

        # Save the score
        session.answers[session.grading_index]["score"] = int(score)
        session.answers[session.grading_index]["graded_by"] = "human"
        session.grading_index += 1

        # Check if all answers have been graded
        if session.grading_index >= len(session.answers):
            return finish_grading(session)

        # Load next answer for grading
        grading_data = load_grading_item(session)
        return (
            session,
            grading_data["progress"],
            grading_data["question"],
            grading_data["user_answer"],
            grading_data["analysis"],
            grading_data["reference"],
            grading_data["score"],
            gr.update(visible=False),  # question_section
            gr.update(visible=True),   # grading_section
            gr.update(visible=False),  # summary_section
            "",  # summary_display
        )

    def handle_grade_back(session: StudySession):
        """Go back to previous answer in grading phase."""
        if session is None or session.grading_index <= 0:
            return (session,) + ("",) * 6 + (0,)

        session.grading_index -= 1
        grading_data = load_grading_item(session)

        return (
            session,
            grading_data["progress"],
            grading_data["question"],
            grading_data["user_answer"],
            grading_data["analysis"],
            grading_data["reference"],
            grading_data["score"],
        )

    def finish_grading(session: StudySession):
        """Complete grading phase, update badge metadata, and show summary."""
        # Update badge metadata
        if session.current_badge and session.session_data and session.session_path:
            update_badge_metadata(
                session.session_data,
                session.current_badge["id"],
                session.mode,
                session.total_score(),
                session.max_score()
            )
            save_session(session.session_data, session.session_path)

        # Generate and show summary
        summary = generate_summary(session)
        return (
            session,
            "",  # grading_progress
            "",  # grading_question
            "",  # grading_user_answer
            "",  # grading_analysis
            "",  # grading_reference
            0,   # score_slider
            gr.update(visible=False),  # question_section
            gr.update(visible=False),  # grading_section
            gr.update(visible=True),   # summary_section
            summary,  # summary_display
        )

    def generate_summary(session: StudySession) -> str:
        """Generate a summary of the session."""
        total = session.total_score()
        max_score = session.max_score()
        percentage = round(total / max_score * 100, 1) if max_score > 0 else 0
        avg = round(total / len(session.answers), 1) if session.answers else 0

        # Badge header
        if session.current_badge:
            badge_name = session.current_badge["name"]
            summary = f"# Badge Complete: {badge_name}\n\n"
        else:
            summary = "# Session Summary\n\n"

        summary += f"""## Overall Performance
- **Total Score:** {total}/{max_score} ({percentage}%)
- **Average per Question:** {avg}/10
- **Questions Answered:** {len(session.answers)}
- **Mode:** {session.mode.capitalize()}
- **Grading:** Human-in-the-loop

"""
        # Performance grade
        if percentage >= 80:
            summary += "**Grade: Excellent!** You've mastered this badge.\n\n"
        elif percentage >= 60:
            summary += "**Grade: Good!** Keep practicing to improve.\n\n"
        else:
            summary += "**Grade: Needs Work.** Review the learning material and try again.\n\n"

        summary += "## Performance by Category\n"
        for cat, perf in session.category_performance().items():
            emoji = "+" if perf["avg_score"] >= 7 else "~" if perf["avg_score"] >= 5 else "-"
            summary += f"- **{cat}:** {perf['avg_score']}/10 avg ({perf['count']} questions) {emoji}\n"

        summary += "\n## Detailed Results\n"
        for i, ans in enumerate(session.answers, 1):
            score = ans.get("score") or 0
            emoji = "+" if score >= 7 else "~" if score >= 5 else "-"
            q_short = ans["question"][:60] + "..." if len(ans["question"]) > 60 else ans["question"]
            summary += f"{i}. **[{score}/10]** {q_short} {emoji}\n"

        return summary

    def save_and_notify(session: StudySession):
        """Save results and notify user."""
        if session is None or not session.answers:
            return "No results to save."
        filepath = save_results(session)
        return f"Results saved to: `{filepath}`"

    def return_to_badge_selector(session: StudySession):
        """Return to badge selector, refreshing badge info."""
        # Reload session data to get updated metadata
        nonlocal session_data
        if os.path.exists(session_path):
            with open(session_path, "r", encoding="utf-8") as f:
                session_data = json.load(f)

        return (
            None,  # session_state
            "",    # progress
            "",    # question
            "",    # answer
            "",    # feedback
            "",    # peek
            gr.update(visible=True),   # badge_selector_section
            gr.update(visible=False),  # question_section
            gr.update(visible=False),  # grading_section
            gr.update(visible=False),  # summary_section
            gr.update(choices=get_badge_choices(), value=None),  # badge_dropdown
            "",    # badge_info
        )

    def get_next_badge_id(current_badge_id: str) -> str | None:
        """Get the ID of the next badge in sequence."""
        for i, badge in enumerate(session_data["badges"]):
            if badge["id"] == current_badge_id and i < len(session_data["badges"]) - 1:
                return session_data["badges"][i + 1]["id"]
        return None

    def start_next_badge(session: StudySession, mode: str, randomize: bool):
        """Start the next badge in sequence."""
        if session is None or session.current_badge is None:
            return return_to_badge_selector(session)

        next_badge_id = get_next_badge_id(session.current_badge["id"])
        if next_badge_id is None:
            return return_to_badge_selector(session)

        return start_badge_session(mode, randomize, next_badge_id)

    # Build UI
    with gr.Blocks(title="AWS Study Tool") as app:
        gr.Markdown("# AWS Study Tool")
        gr.Markdown(f"Loaded **{len(all_questions)}** questions in **{len(session_data['badges'])}** badges from `{data_path}`")

        session_state = gr.State(None)

        # Badge Selector Section
        with gr.Column(visible=True) as badge_selector_section:
            gr.Markdown("## Select a Badge")
            badge_dropdown = gr.Dropdown(
                choices=get_badge_choices(),
                label="Choose Badge",
                value=None,
                interactive=True
            )
            badge_info = gr.Markdown("")

            with gr.Row():
                mode_dropdown = gr.Dropdown(
                    choices=["Learning", "Exam"],
                    value="Learning",
                    label="Mode"
                )
                random_checkbox = gr.Checkbox(label="Shuffle Questions", value=False)
                start_btn = gr.Button("Start Badge", variant="primary")

        # Question section
        with gr.Column(visible=False) as question_section:
            progress_text = gr.Markdown("")
            question_display = gr.Markdown("")
            answer_input = gr.Textbox(
                lines=5,
                label="Your Answer",
                placeholder="Type your answer here..."
            )

            with gr.Row():
                submit_btn = gr.Button("Submit Answer", variant="primary")
                peek_btn = gr.Button("Peek 👀")
                skip_btn = gr.Button("Skip →")
                next_btn = gr.Button("Next Question →", visible=False, variant="secondary")

            feedback_display = gr.Markdown("")

            with gr.Accordion("Learning Material (Peek)", open=False):
                peek_display = gr.Markdown("")

        # Grading section (human-in-the-loop grading phase)
        with gr.Column(visible=False) as grading_section:
            grading_progress = gr.Markdown("")
            gr.Markdown("---")
            grading_question = gr.Markdown("")
            gr.Markdown("### Your Answer")
            grading_user_answer = gr.Markdown("")
            gr.Markdown("---")
            grading_analysis = gr.Markdown("")
            with gr.Accordion("Reference Material", open=False):
                grading_reference = gr.Markdown("")
            gr.Markdown("---")
            score_slider = gr.Slider(
                minimum=0, maximum=10, step=1, value=0,
                label="Your Score (0-10)"
            )
            with gr.Row():
                grade_back_btn = gr.Button("← Previous", variant="secondary")
                grade_submit_btn = gr.Button("Submit Grade & Next →", variant="primary")

        # Summary section
        with gr.Column(visible=False) as summary_section:
            summary_display = gr.Markdown("")
            with gr.Row():
                save_btn = gr.Button("Save Results", variant="primary")
                next_badge_btn = gr.Button("Next Badge", variant="secondary")
                badge_selector_btn = gr.Button("Return to Badge Selector")
            save_status = gr.Markdown("")

        # Wire up events

        # Badge dropdown change -> show badge info
        badge_dropdown.change(
            show_badge_info,
            inputs=[badge_dropdown],
            outputs=[badge_info]
        )

        # Start badge session
        start_btn.click(
            start_badge_session,
            inputs=[mode_dropdown, random_checkbox, badge_dropdown],
            outputs=[
                session_state, progress_text, question_display,
                answer_input, feedback_display, peek_display,
                badge_selector_section, question_section, grading_section, summary_section,
                peek_btn, submit_btn, next_btn
            ]
        )

        submit_btn.click(
            submit_answer,
            inputs=[session_state, answer_input],
            outputs=[session_state, feedback_display, submit_btn, next_btn]
        )

        next_btn.click(
            next_question,
            inputs=[session_state],
            outputs=[
                session_state, progress_text, question_display,
                answer_input, feedback_display, question_section,
                grading_section, summary_section, next_btn, summary_display,
                # Grading section components
                grading_progress, grading_question, grading_user_answer,
                grading_analysis, grading_reference, score_slider
            ]
        ).then(
            lambda s: gr.update(visible=True) if s and s.phase == "answering" and not s.is_complete() else gr.update(visible=False),
            inputs=[session_state],
            outputs=[submit_btn]
        )

        peek_btn.click(
            peek_material,
            inputs=[session_state],
            outputs=[peek_display]
        )

        skip_btn.click(
            skip_question,
            inputs=[session_state],
            outputs=[
                session_state, progress_text, question_display,
                answer_input, feedback_display, question_section,
                grading_section, summary_section, next_btn, summary_display,
                # Grading section components
                grading_progress, grading_question, grading_user_answer,
                grading_analysis, grading_reference, score_slider
            ]
        )

        save_btn.click(
            save_and_notify,
            inputs=[session_state],
            outputs=[save_status]
        )

        # Grading phase events
        grade_submit_btn.click(
            handle_grade_submit,
            inputs=[session_state, score_slider],
            outputs=[
                session_state,
                grading_progress, grading_question, grading_user_answer,
                grading_analysis, grading_reference, score_slider,
                question_section, grading_section, summary_section,
                summary_display
            ]
        )

        grade_back_btn.click(
            handle_grade_back,
            inputs=[session_state],
            outputs=[
                session_state,
                grading_progress, grading_question, grading_user_answer,
                grading_analysis, grading_reference, score_slider
            ]
        )

        # Next badge button
        next_badge_btn.click(
            start_next_badge,
            inputs=[session_state, mode_dropdown, random_checkbox],
            outputs=[
                session_state, progress_text, question_display,
                answer_input, feedback_display, peek_display,
                badge_selector_section, question_section, grading_section, summary_section,
                peek_btn, submit_btn, next_btn
            ]
        )

        # Return to badge selector
        badge_selector_btn.click(
            return_to_badge_selector,
            inputs=[session_state],
            outputs=[
                session_state, progress_text, question_display,
                answer_input, feedback_display, peek_display,
                badge_selector_section, question_section, grading_section, summary_section,
                badge_dropdown, badge_info
            ]
        )

    return app


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AWS Study Tool - Gradio App")
    parser.add_argument("data_path", help="Path to learning materials JSON file")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: File not found: {args.data_path}")
        return

    print(f"Loading learning materials from: {args.data_path}")
    app = create_app(args.data_path)
    print(f"Starting server on http://localhost:{args.port}")
    app.launch(server_port=args.port, share=args.share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
