"""
Docstring für learning. Do not remove this specification.

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
    Each question is a dict with: type, category, question, learning_material
    """
    questions = []

    # Process Domains
    for domain_name, tasks in data.get("Domains", {}).items():
        for task in tasks:
            if "learning-material" in task:
                questions.append({
                    "type": "domain",
                    "category": domain_name,
                    "question": task["task-statement"],
                    "learning_material": task["learning-material"]
                })

    # Process Technologies
    for tech in data.get("Technologies", []):
        if isinstance(tech, dict) and "learning-material" in tech:
            questions.append({
                "type": "technology",
                "category": "Technologies",
                "question": f"Explain the following AWS technology: {tech['technology']}",
                "learning_material": tech["learning-material"]
            })

    # Process AWS Services
    for service in data.get("AWS-Services", []):
        if isinstance(service, dict) and "learning-material" in service:
            questions.append({
                "type": "service",
                "category": "AWS Services",
                "question": f"Explain the following AWS service: {service['aws-service']}",
                "learning_material": service["learning-material"]
            })

    return questions


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


# =============================================================================
# Session State
# =============================================================================

@dataclass
class StudySession:
    """Manages the state of a study/exam session."""
    questions: list = field(default_factory=list)
    mode: str = "learning"  # "learning" or "exam"
    current_index: int = 0
    answers: list = field(default_factory=list)  # list of result dicts
    num_questions: int = 10

    def current_question(self) -> dict | None:
        if 0 <= self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    def is_complete(self) -> bool:
        return self.current_index >= self.num_questions or self.current_index >= len(self.questions)

    def total_score(self) -> int:
        return sum(a.get("score", 0) for a in self.answers)

    def max_score(self) -> int:
        return len(self.answers) * 10

    def category_performance(self) -> dict:
        """Calculate average score per category."""
        categories = {}
        for answer in self.answers:
            cat = answer.get("category", "Unknown")
            if cat not in categories:
                categories[cat] = {"total": 0, "count": 0}
            categories[cat]["total"] += answer.get("score", 0)
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

    def start_session(mode: str, num_q: int, randomize: bool):
        """Initialize a new study session."""
        questions = all_questions.copy()
        if randomize:
            random.shuffle(questions)

        session = StudySession(
            questions=questions[:num_q],
            mode=mode.lower(),
            num_questions=min(num_q, len(questions))
        )

        q = session.current_question()
        progress = f"**Question 1/{session.num_questions}** | Category: {q['category']}"
        question_text = f"### {q['question']}"

        # Show/hide peek button based on mode
        peek_visible = mode.lower() == "learning"

        return (
            session,  # session_state
            progress,  # progress_text
            question_text,  # question_display
            "",  # answer_input (clear)
            "",  # feedback_display
            "",  # peek_display
            gr.update(visible=True),  # question_section
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

        # Get feedback or grade based on mode
        if session.mode == "learning":
            feedback = get_feedback(q["question"], answer, q["learning_material"])
            grade = grade_answer(q["question"], answer, q["learning_material"])
            score = grade["score"]
            feedback_text = f"**Score: {score}/10**\n\n{feedback}"
        else:  # exam mode
            grade = grade_answer(q["question"], answer, q["learning_material"])
            score = grade["score"]
            feedback_text = f"**Answer recorded.** (Score will be shown at the end)"

        # Record the answer
        session.answers.append({
            "question": q["question"],
            "category": q["category"],
            "user_answer": answer,
            "score": score,
            "feedback": grade.get("feedback", "")
        })

        return (
            session,
            feedback_text,
            gr.update(visible=False),  # hide submit
            gr.update(visible=True),   # show next
        )

    def next_question(session: StudySession):
        """Move to the next question or show summary."""
        if session is None:
            return session, "", "", "", "", gr.update(), gr.update(), gr.update(), ""

        session.current_index += 1

        if session.is_complete():
            # Show summary
            summary = generate_summary(session)
            return (
                session,
                "",  # progress
                "",  # question
                "",  # answer
                "",  # feedback
                gr.update(visible=False),  # question_section
                gr.update(visible=True),  # summary_section
                gr.update(visible=False),  # next_btn
                summary,  # summary_display content
            )

        q = session.current_question()
        progress = f"**Question {session.current_index + 1}/{session.num_questions}** | Category: {q['category']}"
        question_text = f"### {q['question']}"

        return (
            session,
            progress,
            question_text,
            "",  # clear answer
            "",  # clear feedback
            gr.update(visible=True),  # question_section
            gr.update(visible=False),  # summary_section
            gr.update(visible=False),  # next_btn (will show after submit)
            "",  # summary_display (empty when not complete)
        )

    def peek_material(session: StudySession):
        """Show the learning material for current question."""
        if session is None or session.current_question() is None:
            return "No question loaded."
        return session.current_question()["learning_material"]

    def skip_question(session: StudySession):
        """Skip current question without answering."""
        if session is None:
            return session, "", "", "", "", gr.update(), gr.update(), gr.update(), ""

        q = session.current_question()
        if q:
            session.answers.append({
                "question": q["question"],
                "category": q["category"],
                "user_answer": "(skipped)",
                "score": 0,
                "feedback": "Question was skipped."
            })

        return next_question(session)

    def generate_summary(session: StudySession) -> str:
        """Generate a summary of the session."""
        total = session.total_score()
        max_score = session.max_score()
        avg = round(total / len(session.answers), 1) if session.answers else 0

        summary = f"""# Session Summary

## Overall Performance
- **Total Score:** {total}/{max_score} ({round(total/max_score*100, 1) if max_score > 0 else 0}%)
- **Average per Question:** {avg}/10
- **Questions Answered:** {len(session.answers)}

## Performance by Category
"""
        for cat, perf in session.category_performance().items():
            emoji = "✓" if perf["avg_score"] >= 7 else "△" if perf["avg_score"] >= 5 else "✗"
            summary += f"- **{cat}:** {perf['avg_score']}/10 avg ({perf['count']} questions) {emoji}\n"

        summary += "\n## Detailed Results\n"
        for i, ans in enumerate(session.answers, 1):
            score = ans["score"]
            emoji = "✓" if score >= 7 else "△" if score >= 5 else "✗"
            q_short = ans["question"][:60] + "..." if len(ans["question"]) > 60 else ans["question"]
            summary += f"{i}. **[{score}/10]** {q_short} {emoji}\n"

        return summary

    def save_and_notify(session: StudySession):
        """Save results and notify user."""
        if session is None or not session.answers:
            return "No results to save."
        filepath = save_results(session)
        return f"Results saved to: `{filepath}`"

    def reset_session():
        """Reset for a new session."""
        return (
            None,  # session_state
            "",    # progress
            "",    # question
            "",    # answer
            "",    # feedback
            "",    # peek
            gr.update(visible=False),  # question_section
            gr.update(visible=False),  # summary_section
        )

    # Build UI
    with gr.Blocks(title="AWS Study Tool", theme=gr.themes.Soft()) as app:
        gr.Markdown("# AWS Study Tool")
        gr.Markdown(f"Loaded **{len(all_questions)}** questions from `{data_path}`")

        session_state = gr.State(None)

        # Settings row
        with gr.Row():
            mode_dropdown = gr.Dropdown(
                choices=["Learning", "Exam"],
                value="Learning",
                label="Mode"
            )
            num_questions = gr.Slider(
                minimum=1,
                maximum=min(50, len(all_questions)),
                value=min(10, len(all_questions)),
                step=1,
                label="Number of Questions"
            )
            random_checkbox = gr.Checkbox(label="Random Order", value=False)
            start_btn = gr.Button("Start Session", variant="primary")

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

        # Summary section
        with gr.Column(visible=False) as summary_section:
            summary_display = gr.Markdown("")
            with gr.Row():
                save_btn = gr.Button("Save Results", variant="primary")
                new_session_btn = gr.Button("New Session")
            save_status = gr.Markdown("")

        # Wire up events
        start_btn.click(
            start_session,
            inputs=[mode_dropdown, num_questions, random_checkbox],
            outputs=[
                session_state, progress_text, question_display,
                answer_input, feedback_display, peek_display,
                question_section, summary_section, peek_btn, submit_btn, next_btn
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
                summary_section, next_btn, summary_display
            ]
        ).then(
            lambda s: gr.update(visible=True) if s and not s.is_complete() else gr.update(visible=False),
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
                summary_section, next_btn, summary_display
            ]
        )

        save_btn.click(
            save_and_notify,
            inputs=[session_state],
            outputs=[save_status]
        )

        new_session_btn.click(
            reset_session,
            outputs=[
                session_state, progress_text, question_display,
                answer_input, feedback_display, peek_display,
                question_section, summary_section
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
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
