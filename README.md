# learn-AWS
helper software for studying AWS


The Certification Exam Guides are the starting point:

1. **Copy the Task Statements** into separate `.txt` files,  
   **Copy all Technologies and Concepts** into one `.txt` file,  
   **Copy all In‑Scope AWS Services** into one `.txt` file  

   1. Create a prompt that, based on a task statement, generates an expert answer.  
   2. Iterate through all task statements and obtain expert answers for each.  
   3. Store the task statements together with the answers in a CSV file.  

2. **Write a learning program** that randomly selects an entry from the CSV and presents it to you as a question.  
   You, the learner, provide your answer in text.  
   The program opens a text editor where you can type your answer.  

3. An LLM will compare the model answer and your answer and give you feedback. *(learning mode)*  

4. There is also an **exam mode** in which the LLM decides, based on the model answer, how well your response was and stores the score.  
   The user can pre‑decide how many questions they will answer.  
   After the specified number of questions has been answered, the program displays a visualization of the results (e.g., a table) or percentage values, together with a recommendation of which tasks you have strengths or weaknesses in.  

→ **Create a corresponding SW (work instruction) and put it on GitHub.**  
   Document it so that it can be reused later.


3 Folder Structure/ Process

1) Input Folder = contains the learning topics, technologies, task statements etc. Also contains a prompt that will be used in the next step to create the actual learning materials with LLM.
2) Learning Materials = contains the learning materials dataset. 
3) exam mode Folder = contains the Results from the exam mode

create_learningMaterial.py = script that expects as an input the path to an input (inside the input folder). It will Iterate through the task statements and use the prompt to generate answers. It will save the results as a json to the learning materials folder. 

learning.py = Gradio-based study app with Learning and Exam modes. Provides LLM-powered feedback on your answers.

## How to Run

### Prerequisites

1. **Install dependencies:**
   ```bash
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start LM Studio (REQUIRED):**

   ⚠️ **Important:** You must manually start LM Studio before running the application!

   - **macOS/Linux**: Run LM Studio locally on your machine
   - **WSL2**: Run LM Studio as a Windows application (not inside WSL)
     - Open LM Studio on Windows
     - Load model: `nvidia/llama-3.1-nemotron-nano-8b-v1`
     - Go to "Local Server" or "Developer" tab
     - Click "Start Server" (make sure it's listening on `0.0.0.0:1234`)
     - The application will connect to Windows host automatically

### Running the Study App

**Quick start (recommended):**
```bash
./run.sh
```

**Manual start:**
```bash
source .venv/bin/activate
python learning.py LearningMaterials/clf-c02.json
```

**Note for WSL users:** If LM Studio connection fails, you may need to set the LM Studio URL:
```bash
export LM_STUDIO_URL="http://YOUR_WINDOWS_IP:1234/v1"
python learning.py LearningMaterials/clf-c02.json
```
To find your Windows IP, run `ipconfig` in Windows Command Prompt and look for the IPv4 address.

**Access the application:**
- **macOS/Linux**: Open `http://localhost:7860` in your local browser
- **WSL2**: Open `http://localhost:7860` in your Windows browser
  - The application automatically configures networking for WSL2
  - No manual configuration needed

**Expected output:**
```
Loading learning materials from: LearningMaterials/clf-c02.json
Starting server on http://localhost:7860
Binding Gradio to: 0.0.0.0:7860    # On WSL2
Binding Gradio to: 127.0.0.1:7860  # On macOS/Linux
```

### Cross-Platform Notes

The application automatically detects your environment (WSL2, macOS, Linux) and configures networking appropriately. See `.claude/CLAUDE.md` for detailed information about:
- Automatic platform detection
- Manual configuration overrides
- Troubleshooting connection issues

# later on, future Features:
* Derive Input File from PDF. (Exam guide.)
* use llm api with browser capability to get the most up to date information on services ... 