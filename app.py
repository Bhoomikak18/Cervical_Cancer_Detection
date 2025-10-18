import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
import numpy as np
import pickle
import os
import time
from supabase import create_client, Client
import uuid # We need this to create unique session IDs
from huggingface_hub import hf_hub_download


# --- NEW: Supabase Setup ---
from supabase import create_client, Client
import uuid # We need this to create unique session IDs


# Load keys from Vercel's Environment Variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


# This is the name of the bucket you just created
IMAGE_BUCKET = "cervix_images" 

# Initialize the Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase client initialized successfully.")
except Exception as e:
    print(f"🔥 Error initializing Supabase: {e}")
    print("Please double-check your SUPABASE_URL and SUPABASE_KEY.")









# --- 1. Define Model Architectures ---
DEVICE = torch.device("cpu") # Forcing CPU as per project goal

def build_model(num_classes):
    """Helper function to build our EfficientNet-B3 model structure."""
    model = models.efficientnet_b3(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

# --- 2. Load All Models ---
print("--- Loading all AI components for the application ---")

# Model 1: "Surveyor"
surveyor_model = build_model(num_classes=3)
surveyor_path = hf_hub_download(
    repo_id="Bhoomika18/cervical-cancer-models",
    filename="cervix_type_classifier_v2.pth"
)
surveyor_model.load_state_dict(torch.load(surveyor_path, map_location=DEVICE))
surveyor_model.eval().to(DEVICE)
print("✅ 'Surveyor' model (Type 1/2/3) loaded from Hugging Face.")

# Model 2: "Screener"
screener_model = build_model(num_classes=2)
screener_path = hf_hub_download(
    repo_id="Bhoomika18/cervical-cancer-models",
    filename="precancer_screener_model.pth"
)
screener_model.load_state_dict(torch.load(screener_path, map_location=DEVICE))
screener_model.eval().to(DEVICE)
print("✅ 'Screener' model (Normal/Abnormal) loaded from Hugging Face.")

# Model 3: "Grader"
grader_model = build_model(num_classes=2) # High-Grade, Low-Grade
grader_path = hf_hub_download(
    repo_id="Bhoomika18/cervical-cancer-models",
    filename="precancer_grader_model_binary.pth"
)
grader_model.load_state_dict(torch.load(grader_path, map_location=DEVICE))
grader_model.eval().to(DEVICE)
print("✅ 'Grader' model (Low-Grade/High-Grade) loaded from Hugging Face.")

# Agent 4: "Decision Agent"
q_table_path = hf_hub_download(
    repo_id="Bhoomika18/cervical-cancer-models",
    filename="rl_q_table_v2.pkl"
)
with open(q_table_path, 'rb') as f:
    q_table = pickle.load(f)
print("✅ RL Agent's Q-Table (V2) loaded from Hugging Face.")


# --- 3. Define Constants and Preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TYPE_CLASS_NAMES = ['Type 1', 'Type 2', 'Type 3']
SCREENER_CLASS_NAMES = ['Abnormal', 'Normal']
GRADER_CLASS_NAMES = ['High-Grade', 'Low-Grade']
ACTION_NAMES = ['Routine Follow-up', 'Schedule Biopsy / Treatment', 'Repeat Colposcopy']

# --- 4. Define Analysis Functions for Each Stage ---

# ✨ NEW: Helper function to save feedback to Supabase
# --- (This goes in Section 4) ---

def save_feedback(session_id, stage, image, ai_prediction, feedback, correct_label):
    """Uploads image to storage and saves feedback to the database."""
    
    # 1. Create a unique file name
    file_path = f"{session_id}/{stage}_{uuid.uuid4()}.png"
    
    # 2. Convert PIL Image to bytes to upload
    import io
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    image_url = ""
    try:
        # 3. Upload the image bytes
        supabase.storage.from_(IMAGE_BUCKET).upload(file=img_bytes, path=file_path, file_options={"content-type": "image/png"})
        
        # 4. Get the public URL
        res = supabase.storage.from_(IMAGE_BUCKET).get_public_url(file_path)
        image_url = res
        print(f"Image uploaded to: {image_url}") # Keep this print just in case

    except Exception as e:
        error_msg = f"🔥 Error uploading image: {e}"
        print(error_msg)
        return error_msg # <-- NEW: Return the error

    # 5. Insert feedback into the database table
    try:
        data = {
            "session_id": session_id,
            "stage": stage,
            "image_url": image_url,
            "ai_prediction": ai_prediction,
            "doctor_feedback": feedback,
            "correct_label": correct_label if feedback == "Disagree" else ai_prediction
        }
        supabase.table("feedback_log").insert(data).execute()
        
        success_msg = f"✅ Feedback for stage '{stage}' saved."
        print(success_msg)
        return success_msg # <-- NEW: Return the success message
        
    except Exception as e:
        error_msg = f"🔥 Error saving feedback to Supabase: {e}"
        print(error_msg)
        return error_msg # <-- NEW: Return the error


# --- Function for Stage 1: Surveyor ---
def run_surveyor_analysis(image):
    """
    Step 1: Runs ONLY the surveyor model and shows feedback buttons.
    """
    if image is None:
        return "Please upload an image.", None, None, None, None, gr.update(visible=False), gr.update(visible=False)

    # Generate a unique ID for this entire patient session
    session_id = str(uuid.uuid4())

    input_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = surveyor_model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        result = TYPE_CLASS_NAMES[pred_idx]

    surveyor_probabilities = {name: prob.item() for name, prob in zip(TYPE_CLASS_NAMES, probs)}

    if result == 'Type 1':
        conclusion = "Conclusion: Fully visible transformation zone. Please provide feedback below."
    elif result == 'Type 2':
        conclusion = "Conclusion: Transformation zone is partially visible. Please provide feedback below."
    else: # Type 3
        conclusion = "Conclusion: Transformation zone is not visible. Please provide feedback below."

    # We return:
    # 1. conclusion text
    # 2. probabilities label
    # 3. session_id (to save in State)
    # 4. prediction index (to save in State)
    # 5. prediction string (to save in State)
    # 6. the original image (to save in State for upload)
    # 7. SHOW the feedback group
    # 8. HIDE the 'rl_group' (in case it was visible from a Type 3)
    return (
        conclusion,
        surveyor_probabilities,
        session_id,
        pred_idx,
        result,
        image, 
        gr.update(visible=True),
        gr.update(visible=False)
    )


# --- Function for Stage 2: Screener ---
def run_screener_analysis(image):
    """
    Step 2: Runs ONLY the screener model and shows feedback buttons.
    """
    if image is None:
        return "Please upload an image.", None, None, None, gr.update(visible=False), gr.update(visible=False)

    input_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    screener_probabilities = None
    screener_result = None

    # Run Screener
    with torch.no_grad():
        screener_output = screener_model(input_tensor)
        screener_probs = torch.softmax(screener_output, dim=1)[0]

    screener_probabilities = {name: prob.item() for name, prob in zip(SCREENER_CLASS_NAMES, screener_probs)}
    screener_result = SCREENER_CLASS_NAMES[torch.argmax(screener_probs).item()]
    
    # We return:
    # 1. screener probabilities
    # 2. screener AI prediction string (to save in State)
    # 3. the original image (to save in State for upload)
    # 4. SHOW the screener feedback group
    # 5. HIDE the grader feedback group (in case it was visible)
    # 6. HIDE the RL group (in case it was visible)
    return (
        screener_probabilities,
        screener_result,
        image,
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False)
    )


# --- Function for Stage 3: Grader ---
def run_grader_analysis(screener_ai_pred, diagnostic_image):
    """
    Step 3: Runs ONLY the grader model (if needed) and shows feedback.
    """
    # If screener said "Normal", we skip this step and go to RL Agent
    if screener_ai_pred == 'Normal':
        # 0 = 'Normal' diagnosis for RL agent
        return "Skipped: Screener was 'Normal'", None, None, 0, gr.update(visible=False), gr.update(visible=True)

    input_tensor = preprocess(diagnostic_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    grader_probabilities = None
    grader_result = None
    final_diagnosis_idx = 0 # 0=Normal

    # Conditionally run Grader
    with torch.no_grad():
        grader_output = grader_model(input_tensor)
        grader_probs = torch.softmax(grader_output, dim=1)[0]
        grader_result = GRADER_CLASS_NAMES[torch.argmax(grader_probs).item()]
        final_diagnosis_idx = 1 if grader_result == 'Low-Grade' else 2 # 1=Low, 2=High
        
    grader_probabilities = {name: prob.item() for name, prob in zip(GRADER_CLASS_NAMES, grader_probs)}

    # We return:
    # 1. grader probabilities
    # 2. grader AI prediction string (to save in State)
    # 3. the final diagnosis index (for RL agent)
    # 4. SHOW the grader feedback group
    # 5. HIDE the RL group (in case it was visible)
    return (
        grader_probabilities,
        grader_result,
        final_diagnosis_idx,
        gr.update(visible=True),
        gr.update(visible=False)
    )

# --- Function for Stage 4: RL Agent ---
def run_rl_agent(surveyor_pred_idx, final_diagnosis_idx):
    """
    Step 4: Runs ONLY the RL agent and shows feedback buttons.
    """
    # Check for 'Type 3' from Surveyor. This is a special case.
    if surveyor_pred_idx == 2: # 2 = 'Type 3'
         # Manually set diagnosis to 'Normal' (0) as per your old logic
        final_diagnosis_idx = 0 
        
    rl_state = (surveyor_pred_idx, final_diagnosis_idx)
    
    # Handle case where state might not be in Q-table (defensive programming)
    if rl_state not in q_table:
        print(f"🔥 Error: RL State {rl_state} not found in Q-table. Defaulting to 'Routine Follow-up'.")
        best_action_idx = 0 # Default to 'Routine Follow-up'
    else:
        best_action_idx = np.argmax(q_table[rl_state])
        
    rl_recommendation = ACTION_NAMES[best_action_idx]

    # We return:
    # 1. The RL recommendation string (for display)
    # 2. The RL recommendation string (to save in State)
    # 3. SHOW the RL output group
    # 4. SHOW the RL feedback group
    return (
        rl_recommendation,
        rl_recommendation,
        gr.update(visible=True),
        gr.update(visible=True)
    )

def finish_and_reset():
    """
    Shows a final message, waits 3 seconds, and then resets the entire UI.
    This function is a generator, so it yields updates step-by-step.
    """
    
    # 1. First, yield the "Case finished" message.
    # We must return a value for ALL 17 outputs.
    # We create a list of 17 "no-change" updates.
    initial_updates = [gr.update() for _ in range(17)]
    # Then we change the one we care about: surveyor_output_text (index 1)
    initial_updates[1] = gr.update(value="Case finished. Thank you for your feedback! The interface will now reset.")
    yield tuple(initial_updates)
    
    # 2. Wait for 3 seconds
    time.sleep(3)
    
    # 3. Yield the final reset state (This is the corrected part)
    yield (
        gr.update(value=None, interactive=True), # 0: surveyor_input (Image)
        gr.update(value=None, interactive=True), # 1: surveyor_output_text (Textbox)
        gr.update(value=None), # 2: surveyor_output_probs (Label)
        gr.update(value=None), # 3: surveyor_feedback (Radio)
        gr.update(value=None, interactive=True), # 4: screener_grader_input (Image)
        gr.update(value=None), # 5: screener_output_probs (Label) <-- FIXED
        gr.update(value=None), # 6: grader_output_probs (Label)
        gr.update(value=None), # 7: screener_feedback (Radio)
        gr.update(value=None), # 8: grader_feedback (Radio)
        gr.update(value=None, interactive=True), # 9: rl_output_text (Textbox)
        gr.update(value=None), # 10: rl_feedback (Radio)
        gr.update(visible=False), # 11: surveyor_feedback_group
        gr.update(visible=False), # 12: screener_grader_group
        gr.update(visible=False), # 13: screener_feedback_group
        gr.update(visible=False), # 14: grader_feedback_group
        gr.update(visible=False), # 15: rl_group
        gr.update(visible=False)  # 16: rl_feedback_group
    )

# --- 5. Build the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI-Powered Cervical Cancer Diagnostic System")
    gr.Markdown("A step-by-step clinical workflow powered by three specialist AI models and an RL Decision Agent.")

    debug_output = gr.Textbox(label="🐞 DEBUG OUTPUT")

    # --- STATE VARIABLES ---
    # These are hidden "memory" for our app
    
    # Holds the unique ID for this patient session
    session_id_state = gr.State() 
    
    # Holds the Surveyor's prediction index (0, 1, or 2)
    surveyor_pred_idx_state = gr.State()
    
    # Holds the AI's string prediction (e.g., "Type 2")
    surveyor_ai_pred_state = gr.State()
    
    # Holds the PIL image for Supabase upload
    surveyor_image_state = gr.State() 
    
    # Holds the AI's string prediction (e.g., "Abnormal")
    screener_ai_pred_state = gr.State()
    
    # Holds the PIL image for Supabase upload
    diagnostic_image_state = gr.State() 
    
    # Holds the AI's string prediction (e.g., "High-Grade")
    grader_ai_pred_state = gr.State() 
    
    # Holds the AI's string recommendation (e.g., "Schedule Biopsy")
    rl_ai_pred_state = gr.State() 
    
    # Holds the final diagnosis index (0, 1, or 2) for the RL agent
    final_diagnosis_idx_state = gr.State() 


    # --- UI COMPONENTS ---

    with gr.Group():
        gr.Markdown("## Stage 1: Analyze Colposcopy View Quality")
        with gr.Row():
            surveyor_input = gr.Image(type="pil", label="Upload Initial Colposcopy Image")
            with gr.Column():
                surveyor_output_text = gr.Textbox(label="Conclusion", lines=3)
                surveyor_output_probs = gr.Label(label="Surveyor Model Probabilities")
        surveyor_button = gr.Button("1. Analyze View", variant="primary")

    # --- ✨ NEW: Surveyor Feedback Group (starts hidden) ---
    with gr.Group(visible=False) as surveyor_feedback_group:
        gr.Markdown("### 🩺 Doctor Feedback (Stage 1)")
        surveyor_feedback = gr.Radio(["Agree", "Disagree"], label="Was this prediction correct?")
        surveyor_correct_label = gr.Dropdown(TYPE_CLASS_NAMES, label="If wrong, select correct type", visible=False)
        surveyor_confirm_button = gr.Button("Confirm Feedback & Continue to Stage 2", variant="stop")

        # This makes the dropdown appear only if "Disagree" is selected
        def show_correct_surveyor(choice):
            return gr.update(visible=True) if choice == "Disagree" else gr.update(visible=False)
        surveyor_feedback.change(fn=show_correct_surveyor, inputs=surveyor_feedback, outputs=surveyor_correct_label)
    # --- End of new group ---

    with gr.Group(visible=False) as screener_grader_group:
        gr.Markdown("## Stage 2: Screen & Grade Abnormalities")
        with gr.Row():
            screener_grader_input = gr.Image(type="pil", label="Upload Diagnostic Image (Post-Acetic Acid/VILI)")
            with gr.Column():
                screener_output_probs = gr.Label(label="Screener Model Probabilities (Normal/Abnormal)")
                grader_output_probs = gr.Label(label="Grader Model Probabilities (Low/High-Grade)")
        screener_grader_button = gr.Button("2. Screen & Grade", variant="primary") # Renamed this button

    # --- ✨ NEW: Screener Feedback Group (starts hidden) ---
    with gr.Group(visible=False) as screener_feedback_group:
        gr.Markdown("### 🩺 Doctor Feedback (Stage 2: Screener)")
        screener_feedback = gr.Radio(["Agree", "Disagree"], label="Screener: Was this prediction correct?")
        screener_correct_label = gr.Dropdown(SCREENER_CLASS_NAMES, label="If wrong, select correct class", visible=False)
        screener_confirm_button = gr.Button("Confirm Screener Feedback & Run Grader", variant="stop")
        
        def show_correct_screener(choice):
            return gr.update(visible=True) if choice == "Disagree" else gr.update(visible=False)
        screener_feedback.change(fn=show_correct_screener, inputs=screener_feedback, outputs=screener_correct_label)
    # --- End of new group ---

    # --- ✨ NEW: Grader Feedback Group (starts hidden) ---
    with gr.Group(visible=False) as grader_feedback_group:
        gr.Markdown("### 🩺 Doctor Feedback (Stage 3: Grader)")
        grader_feedback = gr.Radio(["Agree", "Disagree"], label="Grader: Was this prediction correct?")
        grader_correct_label = gr.Dropdown(GRADER_CLASS_NAMES, label="If wrong, select correct grade", visible=False)
        grader_confirm_button = gr.Button("Confirm Grader Feedback & Run RL Agent", variant="stop")

        def show_correct_grader(choice):
            return gr.update(visible=True) if choice == "Disagree" else gr.update(visible=False)
        grader_feedback.change(fn=show_correct_grader, inputs=grader_feedback, outputs=grader_correct_label)
    # --- End of new group ---

    with gr.Group(visible=False) as rl_group:
        gr.Markdown("## Final Step: AI-Powered Recommendation")
        rl_output_text = gr.Textbox(label="Recommended Clinical Action (RL Agent)")

    # --- ✨ NEW: RL Agent Feedback Group (starts hidden) ---
    with gr.Group(visible=False) as rl_feedback_group:
        gr.Markdown("### 🩺 Doctor Feedback (Stage 4: Recommendation)")
        rl_feedback = gr.Radio(["Agree", "Disagree"], label="Do you agree with this recommendation?")
        rl_correct_label = gr.Dropdown(ACTION_NAMES, label="If wrong, select correct action", visible=False)
        rl_confirm_button = gr.Button("Save Final Feedback & Finish Case", variant="stop")

        def show_correct_rl(choice):
            return gr.update(visible=True) if choice == "Disagree" else gr.update(visible=False)
        rl_feedback.change(fn=show_correct_rl, inputs=rl_feedback, outputs=rl_correct_label)
    # --- End of new group ---


    # --- 6. Define App Logic (EVENT LISTENERS) ---
    
    # We need a function to reset the UI for a new case
    def reset_all():
        """Hides all groups and clears all outputs for a new session."""
        return (
            None, gr.update(value=None, interactive=True), gr.update(value=None), gr.update(value=None),
            None, gr.update(value=None, interactive=True), gr.update(value=None), gr.update(value=None),
            gr.update(value=None), gr.update(value=None), gr.update(value=None),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=False)
        )

    # --- STAGE 1 LOGIC ---
    
    # When "1. Analyze View" is clicked:
    # 1. Run the Surveyor AI
    # 2. Show the results and the Surveyor feedback group
    surveyor_button.click(
        fn=run_surveyor_analysis,
        inputs=surveyor_input,
        outputs=[
            surveyor_output_text,
            surveyor_output_probs,
            session_id_state,
            surveyor_pred_idx_state,
            surveyor_ai_pred_state,
            surveyor_image_state,
            surveyor_feedback_group,
            rl_group # This is to hide the RL group if it was a Type 3
        ]
    )

    # When "Confirm Feedback & Continue to Stage 2" is clicked:
    # 1. Save the feedback to Supabase
    # 2. Hide the feedback group
    # 3. Show the Stage 2 upload group
    surveyor_confirm_button.click(
        fn=save_feedback,
        inputs=[
            session_id_state,
            gr.State("Surveyor"), # Pass the stage name
            surveyor_image_state,
            surveyor_ai_pred_state,
            surveyor_feedback,
            surveyor_correct_label
        ],
        outputs=debug_output # We don't need the image_url output
    ).then(
        lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(value=None), gr.update(value=None)), # Hide feedback, show Stage 2
        outputs=[surveyor_feedback_group, screener_grader_group, surveyor_feedback, surveyor_correct_label]
    )


    # --- STAGE 2 LOGIC ---
    
    # When "2. Screen & Grade" is clicked:
    # 1. Run the Screener AI
    # 2. Show the results and the Screener feedback group
    screener_grader_button.click(
        fn=run_screener_analysis,
        inputs=screener_grader_input,
        outputs=[
            screener_output_probs,
            screener_ai_pred_state,
            diagnostic_image_state,
            screener_feedback_group,
            grader_feedback_group,
            rl_group
        ]
    )

    # When "Confirm Screener Feedback & Run Grader" is clicked:
    # 1. Save the Screener feedback to Supabase
    # 2. Run the Grader AI
    # 3. Show the Grader results
    screener_confirm_button.click(
        fn=save_feedback,
        inputs=[
            session_id_state,
            gr.State("Screener"),
            diagnostic_image_state,
            screener_ai_pred_state,
            screener_feedback,
            screener_correct_label
        ],
        outputs=debug_output
    ).then(
        fn=run_grader_analysis,
        inputs=[
            screener_ai_pred_state,
            diagnostic_image_state
        ],
        outputs=[
            grader_output_probs,
            grader_ai_pred_state,
            final_diagnosis_idx_state,
            grader_feedback_group,
            rl_group
        ]
    ).then(
        lambda: (gr.update(visible=False), gr.update(value=None), gr.update(value=None)), # Hide screener feedback group
        outputs=[screener_feedback_group, screener_feedback, screener_correct_label]
    )


    # --- STAGE 3 LOGIC ---
    
    # When "Confirm Grader Feedback & Run RL Agent" is clicked:
    # 1. Save the Grader feedback to Supabase
    # 2. Run the RL Agent
    # 3. Show the RL recommendation and feedback group
    grader_confirm_button.click(
        fn=save_feedback,
        inputs=[
            session_id_state,
            gr.State("Grader"),
            diagnostic_image_state,
            grader_ai_pred_state,
            grader_feedback,
            grader_correct_label
        ],
        outputs=debug_output
    ).then(
        fn=run_rl_agent,
        inputs=[
            surveyor_pred_idx_state,
            final_diagnosis_idx_state
        ],
        outputs=[
            rl_output_text,
            rl_ai_pred_state,
            rl_group,
            rl_feedback_group
        ]
    ).then(
        lambda: (gr.update(visible=False), gr.update(value=None), gr.update(value=None)), # Hide grader feedback
        outputs=[grader_feedback_group, grader_feedback, grader_correct_label]
    )


    # --- STAGE 4 LOGIC ---
    
    # When "Save Final Feedback & Finish Case" is clicked:
    # 1. Save the RL Agent feedback to Supabase
    # 2. Call our new 'finish_and_reset' generator function
    rl_confirm_button.click(
        fn=save_feedback,
        inputs=[
            session_id_state,
            gr.State("RL_Agent"),
            diagnostic_image_state, # We re-use the diagnostic image for context
            rl_ai_pred_state,
            rl_feedback,
            rl_correct_label
        ],
        outputs=debug_output
    ).then(
        # This one call now handles the message, delay, and reset
        fn=finish_and_reset,
        outputs=[
            surveyor_input, surveyor_output_text, surveyor_output_probs, surveyor_feedback,
            screener_grader_input, screener_output_probs, grader_output_probs, screener_feedback,
            grader_feedback, rl_output_text, rl_feedback,
            surveyor_feedback_group, screener_grader_group, screener_feedback_group,
            grader_feedback_group, rl_group, rl_feedback_group
        ]
    )

demo.launch(debug=False)