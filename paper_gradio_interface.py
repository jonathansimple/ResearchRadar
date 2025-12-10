import gradio as gr
from paper_task_planner import inf_end_to_end
from datetime import datetime    

# --- your core pipeline function (replace with your actual text-to-sql pipeline) ---
def text_to_sql_pipeline(user_input):
    """
    Placeholder for your actual text-to-SQL + SQL execution pipeline.
    Should return Markdown-formatted text.
    """
    response = inf_end_to_end(user_input)
    return response
    
def chat_fn_1(user_message, history):
    if not user_message.strip():
        history.append({"role": "assistant", "content": "⚠️ 請輸入問題。"})
        return history

    response = text_to_sql_pipeline(user_message)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})
    return history
    
def chat_fn(user_message, history):
    history.append({"role": "user", "content": user_message})
    # Add a lightweight custom placeholder instead of the Gradio timer
    history.append({"role": "assistant", "content": "⌛ Processing your query, please wait a moment..."})
    yield history  # interim state

    # --- your actual pipeline ---
    result = text_to_sql_pipeline(user_message)

    # Replace the placeholder with the actual result
    history[-1] = {"role": "assistant", "content": result}
    yield history


# --- quick prompt autofill ---
def autofill_prompt(prompt):
    return prompt

# --- UI ---
def launch_chat():
    with gr.Blocks(title="ResearchRadar", theme=gr.themes.Soft(), css=".progress-bar {display: none !important;}") as demo:
        gr.Markdown("## ResearchRadar - Research Trend Aggregator")
        gr.Markdown("Which CS domain would you like to hear from?")

        # Chat history component (new format)
        chatbot = gr.Chatbot(label="History", type="messages", height=500)

        # Input row
        with gr.Row():
            user_input = gr.Textbox(
                placeholder="Enter your query，ex. \"What are the latest advancements in RAG?\" ",
                lines=2,
                show_label=False,
                scale=4,
            )
            send_btn = gr.Button("Send 🚀", scale=1)

        # Quick prompts row
        with gr.Row():
            sample_prompts = [
                "List out some papers published in December this year related to RAG",
                "How many CV papers were published in the last 3 months and what are their names?",
                "What is the main advancement presented in the paper Training-Time Action Conditioning for Efficient Real-Time Chunking?",
                "Has there been any LLM research lately?"
            ]
            buttons = [gr.Button(p) for p in sample_prompts]

        # Event connections
        send_btn.click(fn=chat_fn, inputs=[user_input, chatbot], outputs=chatbot)
        user_input.submit(fn=chat_fn, inputs=[user_input, chatbot], outputs=chatbot)

        # Autofill prompt buttons (no _js anymore)
        for i, btn in enumerate(buttons):
            btn.click(fn=lambda x=sample_prompts[i]: x, inputs=None, outputs=user_input)

    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # Optional but recommended on Windows

    launch_chat()
    
   
