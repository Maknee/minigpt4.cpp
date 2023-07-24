import os
import sys
import ctypes
import pathlib
from typing import Optional, List
import enum
from pathlib import Path
import argparse
import gradio as gr

import minigpt4_library

title = """<h1 align="center">MiniGPT-4.cpp Demo</h1>"""
description = """<h3>This is the demo of MiniGPT-4 with ggml (cpu only!). Upload your images and start chatting!</h3>"""
article = """<div style='display:flex; gap: 0.25rem; '><a href='https://github.com/Vision-CAIR/MiniGPT-4'><img src='https://img.shields.io/badge/Github-Code-blue'></a></div>
"""
image_ready = False

minigpt4_chatbot: minigpt4_library.MiniGPT4ChatBot = None

def user(message, history):
    history = history or []
    # Append the user's message to the conversation history
    history.append([message, ""])
    return "", history

def chat(history, limit: int = 1024, temp: float = 0.8, top_k: int = 40, top_p: float = 0.9, repeat_penalty: float = 1.1):
    history = history or []

    if not image_ready:
        return "Please upload an image first.", history

    message = history[-1][0]

    history[-1][1] = ""
    for output in minigpt4_chatbot.generate(
        message,
        limit = int(limit),
        temp = float(temp),
        top_k = int(top_k),
        top_p = float(top_p),
    ):
        answer = output
        history[-1][1] += answer
        # stream the response
        yield history, history

def clear_state(history, chat_message, image):
    global image_ready
    history = []
    minigpt4_chatbot.reset_chat()
    image_ready = False
    return history, gr.update(value=None, interactive=True), gr.update(placeholder='Upload image first', interactive=False), gr.update(value="Upload & Start Chat", interactive=True)

def upload_image(image, history):
    global image_ready
    if image is None:
        return None, None, gr.update(interactive=True), history
    history = []
    minigpt4_chatbot.upload_image(image.convert('RGB'))
    image_ready = True
    return gr.update(interactive=False), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), history

def start(share: bool):
    with gr.Blocks() as demo:
        gr.Markdown(title)
        gr.Markdown(description)
        gr.Markdown(article)

        with gr.Row():
            with gr.Column(scale=0.5):
                image = gr.Image(type="pil")
                upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")

                max_tokens = gr.Slider(1, 1024, label="Max Tokens", step=1, value=128)
                temperature = gr.Slider(0.0, 1.0, label="Temperature", step=0.05, value=0.8)
                top_p = gr.Slider(0.0, 1.0, label="Top P", step=0.05, value=0.95)
                top_k = gr.Slider(0, 100, label="Top K", step=1, value=40)
                repeat_penalty = gr.Slider(0.0, 2.0, label="Repetition Penalty", step=0.1, value=1.1)

            with gr.Column():
                chatbot = gr.Chatbot(label='MiniGPT-4')
                message = gr.Textbox(label='User', placeholder='Upload image first', interactive=False)
                history = gr.State()

                with gr.Row():
                    submit = gr.Button(value="Send message", variant="secondary").style(full_width=True)
                    clear = gr.Button(value="Reset", variant="secondary").style(full_width=False)
                    # stop = gr.Button(value="Stop", variant="secondary").style(full_width=False)

        clear.click(clear_state, inputs=[history, image, message], outputs=[history, image, message, upload_button], queue=False)

        upload_button.click(upload_image, inputs=[image, history], outputs=[image, message, upload_button, history])
        
        submit_click_event = submit.click(
            fn=user, inputs=[message, history], outputs=[message, history], queue=True
        ).then(
            fn=chat, inputs=[history, max_tokens, temperature, top_p, top_k, repeat_penalty], outputs=[chatbot, history], queue=True
        )
        message_submit_event = message.submit(
            fn=user, inputs=[message, history], outputs=[message, history], queue=True
        ).then(
            fn=chat, inputs=[history, max_tokens, temperature, top_p, top_k, repeat_penalty], outputs=[chatbot, history], queue=True
        )
        # stop.click(fn=None, inputs=None, outputs=None, cancels=[submit_click_event, message_submit_event], queue=False)

    demo.launch(enable_queue=True, share=share)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test loading minigpt4')
    parser.add_argument('model_path', help='Path to model file')
    parser.add_argument('llm_model_path', help='Path to llm model file')
    parser.add_argument('--share_link', help='Share link publicly', default=False)
    args = parser.parse_args()

    model_path = args.model_path
    llm_model_path = args.llm_model_path
    share_link = args.share_link

    if not Path(model_path).exists():
        print(f'Model does not exist: {model_path}')
        exit(1) 

    if not Path(llm_model_path).exists():
        print(f'LLM Model does not exist: {llm_model_path}')
        exit(1)

    minigpt4_chatbot = minigpt4_library.MiniGPT4ChatBot(model_path, llm_model_path, verbosity=minigpt4_library.Verbosity.SILENT)
    start(share_link)
