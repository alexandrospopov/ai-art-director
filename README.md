---
title: AI Art Director
emoji: 🐢
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: apache-2.0
tags:
    - agent-demo-track
video:
    - https://youtu.be/JKPm7TI9vjI

---


Let me introduce, the AI Art Director !
If you’ve ever processed hundreds of RAW photos, you know the pain: post-processing quickly becomes a time sink.
What if an AI agent could do it for you?

The agent workflow is pretty simple :
The user inputs an image and the desired vibe : dreamy, sunny, badass, ...
A couple of llm, the art director, analyze the input image and the user's desire and deduce what filters should be applied and with which intensity : a lot, barely, ...
The agent interprets these directions and uses internal tools (12 tools) to apply the effects.
A Vision-Language Model (VLM) critic reviews the result and suggests improvements.
This process is repeated at most 5 times.

 Built with:

smolagent (agent orchestration)
Nebius endpoints (tool hosting)

Hackathon track:
Agent Demo Track
