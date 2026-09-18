#!/bin/bash
cd ~/Documents/MLchatbot
source agents_env/bin/activate
ollama serve &
sleep 2
streamlit run app.py