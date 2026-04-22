# Step 1: Build the index (run ONCE)
python -m src.pipeline

# Step 2: Run the chatbot
python -m src.chatbot

# To rebuild the index
python -m src.pipeline --force