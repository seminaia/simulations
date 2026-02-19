from transformers import pipeline

# Load the DeepSeek-Coder model for text generation
model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
coder = pipeline(
    task="text-generation",
    model=model_name
)

print("DeepSeek-Coder is ready! Type 'exit' to quit.")

while True:
    # Ask for user input
    prompt = input("\nYou: ")
    
    # Exit condition
    if prompt.strip().lower() == "exit":
        print("Goodbye!")
        break
    
    # Generate a response from the model
    response = coder(
        prompt,
        max_new_tokens=500,   # Limit on how long the reply can be
        temperature=0.7,      # Controls randomness (lower = more deterministic)
        do_sample=True        # Enable sampling for varied output
    )
    
    # Print the generated text
    print("\nDeepSeek-Coder:")
    print(response[0]["generated_text"])
