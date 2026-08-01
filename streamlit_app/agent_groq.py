import os
from groq import Groq

def get_groq_response(messages, context=None, api_key=None):
    if not api_key:
        return "Error: Groq API key is missing. Please provide it in the sidebar."
        
    try:
        client = Groq(api_key=api_key)
        
        system_prompt = (
            "You are a helpful, empathetic medical AI assistant embedded in the DermaVision skin lesion analysis app. "
            "Your role is to explain the model's classification results to the user and answer their questions about skin health. "
            "Always remind the user that you are an AI and they should consult a real doctor for medical advice. "
        )
        if context:
            system_prompt += f"\n\nThe current analysis context for this user's skin lesion is: {context}. "
        
        # Prepend system prompt
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        
        chat_completion = client.chat.completions.create(
            messages=full_messages,
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=512,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error connecting to Groq API: {str(e)}"
