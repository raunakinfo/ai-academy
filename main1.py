from dotenv import load_dotenv
import os
print("hello")


load_dotenv()
print(f"Current Working Directory: {os.getcwd()}")
print(f"Is Key Found?: {'Yes' if os.getenv('GOOGLE_API_KEY') else 'No'}")

# python rag_pdf_audio_chroma.py \
#   --pdf "file-sample_150kB.pdf" \
#   --audio "LearningEnglishConversations-20260224-TheEnglishWeSpeakGoThroughARoughPatch.mp3" \
#   --persist_dir "./chroma_db" \
#   --collection "course_rag" \
#   --question "What are the main limitations discussed?"
