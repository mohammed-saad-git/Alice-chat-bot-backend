from setuptools import setup
import sys

# Create an executable
setup(
    name="AliceChatbot",
    version="1.0.0",
    description="A modern AI chatbot desktop application",
    options={
        'build_exe': {
            'includes': [
                'customtkinter', 'PIL', 'google.generativeai', 'langchain',
                'langchain_google_genai', 'chromadb', 'bs4', 'requests',
                'threading', 'json', 'os', 're', 'base64', 'io', 'datetime',
                'urllib', 'tempfile', 'dotenv'
            ],
            'include_files': [
                ('user_data.json', 'user_data.json'),
                ('data.txt', 'data.txt'),
                ('.env', '.env')
            ]
        }
    },
    executables=[Executable("app.py", base="Win32GUI" if sys.platform == "win32" else None)]
)