import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()

# Configure the library with your key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def is_valid_url(url):
    """Check if the URL is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def scrape_website(url):
    """Scrape content from a website"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text[:15000]  # Limit to 15k characters to avoid token limits
        
    except Exception as e:
        return f"Error scraping website: {str(e)}"

def summarize_website(url):
    """Summarize website content using Gemini"""
    if not is_valid_url(url):
        return "Please provide a valid URL (e.g., https://example.com)"
    
    website_content = scrape_website(url)
    
    if website_content.startswith("Error"):
        return website_content
    
    # Use Gemini to summarize
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    prompt = f"""
    Please summarize the following website content in a comprehensive yet concise manner.
    Focus on the main points, key information, and overall purpose of the website.
    
    Website content:
    {website_content}
    
    Summary:
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def answer_question_about_website(url, question):
    """Answer a specific question about website content"""
    if not is_valid_url(url):
        return "Please provide a valid URL (e.g., https://example.com)"
    
    website_content = scrape_website(url)
    
    if website_content.startswith("Error"):
        return website_content
    
    # Use Gemini to answer the question
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    prompt = f"""
    Based on the following website content, please answer this question: {question}
    
    Website content:
    {website_content}
    
    If the answer cannot be found in the content, please say so.
    
    Answer:
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating answer: {str(e)}"