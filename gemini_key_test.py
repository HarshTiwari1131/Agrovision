import os
import sys

from dotenv import load_dotenv


def main() -> int:
	load_dotenv()

	api_key = os.getenv("GEMINI_API_KEY")
	if not api_key:
		print("GEMINI_API_KEY not found in environment/.env file")
		return 1

	try:
		from google import genai
	except ImportError:
		print("Missing package: google-genai. Install it with: pip install google-genai python-dotenv")
		return 1

	try:
		client = genai.Client(api_key=api_key)
		response = client.models.generate_content(
			model="gemini-3.5-flash",
			contents="Say 'Gemini key is working.'",
		)
		print("Gemini key is valid.")
		print(response.text)
		return 0
	except Exception as exc:
		print(f"Gemini key test failed: {exc}")
		return 1


if __name__ == "__main__":
	sys.exit(main())
 
 
