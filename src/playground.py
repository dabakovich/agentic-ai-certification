from utils import get_llm, get_response_with_streaming_to_terminal, load_file, save_text_to_file

llm = get_llm(llm="ollama")

file = load_file("data/file.txt")

response = get_response_with_streaming_to_terminal(llm, f"Remove numbers: {file}")

save_text_to_file(response.content, "data/file_no_numbers.txt")
