class Conversation:

    def run(self):
        while True:
            user_input = input("You: ")

            if user_input.lower() == "q":
                break

            print("AI: " + f"You sad: '{user_input}'")