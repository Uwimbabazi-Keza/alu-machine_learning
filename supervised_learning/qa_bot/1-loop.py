#!/usr/bin/env python3
"""takes in input from the user with the prompt Q: and prints A: as a response.
If the user inputs exit, quit, goodbye, or bye, case insensitive,
print A: Goodbye and exit"""


def main():
    exit_commands = ['exit', 'quit', 'goodbye', 'bye']
    
    while True:
        user_input = input('Q: ').strip().lower()
        
        if user_input in exit_commands:
            print('A: Goodbye')
            break
        
        print(f'A: {user_input}')

if __name__ == '__main__':
    main()
