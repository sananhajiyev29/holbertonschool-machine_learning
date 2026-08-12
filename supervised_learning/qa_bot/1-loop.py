#!/usr/bin/env python3
"""Script that loops taking user input and printing a response."""


def main():
    """Runs the input loop until the user exits."""
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        question = input('Q: ')

        if question.lower() in exit_words:
            print('A: Goodbye')
            break

        print('A:')


if __name__ == '__main__':
    main()
