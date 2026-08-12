#!/usr/bin/env python3
"""Module that answers questions from multiple reference texts."""
question_answer_single = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """Answers questions from multiple reference texts in a loop.

    Args:
        corpus_path: path to the corpus of reference documents.
    """
    exit_words = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        question = input('Q: ')

        if question.lower() in exit_words:
            print('A: Goodbye')
            break

        reference = semantic_search(corpus_path, question)
        answer = question_answer_single(question, reference)

        if answer is None or answer == '':
            print('A: Sorry, I do not understand your question.')
        else:
            print('A: {}'.format(answer))


if __name__ == '__main__':
    question_answer('ZendeskArticles')
