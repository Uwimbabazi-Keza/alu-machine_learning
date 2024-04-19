#!/usr/bin/env python3

"""
Python function that lists
all documents in a collection:
"""


def list_all(mongo_collection):
    """Python function that lists
    all documents in a collection:
    """
    # check if collection is empty
    docs = []
    collection = mongo_collection.find()
    for doc in collection:
        docs.append(doc)

    return docs
