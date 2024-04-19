#!/usr/bin/env python3
"""
Python script that provides some stats
about Nginx logs stored in MongoDB"""

from pymongo import MongoClient

from pymongo import MongoClient


def check_logs():
    """
    Python script that provides some stats about
    Nginx logs stored in MongoDB
    """
    client = MongoClient()
    db = client.logs
    collection = db.nginx

    total_logs = collection.count_documents({})

    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    method_counts = {}
    for method in methods:
        count = collection.count_documents({"method": method})
        method_counts[method] = count

    # Count number of documents with method=GET and path=/status
    status_check_count = collection.count_documents(
        {"method": "GET", "path": "/status"})

    return total_logs, method_counts, status_check_count

