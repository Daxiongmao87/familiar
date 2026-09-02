"""Shared pytest fixtures for the voice-chat-dm-assistant test suite.

Sets up sys.path so the ``dmd`` package is importable during tests.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
