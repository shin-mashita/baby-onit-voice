"""Web dashboard wrapping the full-duplex voice agent.

Replaces the UI of the `tiny_vcm` app (../tmp/app) while keeping its command
set: timers, alarms, thermostat, lights, media, music, reminders, questions.
What changed is the routing -- MiniLM intent classification and slot grammars
gave way to an LLM calling the same actions as tools, which is why the agent can
also hold a conversation and run long jobs in the background.
"""
