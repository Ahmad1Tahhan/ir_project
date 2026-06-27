"""SOA layer — independent FastAPI services.

Each service wraps one responsibility from the core ``irsys`` library and can be
run / tested standalone. The API gateway is the single entry point used by the
UI and routes requests to the other services over REST.
"""
