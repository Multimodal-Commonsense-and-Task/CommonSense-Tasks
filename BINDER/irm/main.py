import os
import zmq
from factory import ReasoningFactory

def recv_array(socket, flags=0, copy=True, track=False):
    return socket.recv_pyobj()

if __name__ == "__main__":
    reasoner = ReasoningFactory.create("qwen")
    print(f"Loaded {reasoner.__class__.__name__}")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:1207")

    # Optional sample inference to warm up (skipped if the sample pickle is absent)
    sample_pickle_path = "./pickles/result_0423_132122.pkl"
    sample_prompt = f"""
You are given a short egocentric video from a robot.

Describe all objects visible in the video in a single sentence.
Do NOT include actions or locations. Just name visible objects.
e.g. "yellow cup, red box, apple"
"""
    if os.path.exists(sample_pickle_path):
        reasoner.run_from_pickle(sample_pickle_path, sample_prompt)
    else:
        print(f"Skipping warm-up: sample pickle not found at {sample_pickle_path}")

    while True:
        try:
            result_dict = recv_array(socket)
            result = reasoner.run_from_dict(result_dict, save=True)
            socket.send_pyobj(result)
        except Exception as e:
            print(f"Error: {e}")
            continue