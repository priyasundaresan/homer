import os
import shutil
import threading
from flask import Flask, render_template, request, jsonify
from werkzeug.serving import make_server
import numpy as np
from io import BytesIO
from PIL import Image
import base64

from interactive_scripts.dataset_recorder import ActMode, load_episode, save_episode

app = Flask(__name__)
annotations = []
frames = []
server = None
demo_name = ""
export_event = threading.Event()

def load_demo(episode_fn):
    global demo_name
    demo = load_episode(episode_fn)
    demo_name = episode_fn
    return demo

def load_frames(demo):
    global frames
    frames.clear()
    for step in demo:
        if 'sideview_image' in step['obs']:
            frames.append(np.hstack((step['obs']['sideview_image'], step['obs']['wrist_image'])))
        elif 'viewer_image' in step['obs']:
            frames.append(np.hstack((step['obs']['viewer_image'], step['obs']['wrist_image'])))
        else:
            frames.append(np.hstack((step['obs']['base1_image'], step['obs']['wrist_image'])))

def relabel_demo(demo, annotations_list):
    for t, step in enumerate(demo):
        step['mode'] = annotations_list[t]
        step['waypoint_idx'] = -1  # no waypoint_idx concept anymore
        print(t, step['mode'])
    return demo

@app.route('/get_teleop_modes', methods=['GET'])
def get_teleop_modes():
    demo = load_demo(demo_name)
    teleop_modes = [step.get("teleop_mode", "unknown") for step in demo]
    return jsonify({"teleop_modes": teleop_modes})

@app.route('/')
def index():
    return render_template('index.html', frame_count=len(frames), demo_name=demo_name)

@app.route('/get_demo_name', methods=['GET'])
def get_demo_name():
    return jsonify({"demo_name": demo_name})

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    global annotations
    annotations = request.json  # frontend sends a list directly now
    return jsonify({"status": "success"})

@app.route('/frames/<int:frame_id>')
def get_frame(frame_id):
    if 0 <= frame_id < len(frames):
        img = frames[frame_id]
        img_pil = Image.fromarray(img.astype('uint8'))
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return jsonify({"image": f"data:image/png;base64,{img_str}"})
    return jsonify({"error": "Frame not found"}), 404

@app.route('/get_waypoints', methods=['GET'])
def get_waypoints():
    return jsonify({"waypoints": []})

@app.route('/export_annotations', methods=['GET'])
def export_annotations():
    global export_event
    export_event.set()
    return jsonify({"status": "received"})

def get_annotations(demo):
    global frames, annotations
    annotations_list = [ActMode.Dense] * len(frames)

    segments = [a for a in annotations if a.get("type") == "range"]

    for seg in segments:
        start, end = seg["start"], seg["end"]
        teleop_mode_start = demo[start].get("teleop_mode", "arm")
        annotations_list[start] = ActMode.ArmWaypoint if teleop_mode_start == "arm" else ActMode.BaseWaypoint
        for i in range(start + 1, min(len(demo), end)):
            annotations_list[i] = ActMode.Interpolate

    return annotations_list

def run_flask():
    global server
    server = make_server('127.0.0.1', 5001, app)
    server.serve_forever()

def stop_flask():
    global server
    if server:
        server.shutdown()
        print("Flask server stopped.")

if __name__ == '__main__':
    demo_dir = 'dev1'
    relabel_dir = 'dev1_relabeled'
    os.makedirs(relabel_dir, exist_ok=True)
    shutil.copy(os.path.join(demo_dir, 'env_cfg.yaml'), relabel_dir)

    print('Go to http://127.0.0.1:5001')

    for fn in sorted(os.listdir(demo_dir)):
        if 'pkl' in fn and fn not in os.listdir(relabel_dir):
            episode_fn = os.path.join(demo_dir, fn)
            print('Annotating:', episode_fn)
            demo = load_demo(episode_fn)
            load_frames(demo)

            flask_thread = threading.Thread(target=run_flask)
            flask_thread.start()

            print("Waiting for user to click 'Export Annotations'...")
            export_event.clear()
            export_event.wait()
            print("Annotated demo! Refresh to load next demo...")

            annotations_result = get_annotations(demo)
            demo_relabeled = relabel_demo(demo, annotations_result)
            save_episode(demo_relabeled, episode_fn.replace(demo_dir, relabel_dir))

            stop_flask()
            flask_thread.join()

