from bottle import request, response

import bottle
import os
import queue
import threading
import traceback

from .spec import bytes_to_object, object_to_bytes, BPY_PORT, BPY_PAYLOAD_DIR_ENV

from ..rig_package.parser.bpy import BpyParser, transfer_rigging


def _payload_file(payload_path):
    """payload_path resolved, and only if it is a file directly in the payload directory."""
    payload_dir = os.environ.get(BPY_PAYLOAD_DIR_ENV)
    if not payload_dir:
        raise ValueError(f"payload_path needs {BPY_PAYLOAD_DIR_ENV}")
    if not isinstance(payload_path, str):
        raise ValueError("payload_path is not a string")
    resolved = os.path.realpath(payload_path)
    if os.path.dirname(resolved) != os.path.realpath(payload_dir):
        raise ValueError("payload_path is outside the payload directory")
    return resolved

def _resolve_payload(data):
    if isinstance(data, dict) and "payload_path" in data:
        payload_path = _payload_file(data["payload_path"])
        try:
            with open(payload_path, "rb") as f:
                return bytes_to_object(f.read())
        finally:
            try:
                os.remove(payload_path)
            except OSError:
                pass
    return data

def run():
    path_queue = queue.Queue()
    result_queue = queue.Queue()
    
    app = bottle.Bottle()
    
    @app.route('/load', method='GET') # type: ignore
    def load():
        data = request.body.read() # type: ignore
        path_queue.put(('load', data))
        res = result_queue.get()
        payload = object_to_bytes(res)
        response.content_type = 'application/octet-stream'  # type: ignore
        return payload
    
    @app.route('/ping', method='GET') # type: ignore
    def ping():
        return 'pong'
    
    @app.route('/export', method='post') # type: ignore
    def export():
        data = request.body.read() # type: ignore
        path_queue.put(('export', data))
        res = result_queue.get()
        payload = object_to_bytes(res)
        response.content_type = 'application/octet-stream'  # type: ignore
        return payload
    
    @app.route('/transfer', method='post') # type: ignore
    def transfer():
        data = request.body.read() # type: ignore
        path_queue.put(('transfer', data))
        res = result_queue.get()
        payload = object_to_bytes(res)
        response.content_type = 'application/octet-stream'  # type: ignore
        return payload
    
    def run_server(): bottle.run(app, host='127.0.0.1', port=BPY_PORT, server='tornado')
    threading.Thread(target=run_server, daemon=False).start()
    
    while True:
        d = path_queue.get()
        op = d[0]
        try:
            # The request body is a path or {"payload_path": path}: plain types only,
            # so nothing that reaches the port can unpickle an object.
            data = _resolve_payload(bytes_to_object(d[1], weights_only=True))
            if op == 'load':
                print("[SERVER] received load path:", data)
                asset = BpyParser.load(data)
                result_queue.put(asset)
            elif op == 'export':
                print("[SERVER] received export path:", data['filepath'])
                BpyParser.export(**data)
                result_queue.put('ok')
            elif op == 'transfer':
                print("[SERVER] received transfer path:", data['target_path'])
                transfer_rigging(**data)
                result_queue.put('ok')
            else:
                result_queue.put(f"unsupported op: {str(op)}")
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)
            result_queue.put({
                "error": f"{type(e).__name__}: {e}",
                "traceback": tb,
            })
