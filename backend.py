import os
import signal
import socket
import subprocess
import time
from flask import Flask, jsonify, request, Response

app = Flask(__name__)

########################################
# 配置区（请按你的环境修改）
########################################

# 建议：直接使用环境中可执行文件的绝对路径，更稳定
# 例如：/home/shuyu/venv/vllm015/bin/vllm
#      /home/shuyu/venv/mineru/bin/mineru-api

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PID_DIR = os.path.join(BASE_DIR, "pids")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(PID_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SERVICES = {
    "llm": {
        "port": 8000,
        "cmd": [
            "/home/shuyu/venv/vllm015/bin/vllm",
            "serve",
            "GadflyII/GLM-4.7-Flash-NVFP4",
            "--trust-remote-code",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--enforce-eager",
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.85",
            "--disable-log-stats"
        ],
        "pid_file": os.path.join(PID_DIR, "llm.pid"),
        "log_file": os.path.join(LOG_DIR, "llm.log"),
    },

    "mineru": {
        "port": 6001,
        "cmd": [
            "/home/shuyu/venv/mineru/bin/mineru-api",
            "--host", "0.0.0.0",
            "--port", "6001"
        ],
        "pid_file": os.path.join(PID_DIR, "mineru.pid"),
        "log_file": os.path.join(LOG_DIR, "mineru.log"),
    }
}

########################################
# 工具函数
########################################

def check_port(port, host="127.0.0.1", timeout=1.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def read_pid(pid_file):
    if not os.path.exists(pid_file):
        return None
    try:
        with open(pid_file, "r") as f:
            return int(f.read().strip())
    except Exception:
        return None


def write_pid(pid_file, pid):
    with open(pid_file, "w") as f:
        f.write(str(pid))


def remove_pid(pid_file):
    if os.path.exists(pid_file):
        os.remove(pid_file)


def process_exists(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def derive_state(pid, port_open):
    """
    三态：
      stopped: 无进程且端口不通
      starting: 有进程但端口未通
      ready: 有进程且端口已通
    """
    if not pid or not process_exists(pid):
        return "stopped"
    if process_exists(pid) and not port_open:
        return "starting"
    return "ready"


def start_process(cfg):
    """
    异步启动：不等待模型加载完成
    将 stdout/stderr 重定向到日志文件
    """
    log_fp = open(cfg["log_file"], "ab", buffering=0)
    proc = subprocess.Popen(
        cfg["cmd"],
        stdout=log_fp,
        stderr=log_fp,
        preexec_fn=os.setsid  # 让子进程成为新进程组，便于 killpg
    )
    write_pid(cfg["pid_file"], proc.pid)
    return proc.pid


def stop_process(cfg, grace=8):
    """
    优雅关闭 -> 超时则强杀
    """
    pid = read_pid(cfg["pid_file"])
    if not pid or not process_exists(pid):
        remove_pid(cfg["pid_file"])
        return "not_running"

    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)

        # 等待优雅退出
        t0 = time.time()
        while time.time() - t0 < grace:
            if not process_exists(pid):
                remove_pid(cfg["pid_file"])
                return "stopped"
            time.sleep(0.5)

        # 超时强杀
        if process_exists(pid):
            os.killpg(pgid, signal.SIGKILL)
        remove_pid(cfg["pid_file"])
        return "killed"
    except Exception as e:
        return f"error: {e}"


########################################
# 6 个 GET 指令
########################################

@app.route("/<service>/status", methods=["GET"])
def status(service):
    if service not in SERVICES:
        return jsonify({"error": "unknown service"}), 404

    cfg = SERVICES[service]
    pid = read_pid(cfg["pid_file"])
    port_ok = check_port(cfg["port"])
    running = bool(pid and process_exists(pid))
    state = derive_state(pid, port_ok)

    return jsonify({
        "service": service,
        "state": state,           # stopped | starting | ready
        "running": running,
        "port_open": port_ok,
        "port": cfg["port"],
        "pid": pid
    })


@app.route("/<service>/start", methods=["GET"])
def start(service):
    if service not in SERVICES:
        return jsonify({"error": "unknown service"}), 404

    cfg = SERVICES[service]
    pid = read_pid(cfg["pid_file"])
    port_ok = check_port(cfg["port"])

    # 已就绪
    if pid and process_exists(pid) and port_ok:
        return jsonify({"service": service, "status": "already_running", "pid": pid})

    # 若 PID 存在但端口未开，视为 starting
    if pid and process_exists(pid) and not port_ok:
        return jsonify({"service": service, "status": "starting", "pid": pid})

    # 启动新进程（异步）
    new_pid = start_process(cfg)
    return jsonify({"service": service, "status": "starting", "pid": new_pid})


@app.route("/<service>/stop", methods=["GET"])
def stop(service):
    if service not in SERVICES:
        return jsonify({"error": "unknown service"}), 404

    cfg = SERVICES[service]
    result = stop_process(cfg)
    return jsonify({"service": service, "status": result})


########################################
# 可选增强：等待就绪（脚本友好）
########################################

@app.route("/<service>/wait_ready", methods=["GET"])
def wait_ready(service):
    """
    轮询直到端口可用或超时
    ?timeout=120  （秒）
    """
    if service not in SERVICES:
        return jsonify({"error": "unknown service"}), 404

    timeout = int(request.args.get("timeout", 120))
    cfg = SERVICES[service]

    t0 = time.time()
    while time.time() - t0 < timeout:
        if check_port(cfg["port"]):
            return jsonify({"service": service, "ready": True})
        time.sleep(2)

    return jsonify({"service": service, "ready": False, "timeout": timeout})


########################################
# 可选增强：日志查看 / 流式 tail
########################################

@app.route("/<service>/logs", methods=["GET"])
def logs(service):
    """
    默认返回最后 200 行
    ?lines=500
    """
    if service not in SERVICES:
        return jsonify({"error": "unknown service"}), 404

    lines = int(request.args.get("lines", 200))
    log_file = SERVICES[service]["log_file"]

    if not os.path.exists(log_file):
        return jsonify({"service": service, "logs": ""})

    # 读取最后 N 行
    with open(log_file, "rb") as f:
        data = f.readlines()[-lines:]
    return Response(b"".join(data), mimetype="text/plain")


@app.route("/<service>/logs/stream", methods=["GET"])
def logs_stream(service):
    """
    简单的流式日志（SSE-like，逐行输出）
    """
    if service not in SERVICES:
        return jsonify({"error": "unknown service"}), 404

    log_file = SERVICES[service]["log_file"]

    def generate():
        if not os.path.exists(log_file):
            yield b"(no log yet)\n"
            return
        with open(log_file, "rb") as f:
            # 从文件末尾开始跟随
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(1.0)
                    continue
                yield line

    return Response(generate(), mimetype="text/plain")


########################################
# 入口
########################################

if __name__ == "__main__":
    # 注意：不要开启 debug=True（会导致重复启动子进程）
    app.run(host="0.0.0.0", port=5050)
