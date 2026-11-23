# Sourced by the scripts that boot a test server. Each server runs in its own session, so
# stopping it never touches another engine; it is stopped when the script exits.
SERVER_PID=

# require_free <port>: refuse to share the GPU or the port. The wait lets a server that was
# just stopped release its memory.
require_free() {
	local apps _
	for _ in $(seq 15); do
		apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader) \
			|| { echo "nvidia-smi failed: cannot tell whether the GPU is free" >&2; exit 1; }
		[ -z "$apps" ] && break
		sleep 2
	done
	[ -z "$apps" ] || { echo "the GPU is in use, not starting: $apps" >&2; exit 1; }
	if ss -ltn | awk '{print $4}' | grep -qE "[:.]$1$"; then
		echo "port $1 is already in use, not starting" >&2; exit 1
	fi
}

# start_server <port> <log> <command...>: VAR=value words may precede the command.
start_server() {
	local port=$1 log=$2; shift 2
	require_free "$port"
	rm -f "$log"
	setsid env "$@" > "$log" 2>&1 &
	SERVER_PID=$!
}

server_alive() { [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2> /dev/null; }

# wait_health <port> [seconds]: fails when the server dies or the deadline passes.
wait_health() {
	local end=$((SECONDS + ${2:-1200}))
	while [ $SECONDS -lt $end ] && server_alive; do
		curl -sf --max-time 3 "http://127.0.0.1:$1/health" > /dev/null && return 0
		sleep 2
	done
	return 1
}

stop_server() {
	[ -n "$SERVER_PID" ] || return 0
	local sid=$SERVER_PID _
	SERVER_PID=
	pkill -TERM -s "$sid" || true
	for _ in $(seq 60); do pgrep -s "$sid" > /dev/null || break; sleep 1; done
	pkill -KILL -s "$sid" || true
	wait "$sid" 2> /dev/null || true
}
trap stop_server EXIT
