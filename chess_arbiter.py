import os
import json
import time
import requests as req_lib
import chess
import chess.pgn
import datetime
import threading
from flask import Flask, jsonify
from dotenv import load_dotenv
import anthropic
from openai import OpenAI

load_dotenv()

# ── Clients API ───────────────────────────────────────────────────────────────
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Enregistrement des temps de réflexion ────────────────────────────────────
def _record_think(player, elapsed_ms):
    t = state["think_times"][player]
    t["total"] += elapsed_ms
    t["count"] += 1
    t["avg"]   = t["total"] // t["count"]
    # Historique horodaté pour moyennes glissantes
    entry = {"ts": int(time.time() * 1000), "ms": elapsed_ms}
    t.setdefault("history", []).append(entry)
    # Garder max 30 jours d'historique
    cutoff = int(time.time() * 1000) - 30 * 24 * 3600 * 1000
    t["history"] = [e for e in t["history"] if e["ts"] >= cutoff]
    save_state()

# ── État global ───────────────────────────────────────────────────────────────
state = {
    "fen": chess.STARTING_FEN,
    "moves": [],
    "game_number": 1,
    "scores": {"claude": 0, "gpt": 0, "draws": 0},
    "total_moves": 0,
    "turn": "white",
    "last_move": None,
    "start_date": datetime.datetime.now().strftime("%b %d, %Y"),
    "daily_data": [],
    "status": "playing",
    "next_game_at": None,
    "think_times": {
        "claude": {"avg": 0, "total": 0, "count": 0, "history": []},
        "gpt":    {"avg": 0, "total": 0, "count": 0, "history": []}
    }
}

PGN_FILE = "games.pgn"

# ── Persistance JSONBin ────────────────────────────────────────────────────────
_BIN_ID  = os.environ.get("JSONBIN_BIN_ID", "")
_BIN_KEY = os.environ.get("JSONBIN_API_KEY", "")
_BIN_URL = f"https://api.jsonbin.io/v3/b/{_BIN_ID}"
_BIN_HDR = {"Content-Type": "application/json", "X-Master-Key": _BIN_KEY}

# History bin
_HIST_BIN_ID  = os.environ.get("JSONBIN_HISTORY_BIN_ID", "")
_HIST_BIN_URL = f"https://api.jsonbin.io/v3/b/{_HIST_BIN_ID}"

def save_state():
    try:
        req_lib.put(_BIN_URL, json=state, headers=_BIN_HDR, timeout=10)
    except Exception as e:
        print(f"⚠️  JSONBin save error: {e}")

def load_state():
    global state
    try:
        r = req_lib.get(_BIN_URL + "/latest", headers=_BIN_HDR, timeout=10)
        if r.status_code == 200:
            saved = r.json().get("record", {})
            # If bin is fresh (game_number == 1, no games played), reset start_date to today
            if saved.get("game_number", 1) == 1 and saved.get("scores", {}).get("claude", 0) == 0 and saved.get("scores", {}).get("gpt", 0) == 0:
                saved["start_date"] = datetime.datetime.now().strftime("%b %d, %Y")
            state.update(saved)
            print("✅ State loaded from JSONBin")
        else:
            print(f"⚠️  JSONBin load error: {r.status_code}")
    except Exception as e:
        print(f"⚠️  JSONBin load error: {e}")

def save_game_to_history(game_number, result, pgn_str, move_count):
    """Append completed game to history bin."""
    hist_id = os.environ.get("JSONBIN_HISTORY_BIN_ID", "")
    if not hist_id:
        return
    hist_url = f"https://api.jsonbin.io/v3/b/{hist_id}"
    try:
        # Load current history
        r = req_lib.get(hist_url + "/latest", headers=_BIN_HDR, timeout=10)
        history = r.json().get("record", {}).get("games", []) if r.status_code == 200 else []

        # Map result to label
        if result == "1-0":
            winner = "Claude"
        elif result == "0-1":
            winner = "ChatGPT"
        else:
            winner = "Draw"

        history.append({
            "game": game_number,
            "date": datetime.datetime.utcnow().strftime("%Y-%m-%d"),
            "result": result,
            "winner": winner,
            "moves": move_count,
            "pgn": pgn_str
        })

        req_lib.put(hist_url, json={"games": history}, headers=_BIN_HDR, timeout=10)
        print(f"📚 Game #{game_number} saved to history")
    except Exception as e:
        print(f"⚠️  History save error: {e}")

# ── Cadence ───────────────────────────────────────────────────────────────────
def get_delay():
    """10 minutes fixes entre chaque partie"""
    return 600

# ── Appel Claude (joue les blancs) ────────────────────────────────────────────
def ask_claude(board):
    legal_moves = [board.san(m) for m in board.legal_moves]
    prompt = f"""Tu joues aux échecs. Tu joues les pièces blanches.
Position actuelle (FEN) : {board.fen()}
Coups légaux disponibles : {', '.join(legal_moves)}

Réponds UNIQUEMENT avec un coup en notation SAN (ex: e4, Nf3, O-O).
Si un pion atteint la dernière rangée, indique toujours la promotion (ex: e8=Q).
Ne donne aucune explication, juste le coup."""

    t0 = time.time()
    response = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}]
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    _record_think("claude", elapsed_ms)
    return response.content[0].text.strip()

# ── Appel GPT (joue les noirs) ────────────────────────────────────────────────
def ask_gpt(board):
    legal_moves = [board.san(m) for m in board.legal_moves]
    prompt = f"""Tu joues aux échecs. Tu joues les pièces noires.
Position actuelle (FEN) : {board.fen()}
Coups légaux disponibles : {', '.join(legal_moves)}

Réponds UNIQUEMENT avec un coup en notation SAN (ex: e5, Nf6, O-O-O).
Si un pion atteint la dernière rangée, indique toujours la promotion (ex: e1=Q).
Ne donne aucune explication, juste le coup."""

    t0 = time.time()
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}]
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    _record_think("gpt", elapsed_ms)
    return response.choices[0].message.content.strip()

# ── Valider et jouer un coup ──────────────────────────────────────────────────
def play_move(board, move_san, max_retries=3):
    """Tente de jouer un coup, retente si illégal."""
    for attempt in range(max_retries):
        try:
            # Nettoyer la réponse (enlever +, #, ?, ! éventuels)
            clean = move_san.replace("?", "").replace("!", "").strip()
            move = board.parse_san(clean)
            if move in board.legal_moves:
                uci = move.uci()
                from_sq = uci[:2]
                to_sq   = uci[2:4]
                board.push(move)
                return clean, from_sq, to_sq
        except Exception:
            pass

        # Si échec, demander un nouveau coup
        print(f"  Coup illégal '{move_san}', nouvelle tentative {attempt+1}...")
        if board.turn == chess.WHITE:
            move_san = ask_claude(board)
        else:
            move_san = ask_gpt(board)

    # En dernier recours, jouer un coup aléatoire légal
    import random
    move = random.choice(list(board.legal_moves))
    uci = move.uci()
    san = board.san(move)
    board.push(move)
    return san, uci[:2], uci[2:4]

# ── Mettre à jour les stats journalières ──────────────────────────────────────
def update_daily_data():
    today = datetime.datetime.now().strftime("%d/%m")
    daily = state["daily_data"]
    if daily and daily[-1]["label"] == today:
        daily[-1]["count"] += 1
    else:
        daily.append({"label": today, "count": 1})
    # Garder seulement les 14 derniers jours
    state["daily_data"] = daily[-14:]

# ── Boucle principale ─────────────────────────────────────────────────────────
def game_loop():
    load_state()
    print(f"🚀 AI Chess Battle démarré — Partie #{state['game_number']}")

    while True:
        board = chess.Board()

        # Reprendre depuis la position sauvegardée si en cours
        if state["fen"] != chess.STARTING_FEN and state["status"] == "playing":
            try:
                board.set_fen(state["fen"])
                print(f"  ↩️  Reprise de la partie #{state['game_number']}")
            except Exception:
                board = chess.Board()

        state["status"] = "playing"

        while not board.is_game_over():
            is_white = board.turn == chess.WHITE
            player   = "Claude" if is_white else "ChatGPT"

            # Mettre à jour le turn dans l'état
            state["turn"] = "white" if is_white else "black"
            state["fen"]  = board.fen()
            save_state()

            print(f"  ♟  {player} réfléchit...")

            try:
                if is_white:
                    raw_move = ask_claude(board)
                else:
                    raw_move = ask_gpt(board)

                san, from_sq, to_sq = play_move(board, raw_move)
                print(f"  ✅ {player} joue : {san}")

                state["fen"]        = board.fen()
                state["last_move"]  = {"from": from_sq, "to": to_sq}
                state["total_moves"] += 1
                state["moves"].append({"san": san, "color": "white" if is_white else "black"})
                state["turn"] = "black" if is_white else "white"
                save_state()

            except Exception as e:
                print(f"  ❌ Erreur API : {e}")
                time.sleep(10)
                continue

            # Délai entre les coups (2-4 secondes pour que ce soit lisible)
            time.sleep(3)

        # ── Fin de partie ──────────────────────────────────────────────────
        result = board.result()
        outcome = board.outcome()

        if result == "1-0":
            state["scores"]["claude"] += 1
            winner = "Claude 🎉"
        elif result == "0-1":
            state["scores"]["gpt"] += 1
            winner = "ChatGPT 🎉"
        else:
            state["scores"]["draws"] += 1
            winner = "Nulle 🤝"

        # Generate PGN string
        pgn_game = chess.pgn.Game.from_board(board)
        pgn_game.headers["Event"] = f"Infinite AI Chess Battle - Game {state['game_number']}"
        pgn_game.headers["White"] = "Claude (Anthropic)"
        pgn_game.headers["Black"] = "ChatGPT (OpenAI)"
        pgn_game.headers["Date"] = datetime.datetime.utcnow().strftime("%Y.%m.%d")
        pgn_game.headers["Result"] = result
        pgn_str = str(pgn_game)
        move_count = board.fullmove_number

        print(f"\n🏁 Partie #{state['game_number']} terminée — {winner}")
        print(f"   Score : Claude {state['scores']['claude']} - GPT {state['scores']['gpt']} - Nulles {state['scores']['draws']}\n")

        save_game_to_history(state["game_number"], result, pgn_str, move_count)
        update_daily_data()
        state["game_number"] += 1
        state["moves"]       = []
        state["fen"]         = chess.STARTING_FEN
        state["last_move"]   = None
        state["status"]      = "finished"
        save_state()

        # Délai intelligent entre les parties
        delay = get_delay()
        next_game_at = int((time.time() + delay) * 1000)
        state["next_game_at"] = next_game_at
        save_state()
        print(f"⏳ Prochaine partie dans {delay//60} minutes...")
        time.sleep(delay)
        state["next_game_at"] = None
        save_state()

# ── Serveur Flask ─────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/api/state")
def api_state():
    return jsonify(state)

@app.route("/")
def index():
    return app.send_static_file("viewer.html")

@app.route("/privacy-policy.html")
def mentions():
    return app.send_static_file("privacy-policy.html")

@app.route("/about.html")
def about():
    return app.send_static_file("about.html")

@app.route("/how-it-works.html")
def how_it_works():
    return app.send_static_file("how-it-works.html")

@app.route("/support.html")
def support():
    return app.send_static_file("support.html")

@app.route("/contact.html")
def contact():
    return app.send_static_file("contact.html")

@app.route("/history.html")
def history_page():
    return app.send_static_file("history.html")

@app.route("/api/history")
def api_history():
    hist_id = os.environ.get("JSONBIN_HISTORY_BIN_ID", "")
    if not hist_id:
        return jsonify({"games": []})
    try:
        hist_url = f"https://api.jsonbin.io/v3/b/{hist_id}/latest"
        r = req_lib.get(hist_url, headers=_BIN_HDR, timeout=10)
        games = r.json().get("record", {}).get("games", []) if r.status_code == 200 else []
        return jsonify({"games": list(reversed(games))})
    except Exception as e:
        return jsonify({"games": [], "error": str(e)})

# ── Démarrage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Lancer la boucle de jeu dans un thread séparé
    game_thread = threading.Thread(target=game_loop, daemon=True)
    game_thread.start()

    # Lancer le serveur web
    port = int(os.getenv("PORT", 5000))
    print(f"🌐 Serveur web sur http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
