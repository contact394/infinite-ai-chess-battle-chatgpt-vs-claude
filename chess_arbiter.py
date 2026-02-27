import os
import json
import time
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
    "start_date": datetime.datetime.now().strftime("%d %B %Y"),
    "daily_data": [],
    "status": "playing",
    "think_times": {
        "claude": {"avg": 0, "total": 0, "count": 0, "history": []},
        "gpt":    {"avg": 0, "total": 0, "count": 0, "history": []}
    }
}

STATE_FILE = "state.json"
PGN_FILE   = "games.pgn"

# ── Persistance ───────────────────────────────────────────────────────────────
def save_state():
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def load_state():
    global state
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            saved = json.load(f)
            state.update(saved)

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

        print(f"\n🏁 Partie #{state['game_number']} terminée — {winner}")
        print(f"   Score : Claude {state['scores']['claude']} - GPT {state['scores']['gpt']} - Nulles {state['scores']['draws']}\n")

        update_daily_data()
        state["game_number"] += 1
        state["moves"]       = []
        state["fen"]         = chess.STARTING_FEN
        state["last_move"]   = None
        state["status"]      = "finished"
        save_state()

        # Délai intelligent entre les parties
        delay = get_delay()
        print(f"⏳ Prochaine partie dans {delay//60} minutes...")
        time.sleep(delay)

# ── Serveur Flask ─────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/api/state")
def api_state():
    return jsonify(state)

@app.route("/")
def index():
    return app.send_static_file("viewer.html")

# ── Démarrage ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Lancer la boucle de jeu dans un thread séparé
    game_thread = threading.Thread(target=game_loop, daemon=True)
    game_thread.start()

    # Lancer le serveur web
    port = int(os.getenv("PORT", 5000))
    print(f"🌐 Serveur web sur http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
