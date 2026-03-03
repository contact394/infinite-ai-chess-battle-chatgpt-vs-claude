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
    entry = {"ts": int(time.time() * 1000), "ms": elapsed_ms}
    t.setdefault("history", []).append(entry)
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
            if saved.get("game_number", 1) == 1 and saved.get("scores", {}).get("claude", 0) == 0 and saved.get("scores", {}).get("gpt", 0) == 0:
                saved["start_date"] = datetime.datetime.now().strftime("%b %d, %Y")
            for key, value in state.items():
                if key not in saved:
                    saved[key] = value
                elif isinstance(value, dict):
                    for subkey, subval in value.items():
                        if subkey not in saved[key]:
                            saved[key][subkey] = subval
                        elif isinstance(subval, dict):
                            for k, v in subval.items():
                                if k not in saved[key][subkey]:
                                    saved[key][subkey][k] = v
            state.update(saved)
            print("✅ State loaded from JSONBin")
        else:
            print(f"⚠️  JSONBin load error: {r.status_code}")
    except Exception as e:
        print(f"⚠️  JSONBin load error: {e}")

def save_game_to_history(game_number, result, pgn_str, move_count):
    hist_id = os.environ.get("JSONBIN_HISTORY_BIN_ID", "")
    if not hist_id:
        return
    hist_url = f"https://api.jsonbin.io/v3/b/{hist_id}"
    try:
        r = req_lib.get(hist_url + "/latest", headers=_BIN_HDR, timeout=10)
        history = r.json().get("record", {}).get("games", []) if r.status_code == 200 else []
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
    now = datetime.datetime.now()
    next_hour = (now + datetime.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return int((next_hour - now).total_seconds())

# ══════════════════════════════════════════════════════════════════════════════
# ── ANALYSE APPROFONDIE DU PLATEAU ───────────────────────────────────────════
# ══════════════════════════════════════════════════════════════════════════════

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0
}

PIECE_NAMES = {
    chess.PAWN:   "Pawn",
    chess.KNIGHT: "Knight",
    chess.BISHOP: "Bishop",
    chess.ROOK:   "Rook",
    chess.QUEEN:  "Queen",
    chess.KING:   "King"
}

def material_count(board):
    """Compte le matériel pour chaque camp."""
    white, black = 0, 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            val = PIECE_VALUES[piece.piece_type]
            if piece.color == chess.WHITE:
                white += val
            else:
                black += val
    return white, black

def piece_inventory(board, color):
    """Liste les pièces présentes pour une couleur."""
    counts = {pt: 0 for pt in PIECE_VALUES}
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            counts[piece.piece_type] += 1
    parts = []
    for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
        if counts[pt] > 0:
            parts.append(f"{counts[pt]}x {PIECE_NAMES[pt]}")
    return ", ".join(parts)

def detect_game_phase(board):
    """Détermine la phase de jeu : ouverture, milieu, finale."""
    queens_on_board = (
        len(board.pieces(chess.QUEEN, chess.WHITE)) +
        len(board.pieces(chess.QUEEN, chess.BLACK))
    )
    total_minor = sum(
        len(board.pieces(pt, c))
        for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        for c in [chess.WHITE, chess.BLACK]
    )
    move_number = board.fullmove_number

    if move_number <= 10:
        return "OPENING"
    elif queens_on_board == 0 or (queens_on_board <= 1 and total_minor <= 4):
        return "ENDGAME"
    else:
        return "MIDDLEGAME"

def pawn_structure_analysis(board, color):
    """Analyse la structure de pions : doublés, isolés, passés."""
    pawns     = board.pieces(chess.PAWN, color)
    opp_pawns = board.pieces(chess.PAWN, not color)

    files_with_pawns = [chess.square_file(sq) for sq in pawns]
    issues = []

    # Pions doublés
    for f in range(8):
        if files_with_pawns.count(f) > 1:
            issues.append(f"doubled pawns on file {chess.FILE_NAMES[f]}")

    # Pions isolés
    for sq in pawns:
        f = chess.square_file(sq)
        adjacent_files = [f - 1, f + 1]
        has_neighbor = any(chess.square_file(p) in adjacent_files for p in pawns)
        if not has_neighbor:
            issues.append(f"isolated pawn on {chess.square_name(sq)}")

    # Pions passés
    passed = []
    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        is_passed = True
        for opp_sq in opp_pawns:
            opp_f = chess.square_file(opp_sq)
            opp_r = chess.square_rank(opp_sq)
            if opp_f in [f - 1, f, f + 1]:
                if color == chess.WHITE and opp_r > r:
                    is_passed = False
                    break
                elif color == chess.BLACK and opp_r < r:
                    is_passed = False
                    break
        if is_passed:
            passed.append(chess.square_name(sq))

    return issues, passed

def king_safety(board, color):
    """Évalue la sécurité du roi."""
    king_sq = board.king(color)
    if king_sq is None:
        return "unknown"

    in_check = board.is_check() and board.turn == color
    attackers = board.attackers(not color, king_sq)

    # Pions boucliers
    shield_count = 0
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    shield_rank = king_rank + (1 if color == chess.WHITE else -1)
    if 0 <= shield_rank <= 7:
        for df in [-1, 0, 1]:
            sf = king_file + df
            if 0 <= sf <= 7:
                shield_sq = chess.square(sf, shield_rank)
                piece = board.piece_at(shield_sq)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    shield_count += 1

    status = []
    if in_check:
        status.append("IN CHECK ⚠️")
    if len(attackers) > 0 and not in_check:
        status.append(f"under threat from {len(attackers)} piece(s)")
    if shield_count == 0:
        status.append("exposed king (no pawn shield)")
    elif shield_count >= 2:
        status.append(f"well-protected ({shield_count} pawn shield)")

    return "; ".join(status) if status else "safe"

def tactical_alerts(board, color):
    """Détecte les tactiques immédiates : mat en 1, pièces en prise, coups d'échec."""
    alerts = []

    # Travailler sur une COPIE du plateau pour ne jamais altérer l'original
    b = board.copy()
    legal_moves_list = list(b.legal_moves)

    # Pré-calcul SAN sur la copie avant tout push/pop
    move_san_map = {}
    for move in legal_moves_list:
        try:
            move_san_map[move.uci()] = b.san(move)
        except Exception:
            move_san_map[move.uci()] = move.uci()

    # Mat en 1 — priorité absolue
    for move in legal_moves_list:
        try:
            b.push(move)
            if b.is_checkmate():
                san = move_san_map.get(move.uci(), move.uci())
                alerts.append(f"🏆 CHECKMATE IN ONE: {san} — play this immediately!")
            b.pop()
        except Exception:
            try:
                b.pop()
            except Exception:
                pass

    # Pièces en prise non défendues (hanging) — sur le plateau original, sans push/pop
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.color == color:
            attackers = board.attackers(not color, sq)
            defenders = board.attackers(color, sq)
            if attackers and not defenders and piece.piece_type != chess.KING:
                alerts.append(
                    f"⚠️  {PIECE_NAMES[piece.piece_type]} on {chess.square_name(sq)} "
                    f"is HANGING (undefended & attacked)"
                )

    # Coups donnant échec disponibles
    checking_moves = []
    for move in legal_moves_list:
        try:
            b2 = board.copy()
            b2.push(move)
            if b2.is_check():
                san = move_san_map.get(move.uci(), move.uci())
                checking_moves.append(san)
        except Exception:
            pass
    if checking_moves:
        alerts.append(f"✅ Moves that give check: {', '.join(checking_moves[:6])}")

    return alerts

def format_move_history(board, last_n=14):
    """Formate les N derniers coups joués en notation SAN."""
    moves = list(board.move_stack)
    if not moves:
        return "No moves played yet (starting position)"

    temp = chess.Board()
    san_moves = []
    for m in moves:
        san_moves.append(temp.san(m))
        temp.push(m)

    recent = san_moves[-last_n:]
    offset = len(san_moves) - len(recent)
    formatted = []
    i = 0
    move_num = offset // 2 + 1
    while i < len(recent):
        white_mv = recent[i] if i < len(recent) else "..."
        black_mv = recent[i + 1] if i + 1 < len(recent) else ""
        formatted.append(f"{move_num}. {white_mv} {black_mv}".strip())
        i += 2
        move_num += 1

    return " ".join(formatted)

def build_board_analysis(board, color):
    """Construit une analyse complète du plateau pour le prompt IA."""
    is_white         = (color == chess.WHITE)
    my_color_name    = "WHITE" if is_white else "BLACK"
    enemy_color_name = "BLACK" if is_white else "WHITE"
    opp_color        = not color

    # Matériel
    w_mat, b_mat = material_count(board)
    my_mat    = w_mat if is_white else b_mat
    enemy_mat = b_mat if is_white else w_mat
    mat_diff  = my_mat - enemy_mat

    # Inventaire des pièces
    my_pieces    = piece_inventory(board, color)
    enemy_pieces = piece_inventory(board, opp_color)

    # Phase de jeu
    phase = detect_game_phase(board)

    # Structure de pions
    my_pawn_issues, my_passed = pawn_structure_analysis(board, color)
    en_pawn_issues, en_passed = pawn_structure_analysis(board, opp_color)

    # Sécurité des rois
    my_king_safety    = king_safety(board, color)
    enemy_king_safety = king_safety(board, opp_color)

    # Coups légaux SAN — AVANT tout push/pop dans les sous-fonctions
    legal_moves_snap = list(board.legal_moves)
    legal_moves_san  = sorted([board.san(m) for m in legal_moves_snap])

    # Mobilité
    my_mobility = len(legal_moves_snap)

    # Contrôle du centre
    center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
    my_center    = sum(1 for sq in center_squares if board.piece_at(sq) and board.piece_at(sq).color == color)
    enemy_center = sum(1 for sq in center_squares if board.piece_at(sq) and board.piece_at(sq).color == opp_color)

    # Historique des coups
    move_history = format_move_history(board, last_n=14)

    # Alertes tactiques (push/pop internes — liste des coups déjà gelée)
    tactics = tactical_alerts(board, color)

    lines = [
        "=" * 62,
        f"  CHESS POSITION ANALYSIS — YOU PLAY {my_color_name}",
        "=" * 62,
        "",
        f"  FEN        : {board.fen()}",
        f"  Game Phase : {phase}",
        f"  Move Number: {board.fullmove_number}",
        "",
        "── MATERIAL ──────────────────────────────────────────────────",
        f"  Your pieces   ({my_color_name:<5}): {my_pieces}",
        f"  Enemy pieces  ({enemy_color_name:<5}): {enemy_pieces}",
        f"  Balance: {'+' if mat_diff >= 0 else ''}{mat_diff} cp "
        f"({'you are ahead ↑' if mat_diff > 50 else 'you are behind ↓' if mat_diff < -50 else 'roughly equal'})",
        "",
        "── KING SAFETY ───────────────────────────────────────────────",
        f"  Your king   : {my_king_safety}",
        f"  Enemy king  : {enemy_king_safety}",
        "",
        "── PAWN STRUCTURE ────────────────────────────────────────────",
    ]

    if my_pawn_issues:
        lines.append(f"  Your weaknesses : {', '.join(my_pawn_issues)}")
    else:
        lines.append("  Your pawn structure: healthy ✓")

    if my_passed:
        lines.append(f"  Your passed pawns   : {', '.join(my_passed)}  ← push toward promotion!")

    if en_pawn_issues:
        lines.append(f"  Enemy weaknesses: {', '.join(en_pawn_issues)}  ← target these!")

    if en_passed:
        lines.append(f"  Enemy passed pawns  : {', '.join(en_passed)}  ← blockade or capture!")

    lines += [
        "",
        "── BOARD CONTROL ─────────────────────────────────────────────",
        f"  Center (e4/d4/e5/d5): You {my_center} vs Enemy {enemy_center} pieces",
        f"  Your mobility       : {my_mobility} legal moves",
        "",
        "── RECENT MOVES ──────────────────────────────────────────────",
        f"  {move_history}",
        "",
    ]

    if tactics:
        lines.append("── TACTICAL ALERTS ───────────────────────────────────────────")
        for alert in tactics:
            lines.append(f"  {alert}")
        lines.append("")

    lines.append("── STRATEGIC GUIDELINES ──────────────────────────────────────")
    if phase == "OPENING":
        lines += [
            "  • Control the center with pawns (e4/d4 or e5/d5)",
            "  • Develop all minor pieces (knights before bishops)",
            "  • Castle early to safeguard your king",
            "  • Do NOT move the same piece twice without strong reason",
            "  • Do NOT bring the queen out prematurely",
        ]
    elif phase == "MIDDLEGAME":
        lines += [
            "  • Look for tactical shots: forks, pins, skewers, discoveries",
            "  • Attack enemy weaknesses identified in pawn structure",
            "  • Improve your worst-placed piece",
            "  • Create threats that force your opponent to react",
            "  • Keep your king safe — avoid weakening pawn moves near it",
        ]
    else:  # ENDGAME
        lines += [
            "  • Activate your king — it is a strong piece in the endgame",
            "  • Push your passed pawns aggressively toward promotion",
            "  • Simplify into a winning ending if you are material up",
            "  • Place rooks behind passed pawns (yours or enemy's)",
            "  • Avoid stalemate if you are winning — watch for traps",
        ]

    lines += [
        "",
        "── YOUR LEGAL MOVES ──────────────────────────────────────────",
        f"  {', '.join(legal_moves_san)}",
        "",
        "=" * 62,
    ]

    return "\n".join(lines)


# ── Appel Claude (joue les blancs) ────────────────────────────────────────────
def ask_claude(board):
    analysis = build_board_analysis(board, chess.WHITE)

    prompt = f"""You are a chess grandmaster. Your only goal is to WIN this game. You play as WHITE.

{analysis}

Decide your move using this priority order:
1. If there is a CHECKMATE IN ONE — play it immediately, no discussion.
2. If you can win significant material with a safe tactic — do it.
3. If your opponent has a HANGING piece — capture it if it's safe.
4. Otherwise choose the move that best improves your position (development, center control, king safety, pawn structure).

You MUST pick a move that appears in the legal moves list above.
Respond with ONLY this line (no explanation, no preamble):
MOVE: <move in SAN notation>"""

    t0 = time.time()
    response = claude_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    _record_think("claude", elapsed_ms)
    return response.content[0].text.strip()


# ── Appel GPT (joue les noirs) ────────────────────────────────────────────────
def ask_gpt(board):
    analysis = build_board_analysis(board, chess.BLACK)

    system_prompt = (
        "You are a chess grandmaster. Your only goal is to WIN every game. "
        "You always study the position deeply before choosing a move. "
        "You respond ONLY with: MOVE: <move in SAN notation> — nothing else."
    )

    user_prompt = f"""You play as BLACK. Study the position below and find the strongest move.

{analysis}

Decide your move using this priority order:
1. If there is a CHECKMATE IN ONE — play it immediately, no discussion.
2. If you can win significant material with a safe tactic — do it.
3. If your opponent has a HANGING piece — capture it if it's safe.
4. Otherwise choose the move that best improves your position (development, center control, king safety, pawn structure).

You MUST pick a move that appears in the legal moves list above.
Respond with ONLY this line (no explanation, no preamble):
MOVE: <move in SAN notation>"""

    t0 = time.time()
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=100,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    )
    elapsed_ms = int((time.time() - t0) * 1000)
    _record_think("gpt", elapsed_ms)
    return response.choices[0].message.content.strip()


# ── Valider et jouer un coup ──────────────────────────────────────────────────
def parse_move(board, raw):
    """Tente de parser un coup depuis une réponse libre."""
    text = raw.strip()
    if "MOVE:" in text.upper():
        text = text.upper().split("MOVE:")[-1].strip()
        text = text.split()[0] if text.split() else text

    clean = text.replace("?","").replace("!","").replace("+","").replace("#","").strip()

    # Essai SAN direct
    try:
        move = board.parse_san(clean)
        if move in board.legal_moves:
            return move
    except Exception:
        pass

    # Essai UCI (gère aussi les promotions : e7e8q)
    try:
        move = chess.Move.from_uci(clean.lower())
        if move in board.legal_moves:
            return move
    except Exception:
        pass

    # Correspondance partielle
    for legal_move in board.legal_moves:
        san = board.san(legal_move).replace("+","").replace("#","")
        if san.lower() == clean.lower() or legal_move.uci() == clean.lower():
            return legal_move

    return None

def play_move(board, move_san, max_retries=5):
    """Tente de jouer un coup, retente si illégal."""
    for attempt in range(max_retries):
        try:
            move = parse_move(board, move_san)
            if move and move in board.legal_moves:
                uci = move.uci()
                san = board.san(move)
                board.push(move)
                return san, uci[:2], uci[2:4]
        except Exception as e:
            print(f"  ⚠️  Erreur validation coup '{move_san}': {e}")

        print(f"  Coup illégal '{move_san}', nouvelle tentative {attempt+1}/{max_retries}...")
        try:
            if board.turn == chess.WHITE:
                move_san = ask_claude(board)
            else:
                move_san = ask_gpt(board)
        except Exception as e:
            print(f"  ❌ Erreur API lors du retry {attempt+1}: {e}")
            time.sleep(5)

    # Dernier recours : coup aléatoire légal
    import random
    move = random.choice(list(board.legal_moves))
    uci  = move.uci()
    san  = board.san(move)
    board.push(move)
    print(f"  ⚠️  Fallback aléatoire: {san}")
    return san, uci[:2], uci[2:4]


# ── Mettre à jour les stats journalières ──────────────────────────────────────
def update_daily_data():
    today = datetime.datetime.now().strftime("%d/%m")
    daily = state["daily_data"]
    if daily and daily[-1]["label"] == today:
        daily[-1]["count"] += 1
    else:
        daily.append({"label": today, "count": 1})
    state["daily_data"] = daily[-14:]


# ── Boucle principale ─────────────────────────────────────────────────────────
def game_loop():
    load_state()
    print(f"🚀 AI Chess Battle démarré — Partie #{state['game_number']}")

    while True:
        board = chess.Board()

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

            state["turn"] = "white" if is_white else "black"
            state["fen"]  = board.fen()
            save_state()

            print(f"  ♟  {player} analyse le plateau...")

            try:
                if is_white:
                    raw_move = ask_claude(board)
                else:
                    raw_move = ask_gpt(board)

                san, from_sq, to_sq = play_move(board, raw_move)
                print(f"  ✅ {player} joue : {san}")

                state["fen"]          = board.fen()
                state["last_move"]    = {"from": from_sq, "to": to_sq}
                state["total_moves"] += 1
                state["moves"].append({"san": san, "color": "white" if is_white else "black"})
                state["turn"] = "black" if is_white else "white"
                save_state()

            except Exception as e:
                print(f"  ❌ Erreur API : {e}")
                time.sleep(10)
                continue

            time.sleep(3)

        # ── Fin de partie ──────────────────────────────────────────────────
        result  = board.result()

        if result == "1-0":
            state["scores"]["claude"] += 1
            winner = "Claude 🎉"
        elif result == "0-1":
            state["scores"]["gpt"] += 1
            winner = "ChatGPT 🎉"
        else:
            state["scores"]["draws"] += 1
            winner = "Nulle 🤝"

        pgn_game = chess.pgn.Game.from_board(board)
        pgn_game.headers["Event"]  = f"Infinite AI Chess Battle - Game {state['game_number']}"
        pgn_game.headers["White"]  = "Claude (Anthropic)"
        pgn_game.headers["Black"]  = "ChatGPT (OpenAI)"
        pgn_game.headers["Date"]   = datetime.datetime.utcnow().strftime("%Y.%m.%d")
        pgn_game.headers["Result"] = result
        pgn_str    = str(pgn_game)
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

@app.route("/stats.html")
def stats_page():
    return app.send_static_file("stats.html")

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
    game_thread = threading.Thread(target=game_loop, daemon=True)
    game_thread.start()

    port = int(os.getenv("PORT", 5000))
    print(f"🌐 Serveur web sur http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
