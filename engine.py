import chess
import sys
import multiprocessing as mp
import concurrent.futures
from chessboard import display
import chess.engine

piece_tables = {chess.PAWN : 100, chess.KNIGHT : 320, chess.BISHOP : 330, chess.ROOK : 500, chess.QUEEN : 900,
                chess.KING : 0}
piece_square_tables = {
    chess.PAWN : [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ],
    chess.KNIGHT : [
        -50, -40, -20, -30, -30, -20, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 5, 15, 15, 5, 5, -30,
        -40, -20, 0, 5, 5, 0, -20,-40,
        -50, -40, -20, -30, -30, -20, -40, -50
    ],
    chess.BISHOP : [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -40, -10, -10, -40, -10, -20
    ],
    chess.ROOK : [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0
    ],
    chess.QUEEN : [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
    ],
    chess.KING:[-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30,
-30,-40,-40,-50,-50,-40,-40,-30,
-20,-30,-30,-40,-40,-30,-30,-20,
-10,-20,-20,-20,-20,-20,-20,-10,
 20, 20,  0,  0,  0,  0, 20, 20,
 20, 30, 10,  0,  0, 10, 30, 20]}
center_control_tables = {
    chess.A1 : 1,
    chess.A2 : 1,
    chess.A3 : 1,
    chess.A4 : 1,
    chess.A5 : 1,
    chess.A6 : 1,
    chess.A7 : 1,
    chess.A8 : 1,
    chess.B1 : 1,
    chess.B2 : 1,
    chess.B3 : 1,
    chess.B4 : 1,
    chess.B5 : 1,
    chess.B6 : 1,
    chess.B7 : 1,
    chess.B8 : 1,
    chess.C1 : 5,
    chess.C2 : 5,
    chess.C3 : 5,
    chess.C4 : 5,
    chess.C5 : 5,
    chess.C6 : 5,
    chess.C7 : 5,
    chess.C8 : 5,
    chess.D1 : 10,
    chess.D2 : 10,
    chess.D3 : 3,
    chess.D4 : 10,
    chess.D5 : 10,
    chess.D6 : 3,
    chess.D7 : 10,
    chess.D8 : 10,
    chess.E1 : 10,
    chess.E2 : 10,
    chess.E3 : 3,
    chess.E4 : 10,
    chess.E5 : 10,
    chess.E6 : 3,
    chess.E7 : 10,
    chess.E8 : 10,
    chess.F1 : 5,
    chess.F2 : 5,
    chess.F3 : 5,
    chess.F4 : 5,
    chess.F5 : 5,
    chess.F6 : 4,
    chess.F7 : 5,
    chess.F8 : 5,
    chess.G1 : 1,
    chess.G2 : 1,
    chess.G3 : 1,
    chess.G4 : 1,
    chess.G5 : 1,
    chess.G6 : 1,
    chess.G7 : 1,
    chess.G8 : 1,
    chess.H1 : 1,
    chess.H2 : 1,
    chess.H3 : 1,
    chess.H4 : 1,
    chess.H5 : 1,
    chess.H6 : 1,
    chess.H7 : 1,
    chess.H8 : 1,
}

pawn_structure_tables = {
    chess.A1: -10, chess.A2: -10, chess.A3: 0, chess.A4: 0, chess.A5: 5, chess.A6: 0, chess.A7: -10, chess.A8: -10,
    chess.B1: -5, chess.B2: -5, chess.B3: 0, chess.B4: 3, chess.B5: 5, chess.B6: 0, chess.B7: -5, chess.B8: -5,
    chess.C1: 0, chess.C2: 0, chess.C3: 0, chess.C4: 4, chess.C5: 6, chess.C6: 0, chess.C7: 0, chess.C8: 0,
    chess.D1: 5, chess.D2: 5, chess.D3: 0, chess.D4: 5, chess.D5: 7, chess.D6: 0, chess.D7: 5, chess.D8: 5,
    chess.E1: 5, chess.E2: 5, chess.E3: 0, chess.E4: 5, chess.E5: 7, chess.E6: 0, chess.E7: 5, chess.E8: 5,
    chess.F1: 0, chess.F2: 0, chess.F3: 0, chess.F4: 4, chess.F5: 6, chess.F6: 0, chess.F7: 0, chess.F8: 0,
    chess.G1: -5, chess.G2: -5, chess.G3: 0, chess.G4: 3, chess.G5: 5, chess.G6: 0, chess.G7: -5, chess.G8: -5,
    chess.H1: -10, chess.H2: -10, chess.H3: 0, chess.H4: 0, chess.H5: 5, chess.H6: 0, chess.H7: -10, chess.H8: -10,
}
king_safety_tables = {
    chess.A1: 0, chess.B1: 0, chess.C1: 0, chess.D1: 1,
    chess.E1: 2, chess.F1: 0, chess.G1: 0, chess.H1: 0,
    chess.A2: 1, chess.B2: 1, chess.C2: 1, chess.D2: 1,
    chess.E2: 1, chess.F2: 1, chess.G2: 1, chess.H2: 1,
    chess.A3: 2, chess.B3: 2, chess.C3: 2, chess.D3: 2,
    chess.E3: 2, chess.F3: 2, chess.G3: 2, chess.H3: 2,
    chess.A4: 3, chess.B4: 3, chess.C4: 3, chess.D4: 3,
    chess.E4: 3, chess.F4: 3, chess.G4: 3, chess.H4: 3,
    chess.A5: 4, chess.B5: 4, chess.C5: 4, chess.D5: 4,
    chess.E5: 4, chess.F5: 4, chess.G5: 4, chess.H5: 4,
    chess.A6: 5, chess.B6: 5, chess.C6: 5, chess.D6: 5,
    chess.E6: 5, chess.F6: 5, chess.G6: 5, chess.H6: 5,
    chess.A7: 6, chess.B7: 6, chess.C7: 6, chess.D7: 6,
    chess.E7: 6, chess.F7: 6, chess.G7: 6, chess.H7: 6,
    chess.A8: 7, chess.B8: 7, chess.C8: 7, chess.D8: 8,
    chess.E8: 9, chess.F8: 7, chess.G8: 7, chess.H8: 7,
}
endgame_tables = {
    chess.A1: 0, chess.B1: 0, chess.C1: 0, chess.D1: 0, chess.E1: 0, chess.F1: 0, chess.G1: 0, chess.H1: 0,
    chess.A2: 5, chess.B2: 10, chess.C2: 10, chess.D2: 10, chess.E2: 10, chess.F2: 10, chess.G2: 10, chess.H2: 5,
    chess.A3: 4, chess.B3: 8, chess.C3: 8, chess.D3: 8, chess.E3: 8, chess.F3: 8, chess.G3: 8, chess.H3: 4,
    chess.A4: 3, chess.B4: 6, chess.C4: 6, chess.D4: 6, chess.E4: 6, chess.F4: 6, chess.G4: 6, chess.H4: 3,
    chess.A5: 2, chess.B5: 4, chess.C5: 4, chess.D5: 4, chess.E5: 4, chess.F5: 4, chess.G5: 4, chess.H5: 2,
    chess.A6: 1, chess.B6: 2, chess.C6: 2, chess.D6: 2, chess.E6: 2, chess.F6: 2, chess.G6: 2, chess.H6: 1,
    chess.A7: 0, chess.B7: 0, chess.C7: 0, chess.D7: 0, chess.E7: 0, chess.F7: 0, chess.G7: 0, chess.H7: 0,
    chess.A8: 0, chess.B8: 0, chess.C8: 0, chess.D8: 0, chess.E8: 0, chess.F8: 0, chess.G8: 0, chess.H8: 0,
}
weight_pawn_structure = 1.0
weight_king_safety = 1.0
weight_mobility = 0.5
weight_center_control = 0.5
weight_endgame = 1.0


class NullWindow :
    def __len__(self) :
        return 0

    def __getitem__(self, item) :
        raise IndexError("NullWindow has no elements")


def evaluate(position):
    if position is None:
        return 0
    if position.is_game_over():
        if position.result() == "1-0":
            return 999999
        elif position.result() == "0-1":
            return -999999
        else:
            return 0

    total_evaluation = 0

    # Material score
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        own_pieces = position.pieces(piece_type, position.turn)
        opponent_pieces = position.pieces(piece_type, not position.turn)
        total_evaluation += (len(own_pieces) - len(opponent_pieces)) * piece_tables[piece_type]

    # King safety score
    own_king_square = position.king(position.turn)
    own_king_safety_score = king_safety_tables[own_king_square]
    opponent_king_square = position.king(not position.turn)
    opponent_king_safety_score = king_safety_tables[opponent_king_square]
    king_safety_score = (own_king_safety_score - opponent_king_safety_score) * weight_king_safety / 10
    total_evaluation += king_safety_score

    # Pawn structure score
    own_pawns = position.pieces(chess.PAWN, position.turn)
    own_pawn_files = [chess.square_file(square) for square in own_pawns]
    pawn_structure_score = sum([pawn_structure_tables[square] for square in own_pawn_files])
    total_evaluation += pawn_structure_score * weight_pawn_structure / 10

    # Mobility score
    own_legal_moves = len(list(position.legal_moves))
    opponent_color = not position.turn
    opponent_legal_moves = len(list(position.copy().mirror().legal_moves))
    mobility_score = (own_legal_moves - opponent_legal_moves) * weight_mobility / 10
    total_evaluation += mobility_score

    # Center control score
    own_center_control = sum([1 for square in center_control_tables if position.attackers(position.turn, square)])
    opponent_center_control = sum([1 for square in center_control_tables if position.attackers(not position.turn, square)])
    center_control_score = (center_control_tables[own_center_control] - center_control_tables[opponent_center_control]) * weight_center_control / 10
    total_evaluation += center_control_score

    # Endgame score
    endgame_score = 0
    endgame_threshold = 10
    if abs(total_evaluation) <= endgame_threshold:
        own_pawns = position.pieces(chess.PAWN, position.turn)
        opponent_pawns = position.pieces(chess.PAWN, not position.turn)
        if len(own_pawns) <= 1 and len(opponent_pawns) <= 1:
            endgame_score = (endgame_tables[own_center_control] - endgame_tables[opponent_center_control]) * weight_endgame / 10
    total_evaluation += endgame_score

    return total_evaluation

def iterative_deepening_search(position, max_depth=4):
    best_move = None
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for depth in range(1, max_depth+1):
            future = executor.submit(minimax_search, position, depth)
            try:
                best_move = future.result(timeout=1.0)
            except concurrent.futures.TimeoutError:
                pass
    return best_move

def minimax_search(position, depth, maximizing_player=True):
    if depth == 0 or position.is_terminal():
        return position.evaluate(), None

    best_score = float('-inf') if maximizing_player else float('inf')
    best_move = None

    for move in position.get_legal_moves():
        position.make_move(move)
        score, _ = minimax_search(position, depth - 1, not maximizing_player)
        position.undo_move(move)

        if maximizing_player and score > best_score:
            best_score = score
            best_move = move
        elif not maximizing_player and score < best_score:
            best_score = score
            best_move = move

    return best_score, best_move

def move_ordering(position) :
    """Returns a list of legal moves in order of increasing value"""
    legal_moves = list(position.legal_moves)
    legal_moves.sort(key=lambda move : evaluate(position.copy().push(move)))
    return legal_moves

def alphabeta(position, depth, alpha= float('-inf'), beta= float('inf'), null_window=1, use_lmr=True, use_null=True, use_parallel=True, pool=None):
    """Returns [eval, best move] for the position at the given depth"""
    if depth == 0 or position.is_game_over():
        return [evaluate(position), None]

    legal_moves = move_ordering(position)

    best_score = -float('inf')
    best_move = None

    if use_null and depth > 2 and not position.is_check():
        position.push(chess.Move.null())
        score = -alphabeta(position, depth - 3, -beta, -beta + 1, null_window, False, False, False, None)[0]
        position.pop()
        if score >= beta:
            return [beta, None]

    if use_lmr and depth > 1 and not position.is_check() and best_move is not None and not position.is_capture(best_move) :
        reduction = 1
        if len(legal_moves) >= 10:
            reduction = 2
        for i, move in enumerate(legal_moves):
            if i < 3:
                score, _ = alphabeta(position.copy().push(move), depth - 1 - reduction, -beta, -alpha, None, False, False, False, None)
            else:
                score, _ = alphabeta(position.copy().push(move), depth - 1 - reduction, -alpha - 1, -alpha, None, False, False, False, None)
                if score > alpha:
                    score, _ = alphabeta(position.copy().push(move), depth - 1, -beta, -alpha, None, False, False, False, None)
            score = -score
            if score >= beta:
                return [beta, None]
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

    else:
        if use_parallel and depth > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
                results = [
                    pool.submit(alphabeta, position.copy().push(move), depth - 1, -beta, -alpha, null_window, use_lmr,
                                use_null, use_parallel, None) for move in legal_moves]
                for result in concurrent.futures.as_completed(results):
                    score, move_ = result.result()
                    score = -score
                    if score > alpha:  # player maximizes his score
                        alpha = score
                        best_move = move_
                        if alpha >= beta:  # alpha-beta cutoff
                            break

            best_score = alpha
        else:
            for move in legal_moves:
                score, _ = alphabeta(position.copy().push(move), depth - 1, -beta, -alpha, None, False, False, False, None)
                score = -score
                if score >= beta:
                    return [beta, None]
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)

    return [best_score, best_move]

fen_ = input('Enter fen: ')
board = chess.Board(fen_)
_depth = int(input('Enter depth: '))

while True:
    game_board = display.start()
    if not board.is_game_over():
            display.update(board.fen(), game_board)
            x = {True : "White's turn", False : "Black's turn"}
            move = input('Enter move:')
            board.push_san(str(move))
            engine = alphabeta(board, _depth)
            board.push(engine[1])
            print(f"{board}\n", f"Evaluation: {-engine[0]/100}", f"Best move: {engine[1]}", f"Fen: {board.fen()}", sep='\n')
            display.update(board.fen(), game_board)
            display.check_for_quit()
    else:
        print(f'Game over\nResult: {board.result()}')
        break
sys.exit()
