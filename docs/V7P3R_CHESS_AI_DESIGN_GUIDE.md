# v7p3r Chess Machine Learning AI Design

This document serves as a living design and brainstorming document outlining a next generation v7p3r chess AI and its machine learning functional goals and expectations.

## Engine Configuration
A config.json file is used to configure the v7p3r chess AI engine. This file contains various settings that control the behavior and performance of the AI.


### Configuration Options
* Engine ID: the coded id name of the current v7p3r engine instance
* Core Engine Name: v7p3r, stockfish, chatfish, or any other engine name can be d (code can be updated to add specific engine handlers)
* Engine Version: the version number of the engine

## Scoring Configuration
* Checkmate Threats: the ai is heavily rewarded for finding checkmate moves and massively penalized for getting checkmated
* Stalemate Threats: the ai is moderately penalized for finding stalemate moves
* Draw Prevention: the ai is moderately penalized for finding drawing moves
* Game Phase: the ai is slightly rewarded for finding moves that are appropriate for the current game phase
  * Opening: the ai is moderately rewarded for finding moves that develop pieces off their original squares and slightly penalized for moving the same piece twice.
  * Middlegame: the ai is slightly rewarded for finding moves that defend its own pieces, rewarded moderately for finding moves that attack the opponent's weak pieces, slightly penalized for leaving pieces undefended, and heavily penalized for moves that expose the king.
  * Endgame: the ai is slightly rewarded for moves that place the king on more active squares, moderately rewarded for moves that advance pawns towards promotion, slightly penalized for moves that leave the king undefended, and heavily penalized for allowing the opponent to promote pawns.
* Check: the ai is heavily rewarded for finding moves that give check and heavily penalized for being placed in check
* Capture: the ai is moderately rewarded for finding captures that are safe and profitable, slightly penalized for captures that are not safe, and heavily penalized for captures that are not profitable
* Attack: the ai is slightly rewarded for finding attacking moves that create threats against the opponent's pieces, moderately rewarded for finding moves that lead to a direct attack on the opponent's king, and heavily penalized for moves that leave its own pieces vulnerable to attack.
* Material Count: the ai is slightly rewarded for having a higher raw piece count on the board than the opponent and slightly penalized for having less pieces than the opponent.
* Material Score: based on piece values regardless of raw count, the ai is moderately rewarded for having a higher piece value total than the opponent and moderately penalized for having a lower piece value total than the opponent.
* Pawn Positioning: the ai is slightly rewarded for pawn structures that protect the king and protect other pawns and slightly penalized for doubled pawns and isolated pawns.
* Knight Positioning: the ai is slightly rewarded for finding moves that place the knights on active squares (squares with 5 or more legal moves) and moderately penalized for moves that place the knights on less active squares (squares on the edge of the board)
* Bishop Positioning: the ai is moderately rewarded for finding moves that place the bishops on squares with better vision (the longer diagonals, with higher attacking counts), slightly penalized for having less than 2 bishops (poor color coverage), and moderately penalized for blocking the bishops vision with another piece (such as advancing a pawn and blocking in a bishop from developing)
* Rook Positioning: the ai is slightly rewarded for having the rooks on the opponents second rank, moderately rewarded for having the rooks on the same rank or file, and slightly penalized for having its own pieces blocking the two rooks from "seeing" eachother
* Queen Positioning: the ai is slightly rewarded for having the queen on an active square (a square with 5 or more legal moves), moderately penalized for having the queen on a square that can be easily attacked by a defended piece of the opponents, massively penalized for losing the queen in a non-profitable trade.
* King Positioning: the ai is moderately rewarded for keeping the king behind other pawns, slightly penalized for having the king on the back rank with no rook or queen on the same rank, moderately penalized for moving the king too much in the opening or middlegame, and heavily penalized for moves that expose the king on all sides.
* Castling: the ai is slightly rewarded for preventing the opponent from castling, moderately rewarded for protecting castling rights, heavily rewarded for castling, and moderately penalized for moves that give up castling rights (that are not the act of castling)
* En Passant: the ai is slightly rewarded for capturing an opponent via the en passant rule and slightly penalized for allowing capture by the opponent via en passant.
* Promotion: the ai is slightly rewarded for having a pawn on the 7th rank, heavily rewarded for promoting a pawn to a queen, and moderately penalized for allowing a 7th rank pawn to be captured or promoted pawn to be captured immediately following promotion to a queen.