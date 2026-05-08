### **V7P3RAI v3.0 ChessState Data Model**

This model represents a single "frame" of the game, providing all the objective metadata for the AI's "Thinking Brain" (RNN) to process.

| Field Name | Data Type | Description |
| :---- | :---- | :---- |
| Board\_Features | Dictionary/Object | A collection of global, board-wide metrics. |
| Board\_Features.sideToMove | Integer (0, 1\) | The side to move (White=0, Black=1). |
| Board\_Features.castlingRights | Array of Booleans | \[White\_KS, White\_QS, Black\_KS, Black\_QS\] |
| Board\_Features.enPassantSquare | Integer (0-63) or null | The en passant target square, if available. |
| Board\_Features.halfmoveClock | Integer | The number of halfmoves since the last pawn move or capture. |
| Board\_Features.fullmoveNumber | Integer | The current game number. |
| Board\_Features.isKingInCheck | Boolean | Is the current player's king in check? |
| Board\_Features.threatCounts | Object | A breakdown of threats and control. |
| Board\_Features.threatCounts.whiteAttacked | Integer | Total squares attacked by white. |
| Board\_Features.threatCounts.blackAttacked | Integer | Total squares attacked by black. |
| Board\_Features.materialBalance | Object | The raw material values on the board. |
| Board\_Features.materialBalance.whiteTotal | Integer | Sum of material values for white. |
| Board\_Features.materialBalance.blackTotal | Integer | Sum of material values for black. |
| Board\_Features.materialBalance.difference | Integer | White total minus black total. |
| Board\_Features.pieceCounts | Object | Count of each piece type for each side. |
| Board\_Features.pawnStructure | Object | Metrics on pawn positions. |
| Pieces | Array of Dictionaries | A list of features for each of the 32 pieces on the board. |
| Pieces\[i\].type | Integer (1-6) | The piece type. |
| Pieces\[i\].color | Integer (0, 1\) | The piece color. |
| Pieces\[i\].currentSquare | Integer (0-63) | The piece's current position. |
| Pieces\[i\].mobility | Object | All mobility metrics for the piece. |
| Pieces\[i\].mobility.legalMoves | Integer | Number of legal moves from this square. |
| Pieces\[i\].mobility.attackingMoves | Integer | Number of moves to capture a piece. |
| Pieces\[i\].mobility.maxUnobstructed | Integer | Max possible moves on an empty board. |
| Pieces\[i\].relationships | Object | Metrics on attackers/defenders. |
| Pieces\[i\].relationships.attackers | Array of Integers | Array of pieces attacking this piece. |
| Pieces\[i\].relationships.defenders | Array of Integers | Array of pieces defending this piece. |
| Pieces\[i\].relationships.forkPotential | Boolean | Can this piece fork on the next move? |

