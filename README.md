# backgammon

## TODO

 - [Implement ruleset](#ruleset)
 - [Implement turn-taking](#turn)
 - [AI opponent](#AI)
 - [Implement Endgame](#endgame)
 - [Implement win-checking](#win)
 - [GUI](#graphics)

### Ruleset
 - Player can stack pieces 
 - Player cannot place piece on tile occupied by >1 opposing pieces
 - Player can place piece on tile occupied by 1 opposing piece. Opposing piece removed
 - If piece has been removed, must move back onto board first. 
 - If piece cannot be moved back onto board, turn skipped
### Turn
 - Roll 2 dice
 - If dice show equal score, double the number of dice with this score
 - Each die can move a piece n pieces forward, n = score on die
 - White starts
### AI

### Endgame
 - Move pieces to home board + bear them off
 - Number rolled must equactly = distance to home board to be beared off
### Win
 - All pieces beared off
### Graphics

