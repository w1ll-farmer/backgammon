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
 - Random Agent to test Robustness of engine
 - Genetic Alg
 - Greedy Alg
 - ACO (?)
 - ML
 - RL
 - Adaptive
 
### Endgame
 - Move pieces to home board + bear them off
 - Number rolled must equactly = distance to home board to be beared off
### Win
 - All pieces beared off
### Graphics
 - In backend player (opponent) will be player -1 and AI will be player 1
 - User gets to choose their colour at the start
 - Die is rolled on-screen to determ
