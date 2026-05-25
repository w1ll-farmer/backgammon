 # Backgammon

 ## Deep Agent Dataset Information
  - The dataset for the Deep Agent is yet to be formally annotated - this is coming soon
  - The network takes 2 289-dimension vectors as input. These vectors each represent a possible board state. The network has completely identical weights for both sides of the network between the input and hidden layer. Then, the weights between the hidden and output layer are equal to the other side of the network multiplied by -1. This ensures that if the boards were input the other way around, the result would be the same
  - The boards are encoded differently for the different board positions, but generally: 
    - Each point on the board is encoded into a 10-dimensional binary vector with switches if a point is occupied by > 4 black pieces, 4 black pieces, 3 black pieces, > 1 black piece, 1 black piece, 1 white piece, > 1 white piece, 3 white pieces, 4 white pieces, > 4 white pieces
    - The probability that a blot occupying a point can get hit off is pre-calculated and added as input for each point
    - The pip count for each player is encoded as a 7-bit binary string (i.e. if black's pip count is 17, it is represented in binary as 0010001)
    - The probability of getting past the furthest-forward blockade for each player is precalculated and added as input
    - The existence of a prime for each player is included
    - The number of walls in each player's home is included
    - The number of player's walls in the opponent home is included
    - The number of the player's pieces in the opponent's home is included
    - The number of borne-off pieces is included for racing board positions and for the doubling cube decision
 
