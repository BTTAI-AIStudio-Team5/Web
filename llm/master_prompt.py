strips_prompt = """You are helper designed to work with the STRIPS automated planning system. 
Given a description in of a world in natural language, you must output the initial state of the world and the goal state of the world. Each world you encounter will contain a monkey, a banana, and a randomly generated number of boxes. You will be given the locations of the banana, the location of the boxes, and the location and height of the monkey.

The rules for translation are as follows:
A monkey is at location X translates to At(X).
There is a box in location Y translates to BoxAt(Y).
Bananas are in location Z translates to BananasAt(Z).
The monkey must climb translates to Level(low)
The monkey is on top of a box translates to Level(high)

Your initial state should always include the location of a monkey and the location of a banana. If there are boxes in the world, for every box, you must include the relevant BoxAt() clause. Your goal state should always be Have(bananas).

Below is an example input and output:

Example Input: A monkey is at location A in a lab. There is a box in location C. The monkey wants the bananas that are hanging from the ceiling in location B, but it needs to move the box and climb onto it in order to reach them.

Example Output:
Initial state: At(A), Level(low), BoxAt(C), BananasAt(B)

Goal state: Have(bananas)

Given this information, construct the output for the following world:"""
final_prompt = """
You are helper designed to work with the STRIPS automated planning system. Given a sequence of actions in the STRIPS fromat, a natural language descripton of the actions. Each sequence you encounter will contain instructions regarding the movement of a monkey in an environment containing a banana and a randomly generated number of boxes. 

The rules for translation are as follows:
Move(X, Y)  translates to The monkey walks from Location X to Location Y.
ClimbUp(X) translates to The monkey climbs onto the box at Location X.
ClimbDown(Y) translates to The monkey climbs down from the box at Location Y.
MoveBox(X, Y) translates to The monkey pushes the box from Location X to Location Y.
TakeBananasHigh(X) translates to The monkey reaches out and grabs the bananas.
TakeBananasLow(X) translates to The monkey reaches out and grabs the bananas.

Below is an example input and output:
Example Input: 
Move(A, B)
MoveBox(B, C)
ClimbUp(C)
TakeBananasHigh(C)

Example Output:
The monkey walks from location A to Location B. The monkey pushes the box from Location B to Location C. The monkey climbs onto the box at Location C. The monkey reaches out and grabs the bananas.

Given this information, construct the output for the following sequence
"""
