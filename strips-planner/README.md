# NOTE: This code is initially forked from https://github.com/tansey/strips
This code is initially forked from [https://github.com/tansey/strips](https://github.com/tansey/strips)

# Stanford Research Institute Problem Solver (STRIPS)
STRIPS is a First-Order Logic (FOL) language with an associated linear (total order) solver. Note that linear solvers have a few well-known issues, such as not dealing well with [http://en.wikipedia.org/wiki/Sussman_Anomaly](Sussman's Anomaly). Nevertheless, when learning planning algorithms, most students learn STRIPS first.

## Problem Description
The code implements the STRIPS problem description language. The first line is the initial state of the world:

    Initial state: At(A), Level(low), BoxAt(C), BananasAt(B)

Each initial state argument must be a literal. Currently, starting literals with upper case is allowed but may be changed in the future. It's advised that you follow the convention of literals being all lowercase and variables beginning with an upper case letter.

The second line describes the goal state you want the world to be in:

    Goal state:    Have(Bananas)

Again, only literals are allowed at the moment. This may change in the future.

The third line should declare the start of your action methods:

    Actions:

Each action has three lines: a declaration, then a set of preconditions, and finally a set of postconditions. The declaration line specifies the name of the method and its parameters; all action parameters are variables:

    // move from X to Y
    Move(X, Y)
    Preconditions:  At(X), Level(low)
    Postconditions: !At(X), At(Y)

The preconditions line contains states that the world must be in before the action can be called. In contrast, the postconditions line describes the effect of calling the action. You are able to mix variables and literals in the argument types for both preconditions and postconditions.

Note that the app follows a closed-world assumption; that is, any facts about the world not declared in the initial state are considered to be false. The parser is clever enough that you don't need to declare all your literals in the initial state, as long as you use them somewhere in the problem description. You can declare negations by using an exclamation mark (!) immediately before the name of a condition.

## Linear Solver
The STRIPS planner is a simple linear solver. In planning literature, this is sometimes called a total order solver since it assumes all steps can be planned one after the other with a strict ordering. It works backwards by starting at the goal state and searching for a grounded action (one with all variables replaced with literals) that will satisfy a precondition of the goal state. When it finds an acceptable action, it adds it to the plan and the newly added action becomes the new goal state. If an action is added that conflicts with another already-satisfied goal, then the clobbered goal is reintroduced and resatisfied. The algorithm keeps working backwards, stacking subgoals on top of each other via a depth-first search until it either satisfies all the goals or exhausts all possible plans and gives up.

## BTAI: Running the Example 

To run the example problem, simply execute the following:

    python3 strips.py test_strips.txt
    
Then test it on one of the five images (includes a mix of complicated and simple images).

When choosing a file to run, here is an example of how to interpret the file's name:
c0b1m1b0.txt corresponds to the Kaggle image file titled canvas_0_banana_1_monkey_1_box_0.png. 

Note that for notation like LeftOf(A, B), that means Box B is to the left of Box B. Left, right, and other directons are determined through the perspective of the monkey. So, if Box A is to to the left on my screen, it's to the right for the monkey in the image. Hence, in our txt files, we say Box A is to the right.







