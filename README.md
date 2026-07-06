Run a Python Environment on a command line using __START.txt as reference!
Always make sure to update pip modules.
=============================================================================
run_live.py:
Runs algorithm through a connected webcam's feed. Checks for face & guesses age every frame.
Run with "--mode stable" for a single guess per face detection as intended. Spacebar restarts the process.
Press D to open debug.

run_sweep.py:
Runs algorithm through every image in an "input" directory.
The same process will be applied to each one, outputting the obtained results as new images on the "output" directory.
This method provides benchmark statistics based on your system!
Also outputs results as a .csv in the "output" dir for further analysis.
===============================================================================