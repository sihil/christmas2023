Christmas card for 2023
=======================

This repo defines a generative Christmas card for 2023 for plotting on an iDraw v2.0 plotter (mine is A3).

Getting started
---------------

1. Install the homebrew dependencies, from the root of the repo:
    ```bash
    brew bundle
    ```

2. Make sure you have a python environment:
    ```bash
    pyenv install 3.11.6
    pyenv virtualenv 3.11.6 christmas2023
    ```

3. Install the python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Check that you can generate a single card design:
    ```bash
    python generative_tree.py
    ```

5. That should have generated files under output/layers and output/post-processing. You can now plot the files in output/post-processing on your plotter.

6. If you want to generate a 2-up (A4) or 4-up (A3) set of cards, you can run:
    ```bash
    python multicard.py
    ```
   
Plotting
--------

The `multicard.py` script generates a set of files in `output/grid` which are designed to be plotted with an iDraw plotter.

As well as a multilayer SVG file it also generates a set of iDraw v2 specific gcode files. These can be sent directly to
the plotter using the gcode-cli tool.

The design is intended to be plotted on dark card (I used a dark green). The layers should be plotted as follows:
  - Layer 1: Use a gold pen for this, it's a cockerel from a church local to me and "merry christmas"; I'm using a gold metallic gelly roll pen.
  - Layer 2: This is the main tree and a little credit. I'm using a 0.8 white gelly roll pen.
  - Layer 3: This is the baubles on the tree. You can use any colour you like, I'm using a red glitter gelly roll pen, although I've found that plotting this in white first and then going over it with the red glitter pen gives a better result.
