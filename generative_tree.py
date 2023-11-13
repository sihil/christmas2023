from typing import Optional
from xml.etree import ElementTree
import random

import py5


def combine_svgs(layers, new_svg):
    py5.begin_record(py5.SVG, new_svg)
    py5.end_record()
    ElementTree.register_namespace('', 'http://www.w3.org/2000/svg')
    tree = ElementTree.parse(new_svg)
    combined = tree.getroot()
    combined.set('xmlns:inkscape',
                 'http://www.inkscape.org/namespaces/inkscape')

    for svg in layers:
        markup = ElementTree.parse(svg).getroot()
        group = ElementTree.SubElement(combined, 'g')
        group.set('id', svg)
        group.set('inkscape:groupmode', 'bar')
        group.set('inkscape:groupmode', 'layer')
        group.set('inkscape:label', svg)

        for child in list(markup):
            group.append(child)

    ElementTree.indent(tree, space="\t", level=0)
    tree.write(new_svg)


# draw a branch of a tree
def draw_branch(branch_x, branch_y, length, angle_degrees, needle_density=5, staggering=1.0):
    # convert angle to radians
    branch_angle = py5.radians(angle_degrees)
    print("draw_branch", branch_x, branch_y, length, branch_angle)
    # draw the branch
    py5.line(branch_x, branch_y, branch_x + length * py5.cos(branch_angle), branch_y + length * py5.sin(branch_angle))

    # now draw a series of needles along the branch

    # how many needles will there be?
    needle_count = length // needle_density
    needle_count_first_side = needle_count // 2 + needle_count % 2
    needle_count_second_side = needle_count - needle_count_first_side
    print(f"needle_count {needle_count} 1st={needle_count_first_side} 2nd={needle_count_second_side}")

    # what portion of branch will there be before the first needle?
    # how long is a needle likely to be
    needle_length = length / 6
    first_needle_distance = py5.random(0, length/4)
    last_needle_distance = length - py5.random(0, needle_length)

    # make the list of needle start points
    # if stagger is 1, they will alternate evenly between left and right
    # if stagger is 0, the left and right will be at the same point
    def start_distance(i) -> float:
        first_side = i % 2 == 0
        odd_needle_count = needle_count % 2 == 1
        side_i = i // 2
        if odd_needle_count:
            first_side_interval = (last_needle_distance - first_needle_distance) / needle_count_first_side
            if first_side:
                # just distribute it evenly along the branch
                return first_needle_distance + first_side_interval * side_i
            else:
                # distribute it at an offset from where the first side is
                prev_needle = start_distance(i - 1)
                return prev_needle + first_side_interval * staggering / 2
        else:
            # even number of needles, so we can just distribute evenly taking staggering into account
            first_side_interval = (
                    (last_needle_distance - first_needle_distance) / needle_count_first_side + staggering * 0.5
            )
            if first_side:
                return first_needle_distance + first_side_interval * side_i
            else:
                prev_needle = start_distance(i - 1)
                return prev_needle + first_side_interval * staggering / 2

    needle_starts_all = [start_distance(i) for i in range(needle_count)]
    # filter this list to occasionally remove one of the needles
    # this will make the tree look more natural
    needle_starts = [x for x in needle_starts_all]  # if py5.random(0, 30) % 30 == 0]
    # (last_needle_distance - first_needle_distance) / needle_count

    print(f'needle_starts {["{0:0.2f}".format(i) for i in needle_starts]}')

    # needle angles
    left_angle = py5.radians(angle_degrees - random.randint(30, 50))
    right_angle = py5.radians(angle_degrees + random.randint(30, 50))

    # decide if we're starting  on the left or right side
    left = bool(py5.random_int(0, 1))

    for needle_start_distance in needle_starts:
        # decide if we're on the left or right side
        needle_angle_rads = left_angle if left else right_angle

        print(f"needle_start_distance={needle_start_distance:.2f} left={left} branch_angle={branch_angle:.2f} needle_angle_rads={needle_angle_rads:.2f}")

        # x,y coords of needle start
        needle_x = branch_x + needle_start_distance * py5.cos(branch_angle)
        needle_y = branch_y + needle_start_distance * py5.sin(branch_angle)

        print()

        # draw the needle
        print(needle_x, needle_y, needle_x + needle_length * py5.cos(needle_angle_rads), needle_y + needle_length * py5.sin(needle_angle_rads))
        py5.line(needle_x, needle_y, needle_x + needle_length * py5.cos(needle_angle_rads), needle_y + needle_length * py5.sin(needle_angle_rads))

        # now set up for next needle to be on other side
        left = not left


def settings():
    py5.size(300, 300)


def setup():
    py5.random_seed(123456)


def draw():
    py5.begin_record(py5.SVG, 'tree.svg')
    draw_branch(150, 150, 100, 10, needle_density=16, staggering=0)
    draw_branch(150, 150, 100, 50, needle_density=12, staggering=0)
    draw_branch(150, 150, 100, 90, needle_density=8, staggering=0)
    draw_branch(150, 150, 100, 130, needle_density=16, staggering=0.5)
    draw_branch(150, 150, 100, 170, needle_density=12, staggering=0.5)
    draw_branch(150, 150, 100, 210, needle_density=8, staggering=0.5)
    draw_branch(150, 150, 100, 250, needle_density=16, staggering=1)
    draw_branch(150, 150, 100, 290, needle_density=12, staggering=1)
    draw_branch(150, 150, 100, 320, needle_density=8, staggering=1)
    py5.end_record()

    py5.exit_sketch()


py5.run_sketch()
