from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple, Dict
from xml.etree import ElementTree
import random

import py5


Instruction = Tuple[Callable[..., ...], List, Dict]


class Board:
    def __init__(self):
        self._instructions: List[Instruction] = []

    def d(self, func: Callable[..., ...], *args, **kwargs):
        self._instructions.append((func, list(args), kwargs))

    @property
    def instructions(self) -> List[Instruction]:
        return self._instructions

    def print_instructions(self):
        for instruction in self._instructions:
            print(instruction)


@dataclass
class Curve:
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float

    def as_list(self):
        return [self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4]

    def point_at(self, t) -> Tuple[float, float]:
        x = py5.curve_point(self.x1, self.x2, self.x3, self.x4, t)
        y = py5.curve_point(self.y1, self.y2, self.y3, self.y4, t)
        return x, y


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
def draw_branch(board: Board, branch_x, branch_y, length, angle_degrees, needle_density: Optional[float] = None,
                staggering: Optional[float] = None):
    if needle_density is None:
        needle_density = py5.random(4.5, 6.0)
    if staggering is None:
        staggering = py5.random(0.1, 0.9)

    # convert angle to radians
    branch_angle = py5.radians(angle_degrees)
    print("draw_branch", branch_x, branch_y, length, branch_angle)

    # draw the branch

    # draw a branch as a slight curve
    # how much of a curve do we want?
    # 0.0 = straight, 1.2 = very curvy
    angle_with_zero_top = (angle_degrees + 90) % 360
    degrees_from_up = angle_with_zero_top if angle_with_zero_top < 180 else 360 - angle_with_zero_top
    max_curvy = 0.6 * (degrees_from_up / 180) + 0.6
    curvy_amount = py5.random(0.05, max_curvy)

    # figure out the direction of curve
    # typically branches on the left (angle_with_zero_top 180-360) will curve to the right and vice versa
    # curving to the right will happen with a positive curvy_amount
    if angle_with_zero_top < 180:
        curvy_amount = -curvy_amount

    curve = Curve(
        branch_x - length * py5.cos(branch_angle-curvy_amount),
        branch_y - length * py5.sin(branch_angle-curvy_amount),
        branch_x,
        branch_y,
        branch_x + length * py5.cos(branch_angle),
        branch_y + length * py5.sin(branch_angle),
        branch_x + 2 * length * py5.cos(branch_angle+curvy_amount),
        branch_y + 2 * length * py5.sin(branch_angle+curvy_amount)
    )

    board.d(py5.curve,
            *curve.as_list(),
            notes=f'curviness: {curvy_amount:.2f} angle: {angle_degrees:.2f} angle_with_zero_top: {angle_with_zero_top:.2f} degrees_from_up: {degrees_from_up:.2f}')

    # now draw a series of needles along the branch

    # how many needles will there be?
    needle_count = int(length // needle_density)
    needle_count_first_side = needle_count // 2 + needle_count % 2
    needle_count_second_side = needle_count - needle_count_first_side
    print(f"needle_count {needle_count} 1st={needle_count_first_side} 2nd={needle_count_second_side}")

    # what portion of branch will there be before the first needle?
    # how long is a needle likely to be
    needle_length = length / py5.random(4, 8)
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

    # how loose will this branch be?
    # 0.0 = tight, 1.0 = loose
    # this will determine whether the needles are actually drawn from the branch or whether there is a gap
    looseness = py5.random(0.3, 1.0)

    # decide if we're starting  on the left or right side
    left = bool(py5.random_int(0, 1))

    for needle_start_distance in needle_starts:
        # decide if we're on the left or right side
        needle_angle_rads = left_angle if left else right_angle

        print(f"needle_start_distance={needle_start_distance:.2f} left={left} branch_angle={branch_angle:.2f} needle_angle_rads={needle_angle_rads:.2f}")

        # x,y coords of perfect needle start
        needle_x, needle_y = curve.point_at(needle_start_distance / length)
        # needle_x = branch_x + needle_start_distance * py5.cos(branch_angle)
        # needle_y = branch_y + needle_start_distance * py5.sin(branch_angle)

        # now we need to adjust the needle start point to account for looseness
        actual_needle_x = needle_x + looseness * needle_length/3 * py5.cos(needle_angle_rads)
        actual_needle_y = needle_y + looseness * needle_length/3 * py5.sin(needle_angle_rads)

        # draw the needle
        board.d(py5.line,
                actual_needle_x,
                actual_needle_y,
                actual_needle_x + needle_length * py5.cos(needle_angle_rads),
                actual_needle_y + needle_length * py5.sin(needle_angle_rads))

        # now set up for next needle to be on other side
        left = not left


def draw_tree(board: Board, x, y, width, height):
    # draw the branches of a tree, we don't need to draw a trunk - leave that to the immagination
    # essentially we want to draw the typical branches of a tree; starting in the middle and branching outwards
    # towards the bottom of the tree the branches should be longer and more horizontal, towards the top they should
    # be shorter and more vertical
    # the entire bound of the three should be an isosceles triangle with the top point at the top of the tree
    # and the base at the bottom of the tree
    middle_base_x, middle_base_y = x + width / 2, y + height
    top_x, top_y = x + width / 2, y
    board.d(py5.text, "mb", middle_base_x, middle_base_y)
    board.d(py5.text, "top", top_x, top_y)

    def typical_branch_length(branch_y):
        # branch_y is the distance from the top of the tree

        # longer branches at the bottom, shorter branches at the top
        # 0.1*height at top to 0.18*height at bottom
        return 0.1 * height + 0.08 * height * (branch_y/height)

    top_branch_length = typical_branch_length(0)

    draw_branch(board, top_x, top_y+top_branch_length, top_branch_length, 270)
    draw_branch(board, top_x-10, top_y+top_branch_length+20, top_branch_length-10, 270-50)
    draw_branch(board, top_x+10, top_y+top_branch_length+20, top_branch_length-10, 270+50)


def settings():
    py5.size(300, 600)


def setup():
    py5.random_seed(123456)


def draw():
    board = Board()

    board.d(py5.begin_record, py5.SVG, 'tree.svg')
    board.d(py5.no_fill)
    draw_tree(board,50, 50, 200, 500)
    board.d(py5.end_record)

    board.print_instructions()

    for instruction in board.instructions:
        instruction[0](*instruction[1])
    # draw_branch(150, 150, 100, 10, needle_density=16, staggering=0)
    # draw_branch(150, 150, 100, 50, needle_density=12, staggering=0)
    # draw_branch(150, 150, 100, 90, needle_density=8, staggering=0)
    # draw_branch(150, 150, 100, 130, needle_density=16, staggering=0.5)
    # draw_branch(150, 150, 100, 170, needle_density=12, staggering=0.5)
    # draw_branch(150, 150, 100, 210, needle_density=8, staggering=0.5)
    # draw_branch(150, 150, 100, 250, needle_density=16, staggering=1)
    # draw_branch(150, 150, 100, 290, needle_density=12, staggering=1)
    # draw_branch(150, 150, 100, 320, needle_density=8, staggering=1)

    py5.exit_sketch()


py5.run_sketch()
