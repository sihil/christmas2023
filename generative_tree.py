import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Set, Protocol
from xml.etree import ElementTree
import random

import py5
import vpype_cli
from shapely import Polygon, GEOSException
from vpype import text_line, FONT_NAMES


parser = argparse.ArgumentParser(
    prog='GenerativeTree',
    description='Generates a christmas tree card')
parser.add_argument('--svg-std-out', dest='svg_std_out', action='store_true', help='output svg to stdout')
parser.add_argument('--seed', type=int, help='seed for random number generator')

args = parser.parse_args(args=sys.argv[1:])
print(args, file=sys.stderr)

seed = args.seed
pixels_per_mm = 3.7795275591


class SpatialShape(Protocol):
    @property
    def coords(self) -> List[Tuple[float, float]]:
        raise NotImplemented


class SpatialHash(object):
    """
    a spatial index which can be used for a broad-phase collision detection strategy.
    """

    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int], List[SpatialShape]] = {}

    def key(self, x: float, y: float):
        cell_size = self.cell_size
        import math
        return (
            int((math.floor(x / cell_size)) * cell_size),
            int((math.floor(y / cell_size)) * cell_size)
        )

    def insert(self, x: float, y: float, value):
        """
        Insert object into the spatial hash map.
        """
        key = self.key(x, y)
        objects = self.grid.get(key)
        if objects is None:
            objects = []
            self.grid[key] = objects
        objects.append(value)

    def insert_shape(self, shape: SpatialShape):
        """
        Insert object into the spatial hash map.
        """
        for x, y in shape.coords:
            self.insert(x, y, shape)

    def query(self, x: float, y: float) -> List[SpatialShape]:
        """
        Return all objects in the cell specified by point.
        """
        return self.grid.get(self.key(x, y), set())

    def query_shape(self, shape: SpatialShape) -> List[SpatialShape]:
        """
        Return all objects in the cell specified by point.
        """
        return [b
                for x, y in shape.coords
                for b in self.query(x, y)]


Instruction = Tuple[Callable[..., ...], List, Dict]


def instr(func: Callable[..., ...], *args, **kwargs) -> Instruction:
    return func, list(args), kwargs


class Board:
    def __init__(self):
        self._instructions: Dict[str, List[Instruction]] = {}

    def d(self, layer: str, func: Callable[..., ...], *args, **kwargs):
        self.instructions(layer).append((func, list(args), kwargs))

    def ds(self, layer: str, instructions: List[Instruction]):
        self.instructions(layer).extend(instructions)

    def instructions(self, layer: str) -> List[Instruction]:
        if layer not in self._instructions:
            self._instructions[layer] = []
        return self._instructions[layer]

    def layers(self) -> List[str]:
        return list(self._instructions.keys())

    def print_instructions_for_layer(self, layer: str):
        for instruction in self.instructions(layer):
            print(instruction, file=sys.stderr)


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

    def angle_at(self, t) -> float:
        tx = py5.curve_tangent(self.x1, self.x2, self.x3, self.x4, t)
        ty = py5.curve_tangent(self.y1, self.y2, self.y3, self.y4, t)
        return py5.atan2(ty, tx)


@dataclass
class Branch:
    x: float
    y: float
    length: float
    angle: float  # degrees
    branch_needle_ratio: float = field(default_factory=lambda: 1.0)
    needle_density: Optional[float] = field(default_factory=lambda: py5.random(4, 4.5))
    staggering: Optional[float] = field(default_factory=lambda: py5.random(0.1, 0.8))
    needle_angle_offset: Optional[float] = field(default_factory=lambda: py5.random(20, 50))
    instructions: List[Instruction] = field(init=False)
    outline_polygon_points: List[Tuple[float, float]] = field(init=False)
    polygon: Polygon = field(init=False)

    def __post_init__(self):
        self.instructions = []

        # convert angle to radians
        branch_angle = py5.radians(self.angle)

        # draw the branch

        # draw a branch as a slight curve
        # how much of a curve do we want?
        # 0.0 = straight, 1.2 = very curvy
        angle_with_zero_top = (self.angle + 90) % 360
        degrees_from_up = angle_with_zero_top if angle_with_zero_top < 180 else 360 - angle_with_zero_top
        max_curvy = 0.6 * (degrees_from_up / 180) + 0.4
        curvy_amount = py5.random(0.05, max_curvy)

        # figure out the direction of curve
        # typically branches on the left (angle_with_zero_top 180-360) will curve to the right and vice versa
        # curving to the right will happen with a positive curvy_amount
        curve_left = angle_with_zero_top < 180
        if curve_left:
            curvy_amount = -curvy_amount

        branch_end_x = self.x + self.length * py5.cos(branch_angle)
        branch_end_y = self.y + self.length * py5.sin(branch_angle)
        curve = Curve(
            self.x - self.length * py5.cos(branch_angle - curvy_amount),
            self.y - self.length * py5.sin(branch_angle - curvy_amount),
            self.x,
            self.y,
            branch_end_x,
            branch_end_y,
            self.x + 2 * self.length * py5.cos(branch_angle + curvy_amount),
            self.y + 2 * self.length * py5.sin(branch_angle + curvy_amount)
        )

        self.instructions.append(
            instr(
                py5.curve,
                *curve.as_list(),
                notes=f'curviness: {curvy_amount:.2f} angle: {self.angle:.2f} angle_with_zero_top: '
                      f'{angle_with_zero_top:.2f} degrees_from_up: {degrees_from_up:.2f}'
            )
        )

        # now draw a series of needles along the branch

        # how long is a needle likely to be
        shortest_length = self.length * self.branch_needle_ratio / (self.needle_angle_offset / 3)
        longest_length = self.length * self.branch_needle_ratio / (self.needle_angle_offset / 6)
        needle_length = py5.random(shortest_length, longest_length)

        # what portion of branch will there be before the first needle?
        first_needle_distance = py5.random(0, self.length / 4)
        last_needle_distance = self.length - py5.random(0, needle_length)

        needle_populated_length = last_needle_distance - first_needle_distance

        # how many needles will there be?
        needle_count = int(needle_populated_length // self.needle_density)
        needle_count_first_side = needle_count // 2 + needle_count % 2

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
                    return prev_needle + first_side_interval * self.staggering / 2
            else:
                # even number of needles, so we can just distribute evenly taking staggering into account
                first_side_interval = (
                        (last_needle_distance - first_needle_distance) / needle_count_first_side + self.staggering * 0.5
                )
                if first_side:
                    return first_needle_distance + first_side_interval * side_i
                else:
                    prev_needle = start_distance(i - 1)
                    return prev_needle + first_side_interval * self.staggering / 2

        needle_starts_all = [start_distance(i) for i in range(needle_count)]
        # filter this list to occasionally remove one of the needles
        # this will make the tree look more natural
        needle_starts = [x for x in needle_starts_all]  # if py5.random(0, 30) % 30 == 0]
        # (last_needle_distance - first_needle_distance) / needle_count

        # needle angle offset from branch angle (30º to 50º)
        left_angle = py5.radians(self.needle_angle_offset + random.randint(-2, 2))
        right_angle = py5.radians(self.needle_angle_offset + random.randint(-2, 2))

        # how loose will this branch be?
        # 0.0 = tight, 1.0 = loose
        # this will determine whether the needles are actually drawn from the branch or whether there is a gap
        looseness = py5.random(0.3, 1.0)

        base_needle_curviness = abs(curvy_amount) * 0.7

        outer_needle_curve_follows_branch = py5.random() + base_needle_curviness > 0.7

        # decide if we're starting  on the left or right side
        left = bool(py5.random_int(0, 1))

        self.outline_polygon_points = [(self.x, self.y)]

        for needle_start_distance in needle_starts:
            # calculate the needle angle
            branch_angle_at_needle_start = curve.angle_at(needle_start_distance / self.length)
            needle_angle_rads = branch_angle_at_needle_start - left_angle if left \
                else branch_angle_at_needle_start + right_angle

            # x,y coords of perfect needle start
            needle_x, needle_y = curve.point_at(needle_start_distance / self.length)

            # this needles length
            this_needle_length = needle_length * py5.random(0.9, 2)
            this_needle_looseness = looseness * py5.random(0.7, 1.3)

            # now we need to adjust the needle start point to account for looseness
            actual_needle_x = needle_x + this_needle_looseness * this_needle_length / 3 * py5.cos(needle_angle_rads)
            actual_needle_y = needle_y + this_needle_looseness * this_needle_length / 3 * py5.sin(needle_angle_rads)

            outer_needle = curvy_amount > 0 and left or curvy_amount < 0 and not left
            needle_curves_away_from_branch = not outer_needle or outer_needle and not outer_needle_curve_follows_branch
            needle_curviness = base_needle_curviness + py5.random(0.0, 0.1)

            needle_curves_left = (needle_curves_away_from_branch and curvy_amount < 0 or
                                  not needle_curves_away_from_branch and curvy_amount > 0)

            if needle_curves_left:
                needle_curviness = -needle_curviness

            needle_end_x = actual_needle_x + this_needle_length * py5.cos(needle_angle_rads)
            needle_end_y = actual_needle_y + this_needle_length * py5.sin(needle_angle_rads)
            needle_curve = Curve(
                actual_needle_x - this_needle_length * py5.cos(needle_angle_rads - needle_curviness),
                actual_needle_y - this_needle_length * py5.sin(needle_angle_rads - needle_curviness),
                actual_needle_x,
                actual_needle_y,
                needle_end_x,
                needle_end_y,
                actual_needle_x + 2 * this_needle_length * py5.cos(needle_angle_rads + needle_curviness),
                actual_needle_y + 2 * this_needle_length * py5.sin(needle_angle_rads + needle_curviness)
            )

            if left:
                self.outline_polygon_points.append((needle_end_x, needle_end_y))
            else:
                self.outline_polygon_points.insert(0, (needle_end_x, needle_end_y))

            # draw the needle
            self.instructions.append(
                instr(
                    py5.curve,
                    *needle_curve.as_list()
                )
            )

            # now set up for next needle to be on other side
            left = not left

        self.outline_polygon_points.append((branch_end_x, branch_end_y))

        self.polygon = Polygon(self.outline_polygon_points)

    @property
    def coords(self) -> List[Tuple[float, float]]:
        return list(self.polygon.exterior.coords)

    def will_overlap(self, branch: 'Branch') -> bool:
        return self.polygon.intersects(branch.polygon)

    def will_overlap_a_lot(self, branch: 'Branch', threshold: int) -> bool:
        # returns true if the overlap is more than threshold% of the area of this branch or the other branch
        if not self.will_overlap(branch):
            return False
        try:
            area = self.polygon.intersection(branch.polygon).area
            threshold_multiplier = threshold / 100
            return area > threshold_multiplier * self.polygon.area or area > threshold_multiplier * branch.polygon.area
        except GEOSException:
            # this means the intersection could not be created because the polygons are too close together
            return True

    def debug_instructions(self):
        debug_is = [instr(
            py5.begin_shape
        )]
        for x, y in self.outline_polygon_points:
            debug_is.append(
                instr(
                    py5.vertex,
                    x, y
                )
            )
        debug_is.append(
            instr(
                py5.end_shape, py5.CLOSE
            )
        )
        return debug_is


class Bauble:
    def __init__(self, x, y, radius: int):
        self.x = x
        self.y = y
        self.radius = radius

    @property
    def coords(self) -> List[Tuple[float, float]]:
        return [(self.x, self.y)]

    def instructions(self) -> List[Instruction]:
        radii = reversed(list(range(self.radius, 0, -2)))
        return [
            instr(py5.ellipse, self.x, self.y, r, r) for r in radii
        ]


def combine_svgs(layers: List[Tuple[str, str]], new_svg: str):
    py5.begin_record(py5.SVG, new_svg)
    py5.end_record()
    ElementTree.register_namespace('', 'http://www.w3.org/2000/svg')
    tree = ElementTree.parse(new_svg)
    combined = tree.getroot()
    combined.set('xmlns:inkscape',
                 'http://www.inkscape.org/namespaces/inkscape')

    for name, svg in layers:
        markup = ElementTree.parse(svg).getroot()
        group = ElementTree.SubElement(combined, 'g')
        group.set('id', name)
        group.set('inkscape:groupmode', 'bar')
        group.set('inkscape:groupmode', 'layer')
        group.set('inkscape:label', name)

        for child in list(markup):
            group.append(child)

    ElementTree.indent(tree, space="\t", level=0)
    tree.write(new_svg)


def draw_tree(board: Board, x, y, width, height, density=1.0):
    # draw the branches of a tree, we don't need to draw a trunk - leave that to the immagination
    # essentially we want to draw the typical branches of a tree; starting in the middle and branching outwards
    # towards the bottom of the tree the branches should be longer and more horizontal, towards the top they should
    # be shorter and more vertical
    # the entire bound of the three should be an isosceles triangle with the top point at the top of the tree
    # and the base at the bottom of the tree

    # density is the number of branches,
    # if it is increased then the tree is more detailed,
    # if decreased then less detailed
    middle_base_x, middle_base_y = x + width / 2, y + height
    top_x, top_y = x + width / 2, y

    board.d("tree", py5.text, "mb", middle_base_x, middle_base_y)
    board.d("tree", py5.text, "top", top_x, top_y)

    board.d("debug", py5.begin_shape)
    board.d("debug", py5.vertex, top_x, top_y)
    board.d("debug", py5.vertex, x, middle_base_y)
    board.d("debug", py5.vertex, x + width, middle_base_y)
    board.d("debug", py5.end_shape, py5.CLOSE)

    resolution = 30 * density

    jump_pixels = int(width / resolution)

    def typical_branch_length(branch_y):
        # branch_y is the distance from the top of the tree

        # longer branches at the bottom, shorter branches at the top
        # 0.1*height at top to 0.18*height at bottom
        return (0.1 * height + 0.04 * height * (branch_y / height)) / density

    def branch_angle_at(branch_x: float, branch_y: float, typical: bool) -> float:
        # branch_y is the distance from the top of the tree
        # branch_x is the distance from the middle of the tree (negative to the left, positive to the right)

        # imagine the tree split into three horizontal sections
        # the top section is the top 30% of the tree
        # the middle section is the middle 40% of the tree
        # the bottom section is the bottom 30% of the tree

        # imagine a point on the transition between the top and middle sections and another
        top_middle_transition_y = 0.3 * height
        # point on the transition between the middle and bottom sections
        middle_bottom_transition_y = 0.7 * height
        # if you drew a line between these two points, the angle of the branch should trend towards pointing towards
        # the nearest part of that line.
        radial_y = top_middle_transition_y
        if branch_y > top_middle_transition_y:
            radial_y = branch_y
        if branch_y > middle_bottom_transition_y:
            radial_y = middle_bottom_transition_y

        # now we have a radial_y, we can calculate the typical angle of the branch
        # were it to start at 0,radial_y and end at branch_x,branch_y
        angle_rads = py5.atan2(branch_y - radial_y, branch_x)
        angle_degrees = py5.degrees(angle_rads)
        if typical:
            return angle_degrees
        if branch_x > 0:
            # right side of tree
            limit1 = angle_degrees - 30
            limit2 = branch_angle_at(-branch_x, branch_y, typical=True) - 30
            return py5.random(min(limit1, limit2), max(limit1, limit2))
        else:
            # left side of tree
            limit1 = angle_degrees + 30
            limit2 = branch_angle_at(-branch_x, branch_y, typical=True) + 30
            return py5.random(min(limit1, limit2), max(limit1, limit2))

    top_branch_length = typical_branch_length(0)

    spatial_hash = SpatialHash(jump_pixels)

    branches = [
        # Branch(top_x, top_y + top_branch_length, top_branch_length,
        #                branch_angle_at(0, top_y + top_branch_length, typical=True))
    ]

    def add_branch(b: Branch):
        branches.append(b)
        spatial_hash.insert_shape(b)
        print("|", end="", file=sys.stderr)

    def branch_will_overlap_existing(b: Branch, allow_some_overlap=False):
        shortlist = spatial_hash.query_shape(b)
        if allow_some_overlap:
            return any([b.will_overlap_a_lot(existing, threshold=5) for existing in shortlist])
        return any([b.will_overlap(existing) for existing in shortlist])

    # new strategy

    # 1. draw branches using only typical angles (±)
    #   a. at the outside edges of the triangle (perhaps this is width*0.1 from the edge)
    #   b. at the bottom of the triangle (perhaps this is height*0.1 from the bottom)
    # 2. now we need to populate the inside of the triangle with branches
    #   a. we need to find all the gaps in the branches
    #   b. we need to fill the gaps with branches
    # 3. now fill in the remaining gaps with solo needles

    # 1 a/b.
    def populate_with_branches(edge_size: float, fill_prob: float, shorten: bool):
        for search_y in range(int(top_y + top_branch_length), int(middle_base_y), jump_pixels):
            ratio = (search_y - top_y) / height
            left_edge = middle_base_x - (width / 2) * ratio
            right_edge = middle_base_x + (width / 2) * ratio
            width_at_y = right_edge - left_edge
            if width_at_y < edge_size * 2 or search_y > middle_base_y - edge_size:
                x_range_at_y = range(int(left_edge), int(right_edge), jump_pixels)
            else:
                x_range_at_y = (*range(int(left_edge), int(left_edge + edge_size), jump_pixels),
                                *range(int(right_edge - edge_size), int(right_edge), jump_pixels))
            for search_x in x_range_at_y:
                if py5.random() < fill_prob:  # one time in four try to add a branch here
                    print(".", end="", file=sys.stderr)
                    branch_length = typical_branch_length(search_y - top_y)
                    branch_angle = branch_angle_at(search_x - middle_base_x, search_y - top_y, typical=True)
                    new_branch = Branch(search_x, search_y, branch_length, branch_angle)
                    if not branch_will_overlap_existing(new_branch, allow_some_overlap=True):
                        add_branch(new_branch)
                    elif shorten:
                        # try to shorten the branch
                        branch_angle = branch_angle_at(search_x - middle_base_x, search_y - top_y, typical=False)
                        new_branch = Branch(search_x, search_y, branch_length * (2 / 3), branch_angle,
                                            branch_needle_ratio=3 / 2)
                        if not branch_will_overlap_existing(new_branch, allow_some_overlap=True):
                            add_branch(new_branch)
                        else:
                            # try to shorten the branch
                            new_branch = Branch(search_x, search_y, branch_length * (1 / 2), branch_angle,
                                                branch_needle_ratio=2)
                            if not branch_will_overlap_existing(new_branch, allow_some_overlap=True):
                                add_branch(new_branch)

    print(f"\nplotting tree branches at edges", file=sys.stderr)
    populate_with_branches(0.1 * width, 0.25, shorten=False)

    print(f"\nplotting tree branches in next layer", file=sys.stderr)
    populate_with_branches(0.3 * width, 0.75, shorten=True)

    print(f"\nplotting tree branches in middle", file=sys.stderr)
    populate_with_branches(0.5 * width, 0.9, shorten=True)

    print(f"\nmaking baubles", file=sys.stderr)

    for branch in branches:
        board.ds("tree", branch.instructions)
        board.ds("debug", branch.debug_instructions())

    # how about we now add some baubles to the tree?
    # we'll do this by adding some circles to the tree

    def make_baubles() -> List[Bauble]:
        baubles: List[Bauble] = []

        # roughly how many baubles can we place? (based on the size of the tree)
        min_distance = (width / 4.5) / density
        area = width * height * 0.5
        bauble_area = py5.PI * (min_distance/2)**2
        max = area / bauble_area
        print(f"max baubles: {max}", file=sys.stderr)
        aim = int(max * py5.random(0.6, 0.8))
        print(f"aim baubles: {aim}", file=sys.stderr)

        attempt = 0

        while len(baubles) <= aim and attempt < aim*100:
            attempt += 1
            search_y = random.randint(int(top_y + top_branch_length), int(middle_base_y))

            ratio = (search_y - top_y) / height
            left_edge = middle_base_x - (width / 2) * ratio
            right_edge = middle_base_x + (width / 2) * ratio
            search_x = random.randint(int(left_edge), int(right_edge))

            # is there another bauble within the min distance?
            if not any([py5.dist(search_x, search_y, b.x, b.y) < min_distance for b in baubles]):
                baubles.append(Bauble(search_x, search_y, 18))

        return baubles

    for bauble in make_baubles():
        board.ds("baubles", bauble.instructions())


def draw_cockerel(board: Board, x: float, y: float, width: float, flip: bool):
    scaling = width / cockerel.get_width()
    print(f"scaling: {scaling}", file=sys.stderr)

    flipping_multiplier = -1 if flip else 1
    cockerel.scale(flipping_multiplier, 1)

    # put in a debug circle to align the stand
    # board.d("debug", py5.ellipse, x, y, 10, 10)

    board.d("cockerel", py5.scale, scaling, scaling)
    # the x,y given is a location for the bottom of the cockerel stand
    top_x = x/scaling - cockerel.get_width() * 0.4 * flipping_multiplier
    top_y = y/scaling - cockerel.get_height() * 0.8
    board.d("cockerel", py5.shape, cockerel, top_x, top_y)


def plot_text(board, layer: str, x, y, text_lines):
    for line in text_lines.as_mls().geoms:
        coords = list(line.coords)
        if len(coords) == 2:
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            board.d(layer, py5.line, x + x1, y + y1, x + x2, y + y2)
        else:
            board.d(layer, py5.begin_shape)
            for x1, y1 in coords:
                board.d(layer, py5.vertex, x + x1, y + y1)
            board.d(layer, py5.end_shape)


def draw_merry_christmas(board: Board, x: float, y: float, font_name: str = "futural"):
    text_lines = text_line("Merry Christmas", align="center", font_name=font_name, size=35)
    plot_text(board, "merry-christmas", x, y, text_lines)


def draw_credit(board: Board, x: float, y: float, git_commit: str, font_name: str = "futural"):
    text_lines = text_line(f"Design by Simon Hildrew | Card seed {seed}/{git_commit}", align="center", font_name=font_name, size=10)
    plot_text(board, "credits", x, y, text_lines)


def settings():
    # size in px
    width_mm = 210
    height_mm = 148
    py5.size(int(width_mm * pixels_per_mm), int(height_mm * pixels_per_mm))


def setup():
    global cockerel, seed
    if not seed:
        seed = py5.random_int(10000, 999999)
    print(f"seed: {seed}", file=sys.stderr)
    py5.random_seed(seed)
    cockerel = py5.load_shape("cockerel-source.svg")


def draw():
    # initialise a board which will hold multiple layers of instructions
    board = Board()

    border = 0.08 * py5.height
    top = border
    bottom = py5.height - border
    front_left = py5.width/2 + border
    front_right = py5.width - border
    front_middle_x = py5.width * 3/4
    front_middle_y = py5.height / 2

    # some initial instructions
    board.d("tree", py5.no_fill)

    board.d("baubles", py5.stroke, 255, 50, 50)  # red

    board.d("cockerel", py5.no_fill)
    board.d("cockerel", py5.stroke, 212, 175, 55)  # gold

    board.d("debug", py5.no_fill)
    board.d("debug", py5.stroke, 50, 50, 255)  # blue
    board.d("debug", py5.rect, front_left, top, front_right-front_left, bottom-top)
    board.d("debug", py5.line, py5.width/2, 0, py5.width/2, py5.height)

    board.d("merry-christmas", py5.no_fill)
    board.d("merry-christmas", py5.stroke, 212, 175, 55)  # gold
    board.d("credits", py5.no_fill)

    # draw the tree
    draw_tree(board,
              front_left+border, top+border,
              front_right - front_left - border*2, bottom-top - border*2.5,
              density=0.8)

    draw_cockerel(board, front_middle_x, top+(front_right-front_left)*0.2, (front_right-front_left)*0.2, flip=py5.random_choice([True, False]))

    draw_merry_christmas(board, front_middle_x, bottom - 12, font_name='scriptc')

    git_commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    draw_credit(board, py5.width/4, py5.height*0.9, git_commit)

    # debug: print out the instructions for the tree layer
    board.print_instructions_for_layer("cockerel")

    # now actually write each layer into an SVG file
    for layer in board.layers():
        py5.begin_record(py5.SVG, f'{layer}.svg')
        for instruction in board.instructions(layer):
            instruction[0](*instruction[1])
        py5.end_record()

    print("Combining SVGs", file=sys.stderr)

    combine_svgs(
        [
            ('1-cockerel', 'cockerel.svg'),
            ('2-tree', 'tree.svg'),
            ('3-baubles', 'baubles.svg'),
            ('1-merry-christmas', 'merry-christmas.svg'),
            ('2-credits', 'credits.svg'),
            ('9-debug', 'debug.svg'),
        ],
        'combined.svg')

    print("Post-processing combined SVG", file=sys.stderr)

    vpype_cli.execute("read combined.svg "
                      f"scaleto 210mm 148mm "
                      "occult --layer 2,3 -a "
                      "linesort --layer 1,2,4,5 --no-flip "
                      f"write combined-processed.svg")

    # copy file to seed indexed version
    indexed_file_name = f"combined-processed-{seed}-{git_commit}.svg"
    shutil.copyfile("combined-processed.svg", indexed_file_name)

    if args.svg_std_out:
        # print the svg to stdout
        with open("combined-processed.svg", "r") as f:
            print(f.read())

    # close the sketch
    py5.exit_sketch()


py5.run_sketch()
