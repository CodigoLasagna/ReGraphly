import random
import math
#from PIL import Image, ImageDraw, ImageFont


def gen_random_number(size):
    return random.randint(0, size)


class node:
    def __init__(self, n_size, tag, w_width, w_height):
        self.value = gen_random_number(n_size)
        self.tag = tag
        self.neighbors = []
        self.sep_factor = 300
        self.x = random.randint((w_width/2)-self.sep_factor, (w_width/2)+self.sep_factor)
        self.y = random.randint((w_height/2)-self.sep_factor, (w_height/2)+self.sep_factor)


class weight:
    def __init__(self, node_a, node_b, n_range):
        self.cost = random.randint(0, n_range)
        self.node_a = node_a
        self.node_b = node_b


class graph:
    def __init__(self):
        self.tag = ""
        self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.tag_count = 0
        self.nodes = []
        self.weights = []
        self.structure_size = 10
        self.node_n_size = 10
        self.w_width = 900
        self.w_height = 900
        self.density = 100
        self.weight_val_range = 10

    def set_canvas_size(self, width, height):
        self.w_width = width
        self.w_height = height

    def set_structure_size(self, size):
        self.structure_size = size

    def set_nodes_max_val(self, n_size):
        self.node_n_size = n_size

    def set_weights_max_val(self, weight_val_range):
        self.weight_val_range = weight_val_range

    def generate_next_tag(self):
        result = ""
        quotient = self.tag_count
        while quotient >= 0:
            quotient, remainder = divmod(quotient, len(self.letters))
            result = self.letters[remainder] + result
            quotient -= 1
        return result

    def add_custom_node(self, tag, value, x, y):
        new_node = node(0, tag, self.w_width, self.w_height)
        new_node.value = value
        new_node.x = x
        new_node.y = self.w_height - y
        self.nodes.append(new_node)

    def add_custom_connection(self, node_a, node_b):
        if isinstance(node_a, str):
            node_a = next((node for node in graph.nodes if node.tag == start_node), None)
        if isinstance(node_b, str):
            node_b = next((node for node in graph.nodes if node.tag == start_node), None)
    
        if node_a is not None and node_b is not None and node_b not in node_a.neighbors:
            node_a.neighbors.append(node_b)
            self.weights.append(weight(node_a, node_b, self.weight_val_range))

    def build_with_geo_data(self, data = []):
        for tag, x, y in data:
            new_node = node(0, tag, self.w_width, self.w_height)
            new_node.x = x
            new_node.y = y
            self.nodes.append(new_node)

    def build_with_custom(self, data = []):
        for tag, value, x, y in data:
            self.add_custom_node(tag, value, x, y)


    def prepare_random_nodes(self):
        for i in range(self.structure_size):
            self.tag = self.generate_next_tag()
            self.nodes.append(node(self.node_n_size, self.tag, self.w_width, self.w_height))
            self.tag_count += 1

    def build_random(self, full_connected = True, density = 100):
        st = self.structure_size
        if (st > len(self.nodes)):
            st = len(self.nodes)
        existing_weights = set()
        main_tree = []
        main_tree.append(random.choice(self.nodes))
        self.density = (density/100)

        for node in self.nodes:
            for i in range(random.randint(0, math.floor((st-1)*self.density))):
                new_node = self.nodes[random.randint(0, st-1)]
                while ((new_node in node.neighbors) or node == new_node):
                    new_node = self.nodes[random.randint(0, st-1)]

                if (new_node, node) not in existing_weights and (node, new_node) not in existing_weights:
                    node.neighbors.append(new_node)
                    if (new_node in main_tree):
                        if (node not in main_tree):
                            main_tree.append(node)
                    self.weights.append(weight(node, new_node, self.weight_val_range))
                    existing_weights.add((node, new_node))
                    existing_weights.add((new_node, node))

        if (full_connected):
            for node in self.nodes:
                if (node not in main_tree):
                    new_r_node = random.choice(main_tree)
                    if (node, new_r_node) not in existing_weights and (new_r_node, node) not in existing_weights:
                        self.weights.append(weight(node, new_r_node, self.weight_val_range))
                        existing_weights.add((node, new_r_node))
                        existing_weights.add((new_r_node, node))
                        main_tree.append(node)


    def print_graph(self):
        result = ""
        for node in self.nodes:
            # FIX: this is broken
            result = node.tag + " - VAL["+str(node.value)+"] - N["
            result += str(node.neighbor.tag) + "]"
            print(result)

    def print_weitghts(self):
        result = ""
        for w in self.weights:
            result = "N1["+str(w.node_a.value)+"]["+w.node_a.tag+"] - "
            result += "N2["+str(w.node_b.value)+"]["+w.node_b.tag+"] "
            result += "w["+str(w.cost)+"] "
            print(result)


def calculate_repulsion(node1, node2, k_repulsion):
    dx = node1.x - node2.x
    dy = node1.y - node2.y
    distance = math.sqrt(dx**2 + dy**2)
    if distance == 0:
        return (0, 0)
    force = k_repulsion / distance**2
    force_x = force * dx / distance
    force_y = force * dy / distance
    return (force_x, force_y)

def calculate_attraction(node1, node2, k_attraction):
    dx = node2.x - node1.x
    dy = node2.y - node1.y
    distance = math.sqrt(dx**2 + dy**2)
    force = k_attraction * distance
    force_x = force * dx / distance
    force_y = force * dy / distance
    return (force_x, force_y)

def force_directed_layout(graph, iterations=100, k_repulsion=1.0, k_attraction=0.1):
    for _ in range(iterations):
        for node in graph.nodes:
            node.force_x = 0
            node.force_y = 0

        for i, node1 in enumerate(graph.nodes):
            for j, node2 in enumerate(graph.nodes):
                if i < j:
                    repulsion_x, repulsion_y = calculate_repulsion(node1, node2, k_repulsion)
                    node1.force_x += repulsion_x
                    node1.force_y += repulsion_y
                    node2.force_x -= repulsion_x
                    node2.force_y -= repulsion_y

        for weight in graph.weights:
            attraction_x, attraction_y = calculate_attraction(weight.node_a, weight.node_b, k_attraction)
            weight.node_a.force_x += attraction_x
            weight.node_a.force_y += attraction_y
            weight.node_b.force_x -= attraction_x
            weight.node_b.force_y -= attraction_y

        for node in graph.nodes:
            node.x += node.force_x
            node.y += node.force_y

def force_directed_layout_weight(graph, iterations=100, k_repulsion=1.0, k_attraction_base=0.1):
    for _ in range(iterations):
        for node in graph.nodes:
            node.force_x = 0
            node.force_y = 0

        for i, node1 in enumerate(graph.nodes):
            for j, node2 in enumerate(graph.nodes):
                if i < j:
                    repulsion_x, repulsion_y = calculate_repulsion(node1, node2, k_repulsion * len(graph.nodes))
                    node1.force_x += repulsion_x
                    node1.force_y += repulsion_y
                    node2.force_x -= repulsion_x
                    node2.force_y -= repulsion_y

        for weight in graph.weights:
            attraction_x, attraction_y = calculate_attraction(weight.node_a, weight.node_b, k_attraction_base * weight.cost * (graph.density))
            weight.node_a.force_x += attraction_x
            weight.node_a.force_y += attraction_y
            weight.node_b.force_x -= attraction_x
            weight.node_b.force_y -= attraction_y

        for node in graph.nodes:
            node.x += node.force_x
            node.y += node.force_y

class MinHeap:
    def __init__(self):
        self.heap = []

    def push(self, item):
        self.heap.append(item)
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        if len(self.heap) == 0:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()

        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return root

    def _heapify_up(self, index):
        parent_index = (index - 1) // 2
        while index > 0 and self.heap[index][0] < self.heap[parent_index][0]:
            self.heap[index], self.heap[parent_index] = self.heap[parent_index], self.heap[index]
            index = parent_index
            parent_index = (index - 1) // 2

    def _heapify_down(self, index):
        while True:
            left_child_index = 2 * index + 1
            right_child_index = 2 * index + 2
            smallest = index

            if (
                left_child_index < len(self.heap)
                and self.heap[left_child_index][0] < self.heap[smallest][0]
            ):
                smallest = left_child_index

            if (
                right_child_index < len(self.heap)
                and self.heap[right_child_index][0] < self.heap[smallest][0]
            ):
                smallest = right_child_index

            if smallest == index:
                break

            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            index = smallest

class SimpleDeque:
    def __init__(self):
        self.items = []

    def append(self, item):
        self.items.append(item)

    def popleft(self):
        if not self.is_empty():
            return self.items.pop(0)
        else:
            raise IndexError("pop from an empty deque")

    def is_empty(self):
        return len(self.items) == 0
