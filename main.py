import graphlib as gl
import geo_cords

import pyglet
from pyglet import shapes
from pyglet import image



def dfs_algorithm(graph, start_node):
    visited = set()

    if isinstance(start_node, str):
        start_node = next((node for node in graph.nodes if node.tag == start_node), None)
    
    if start_node is None:
        print("El nodo no existe")
        return

    def dfs_recursive(node):
        visited.add(node)
        print("Visitando nodo:", node.tag)

        for edge in graph.weights:
            if edge.node_a == node and edge.node_b not in visited:
                dfs_recursive(edge.node_b)
            elif edge.node_b == node and edge.node_a not in visited:
                dfs_recursive(edge.node_a)

    dfs_recursive(start_node)

def bfs_algorithm(graph, start_node):
    visited = set()

    if isinstance(start_node, str):
        start_node = next((node for node in graph.nodes if node.tag == start_node), None)

    if start_node is None:
        print("Nodo no encontrado")
        return

    queue = gl.SimpleDeque()
    queue.append(start_node)
    visited.add(start_node)

    while not queue.is_empty():
        current_node = queue.popleft()
        print("visitando nodo:", current_node.tag)

        for edge in graph.weights:
            if edge.node_a == current_node and edge.node_b not in visited:
                queue.append(edge.node_b)
                visited.add(edge.node_b)
            elif edge.node_b == current_node and edge.node_a not in visited:
                queue.append(edge.node_a)
                visited.add(edge.node_a)


def dfs_algorithm_route(graph, start_node, goal_node):
    visited = set()
    path = []

    def dfs_recursive(node, current_path):
        visited.add(node)

        if node == goal_node:
            path.extend(current_path)
            return True

        for edge in graph.weights:
            neighbor = None
            if edge.node_a == node and edge.node_b not in visited:
                neighbor = edge.node_b
            elif edge.node_b == node and edge.node_a not in visited:
                neighbor = edge.node_a

            if neighbor:
                if dfs_recursive(neighbor, current_path + [edge]):
                    return True

        return False

    if isinstance(start_node, str):
        start_node = next((node for node in graph.nodes if node.tag == start_node), None)

    if isinstance(goal_node, str):
        goal_node = next((node for node in graph.nodes if node.tag == goal_node), None)

    if start_node is None or goal_node is None:
        print("Start or goal node not found.")
        return []

    dfs_recursive(start_node, [])
    return path

def bfs_algorithm_route(graph, start_node, goal_node):
    visited = set()
    path = []

    if isinstance(start_node, str):
        start_node = next((node for node in graph.nodes if node.tag == start_node), None)

    if isinstance(goal_node, str):
        goal_node = next((node for node in graph.nodes if node.tag == goal_node), None)

    if start_node is None or goal_node is None:
        print("Start or goal node not found.")
        return []

    queue = gl.SimpleDeque()
    queue.append((start_node, []))

    while not queue.is_empty():
        current_node, current_path = queue.popleft()
        visited.add(current_node)

        if current_node == goal_node:
            path = current_path
            break

        for edge in graph.weights:
            neighbor = None
            if edge.node_a == current_node and edge.node_b not in visited:
                neighbor = edge.node_b
            elif edge.node_b == current_node and edge.node_a not in visited:
                neighbor = edge.node_a

            if neighbor:
                queue.append((neighbor, current_path + [edge]))
                visited.add(neighbor)

    return path

def bfs_algorithm_selection(graph, start_node):
    visited = set()

    if isinstance(start_node, str):
        start_node = next((node for node in graph.nodes if node.tag == start_node), None)

    if start_node is None:
        print("Nodo no encontrado")
        return

    queue = gl.SimpleDeque()
    queue.append((start_node, 0))
    visited.add(start_node)

    max_distance = 0
    eccentricities = []

    while not queue.is_empty():
        current_node, distance = queue.popleft()

        for edge in graph.weights:
            if edge.node_a == current_node and edge.node_b not in visited:
                queue.append((edge.node_b, distance + 1))
                visited.add(edge.node_b)
            elif edge.node_b == current_node and edge.node_a not in visited:
                queue.append((edge.node_a, distance + 1))
                visited.add(edge.node_a)

        # Actualizar la distancia máxima para obtener la excentricidad
        max_distance = max(max_distance, distance)

    eccentricities.append(max_distance)

    # Calcular el radio y el diámetro
    radius = min(eccentricities)
    diameter = max(eccentricities)

    return eccentricities, radius, diameter

class main_window(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        config = pyglet.gl.Config(double_buffer=True, sample_buffers=1, samples=4)
        super().__init__(*args, config=config, **kwargs)
        self.graph_g = None
        self.batch = None
        self.graph = None
        self.vp_x = 0
        self.vp_y = 0
        self.vp_z = 0
        
        bkgcolor_h = "#1e2935"
        bkgcolor_h = bkgcolor_h.strip('#')
        self.bkg_c = tuple(int(bkgcolor_h[i:i+2], 16) for i in (0, 2, 4))
        #pyglet.gl.glClearColor(self.bkg_c[0]/255, self.bkg_c[1]/255, self.bkg_c[2]/255, 1)

        self.cur_node = None
        self.node_index = -1
        self.cur_edges = []

    def on_draw(self):
        self.clear()
        if (self.batch):
                self.batch.draw()

    def on_key_release(self, symbol, modifiers):
        if (symbol == pyglet.window.key.Q):
            pyglet.app.exit()
        if (symbol == pyglet.window.key.D):
            self.vp_x += 10
            self.view = self.view.from_translation((self.vp_x, self.vp_y, 0))
        if (symbol == pyglet.window.key.A):
            self.vp_x -= 10
            self.view = self.view.from_translation((self.vp_x, self.vp_y, 0))

        if (symbol == pyglet.window.key.W):
            self.vp_y += 10
            self.view = self.view.from_translation((self.vp_x, self.vp_y, 0))
        if (symbol == pyglet.window.key.S):
            self.vp_y -= 10
            self.view = self.view.from_translation((self.vp_x, self.vp_y, 0))

    def on_mouse_press(self, x, y, buttons, modifiers):
        if (buttons == pyglet.window.mouse.LEFT):
            for i, n in enumerate(self.graph.nodes):
                distance = gl.math.sqrt(gl.math.pow((x - self.vp_x) - n.x, 2) + gl.math.pow((y - self.vp_y) - n.y, 2))
                if (distance < 20):
                    self.cur_node = n
                    self.node_index = i
            if (self.cur_node):
                self.node_index = self.node_index
                for i, e in enumerate(self.graph.weights):
                    if (e.node_a == self.cur_node or e.node_b == self.cur_node):
                        self.cur_edges.append(self.graph_g.edges[i])

    def on_mouse_release(self, x, y, buttons, modifiers):
        if (buttons == pyglet.window.mouse.LEFT):
            self.cur_node = None
            self.node_index = -1
            self.cur_edges.clear()
            #draw_edges(lines_g=self.lines_g, edges=self.graph.weights, color='#00CFD5', batch=self.batch, draw_cost=True)
            graph_g.update()
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.vp_z += (scroll_y*2)
        print(self.vp_z)
        self.view = self.view.from_translation((self.vp_x, self.vp_y, self.vp_z))


    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if ( buttons == pyglet.window.mouse.MIDDLE or buttons == pyglet.window.mouse.RIGHT):
            self.vp_x += dx
            self.vp_y += dy
            self.view = self.view.from_translation((self.vp_x, self.vp_y, 0))
        if (buttons == pyglet.window.mouse.LEFT):
            if (self.cur_node):
                self.cur_node.x = x - self.vp_x
                self.cur_node.y = y - self.vp_y
                graph_g.nodes[self.node_index].draw()
                if (len(self.cur_edges) > 0):
                    for c_e in self.cur_edges:
                        c_e.draw()


class node_g:
    def __init__(self, node : gl.node, outline_color : str = '#C40F8F', fill_color : str = '#3B435B', radius : float = 20.0, batch = None):
        self.fill_color_h = fill_color.lstrip("#")
        self.outline_color_h = outline_color.lstrip("#")
        self.outline_color = tuple(int(self.outline_color_h[i:i+2], 16) for i in (0, 2, 4))
        self.fill_color = tuple(int(self.fill_color_h[i:i+2], 16) for i in (0, 2, 4))
        self.node = node
        self.batch = batch
        self.tag = node.tag
        self.radius = radius
        self.draw()
    def draw(self):
        self.fill_color_h = self.fill_color_h.lstrip("#")
        self.outline_color_h = self.outline_color_h.lstrip("#")
        self.outline_color = tuple(int(self.outline_color_h[i:i+2], 16) for i in (0, 2, 4))
        self.fill_color = tuple(int(self.fill_color_h[i:i+2], 16) for i in (0, 2, 4))
        self.outline_s = shapes.Circle(self.node.x, self.node.y, self.radius+2, color=self.outline_color, batch=self.batch, group=pyglet.graphics.Group(order=1))
        self.fill_s = shapes.Circle(self.node.x, self.node.y, self.radius, color=self.fill_color, batch=self.batch, group=pyglet.graphics.Group(order=2))
        self.text = pyglet.text.Label(self.node.tag + ":" + str(self.node.value), font_name='Agave Nerd Font', font_size=11, x = self.node.x, y = self.node.y, anchor_x='center', anchor_y='center', batch=self.batch, group=pyglet.graphics.Group(order=3))

class edge_g:
    def __init__(self, color : str = '#00CFD5', width = 1, edge : gl.weight = None, batch = None, show_cost : bool = False):
        self.color_h = color
        self.color_hs = self.color_h.lstrip("#")
        self.color = tuple(int(self.color_hs[i:i+2], 16) for i in (0, 2, 4))
        self.width = width
        self.edge = edge
        self.batch = batch
        self.show_cost = show_cost
        self.cost_bkg = None
        self.label = None
        self.draw()

    def draw(self):
        self.color_hs = self.color_h.lstrip("#")
        self.color = tuple(int(self.color_hs[i:i+2], 16) for i in (0, 2, 4))
        self.line_s = shapes.Line(x=self.edge.node_a.x, y=self.edge.node_a.y, x2=self.edge.node_b.x, y2=self.edge.node_b.y, color=self.color, batch=self.batch, width=self.width, group=pyglet.graphics.Group(order=0))
        if (self.show_cost):
            w_x = (self.edge.node_a.x + self.edge.node_b.x)/2
            w_y = (self.edge.node_a.y + self.edge.node_b.y)/2
            self.cost_bkg = shapes.Rectangle(x=w_x-8, y=w_y-8, width=16, height=16, color=(0, 0, 0), batch=self.batch, group=pyglet.graphics.Group(order=1))
            self.label = pyglet.text.Label(str(self.edge.cost), font_name='Cantarell', font_size=8, x = w_x, y = w_y, anchor_x='center', anchor_y='center', batch=self.batch, group=pyglet.graphics.Group(order=1))
        else:
            self.cost_bkg = None
            self.label = None

class graph_g:
    def __init__(self, graph : gl.graph, batch = None, nodes_radius : float = 20.0):
        self.nodes_radius = nodes_radius
        self.graph = graph
        self.batch = batch
        self.nodes = []
        self.edges = []


    def prepare(self):
        if (self.graph.nodes):
            for n in self.graph.nodes:
                self.nodes.append(node_g(node=n, batch=self.batch, radius=self.nodes_radius))
        if (self.graph.weights):
            for w in self.graph.weights:
                self.edges.append(edge_g(edge=w, batch=self.batch))
    def update(self):
        for n in self.nodes:
            n.draw()
        for w in self.edges:
            w.draw()

    def update_weights(self, n_edges, color='#00CFD5', width=1, show_cost=False):
        for w in self.edges:
            for w_n in n_edges:
                if (w.edge == w_n):
                    w.color_h = color
                    w.width = width
                    w.show_cost = show_cost
        self.update()

    def update_nodes(self, n_nodes, outline_color='#C40F8F', fill_color='#3B435B'):
        for n in self.nodes:
            for n_node in n_nodes:
                if n.node == n_node:
                    n.outline_color_h = outline_color
                    n.fill_color_h = fill_color
        self.update()

if __name__ == '__main__1':
    seed = gl.random.randint(0, 1000000000)
    seed = 20
    gl.random.seed(seed)
    print(seed)
    graph = gl.graph()
    graph.set_canvas_size(1000, 1000)
    graph.set_structure_size(size=15)
    graph.set_nodes_max_val(99)
    graph.set_weights_max_val(20)
    graph.prepare_random_nodes()
    graph.build_random(full_connected=True, density=20)

    gl.force_directed_layout_weight(graph, iterations=1000, k_repulsion=1200.0, k_attraction_base=0.005)
    mst = kruskal_algorithm(graph)
    path_to, distance_to = shortest_distance_to(graph, "A", "D")
    print("distance: " + str(distance_to))

    batch = pyglet.graphics.Batch()
    nodes_list = []
    window = main_window(width=1920, height=1080, resizable=True)
    graph_g = graph_g(graph, batch)
    graph_g.prepare()
    window.graph_g = graph_g
    #graph_g.update_weights(mst, '#BA5337', 2)
    #graph_g.update_weights(graph.weights, '#00CFD5', 1, True)
    graph_g.update_weights(path_to, '#90FF09', 3, False)
    window.batch = batch
    window.graph = graph

    pyglet.app.run()


def load_metro_line(graph : gl.graph, name : str):
    sep = 0
    with open(name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if ("---" in line):
                lines = lines[lines.index(line) + 1:]
                break
            parts = line.strip().split("\"")
            tag = parts[1]
            value = int(parts[0].replace("[", '').replace("]", ''))
            x, y = map(int, parts[2][1:-1].replace("(", '').split(','))
            graph.add_custom_node(tag, value, x*4, y*4)
            sep += 1
        for line in lines:
            parts = line.strip().split("][")
            n1 = int(parts[0].replace("[", ''))
            n2 = int(parts[1].replace("]", ''))
            graph.add_custom_connection(graph.nodes[n1], graph.nodes[n2])

if __name__ == '__main__':
    graph = gl.graph()
    load_metro_line(graph, "maps/linea6.txt")
    load_metro_line(graph, "maps/linea5.txt")
    load_metro_line(graph, "maps/linea1.txt")

    window = main_window(width=800, height=800, resizable=True)
    batch = pyglet.graphics.Batch()
    graph_g = graph_g(graph, batch)
    graph_g.prepare()
    window.graph_g = graph_g
    window.batch = batch
    window.graph = graph
    pyglet.app.run()
