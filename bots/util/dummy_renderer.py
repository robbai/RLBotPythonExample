from rlbot.utils.rendering.rendering_manager import RenderingManager

class DummyRenderer(RenderingManager):

    def __init__(self, renderer):
        self.renderGroup = renderer.renderGroup
        self.render_state = renderer.render_state
        self.builder = renderer.builder
        self.render_list = renderer.render_list
        self.group_id = renderer.group_id
        self.bot_index = renderer.bot_index
        self.bot_team = renderer.bot_team
        
    def clear_screen(self, group_id='default'):
        pass

    def draw_line_2d(self, x1, y1, x2, y2, color):
        return self

    def draw_polyline_2d(self, vectors, color):
        return self

    def draw_line_3d(self, vec1, vec2, color):
        return self

    def draw_polyline_3d(self, vectors, color):
        return self

    def draw_line_2d_3d(self, x, y, vec, color):
        return self

    def draw_rect_2d(self, x, y, width, height, filled, color):
        return self

    def draw_rect_3d(self, vec, width, height, filled, color, centered=False):
        return self

    def draw_string_2d(self, x, y, scale_x, scale_y, text, color):
        return self

    def draw_string_3d(self, vec, scale_x, scale_y, text, color):
        return self

    def begin_rendering(self, group_id='default'):
        pass

    def end_rendering(self):
        pass

    def clear_all_touched_render_groups(self):
        pass
