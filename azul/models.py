from pygame.math import Vector2
from utils import load_sprite
import pygame
from pygame.transform import scale
import numpy as np
# from logic_wrapper import

SIZE_SCREEN = Vector2([1280, 720])
SIZE_BOARD = Vector2([600, 400])
SIZE_FACTORY = Vector2([130, 130])
SIZE_TILE = Vector2([50, 50])

WHITE = (255, 255, 255)
BEIGE = (240, 186, 140)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 50)

NUM_ROWS = 5
EMPTY_TILE = -1


class GameObject:
    def __init__(self, position, sprite, size, zoom=1):
        self.position = Vector2(position)
        self.sprite = sprite
        self.size = Vector2(size)
        self.zoom = zoom

    def draw(self, surface):
        blit_position = self.position - Vector2(self.size) * self.zoom / 2
        surface.blit(scale(self.sprite, self.size * self.zoom), blit_position)


class Board(GameObject):

    def __init__(self, name, content, position, size=(600, 400), zoom=1.0, isHuman=False):  # 900x600
        super().__init__(
            position,
            scale(load_sprite("board2.png"), size),
            Vector2(size),
            zoom
        )
        self.content = content
        self.name = name
        self.rect = pygame.Rect(position - size/2, size)
        self.isHuman = isHuman
        # {'grid': np.array 5x5,
        #  'line': [[]]*5,
        #  'score': int,
        #  'floor': []}

        # rects for lines
        self.rect_line = []
        p1 = self.position + Vector2(430-900/2, 100-600/2)*2/3 * self.zoom
        w = (85*2/3) * self.zoom
        for iy in range(5):
            wh = Vector2(iy+1, 1) * w
            p0 = p1 + Vector2(-iy-1, iy-1) * w
            self.rect_line.append(pygame.Rect(p0, wh))

        # rect for floor
        p0 = self.position + Vector2(9-900/2, 512-600/2)*2/3 * self.zoom
        w = (90*2/3) * self.zoom
        self.rect_floor = pygame.Rect(p0, Vector2(w*7, w))

        # rect for grid
        p0 = self.position + Vector2(470-900/2, 15-600/2)*2/3 * self.zoom
        w = (84*2/3) * self.zoom
        self.rect_grid = []
        for iy in range(5):
            for ix in range(5):
                p1 = p0 + Vector2(ix * w, iy * w * 1.03)
                self.rect_grid.append(pygame.Rect(p1, Vector2(w, w)))

    def draw(self, surface):
        super().draw(surface)

        # draw grid
        p0 = self.position + Vector2(510-900/2, 60-600/2)*2/3 * self.zoom
        w = (85*2/3) * self.zoom
        for iy, trow in enumerate(self.content['grid']):
            for ix, t in enumerate(trow):
                # if t != 0:
                pos = p0 + Vector2(ix, iy) * w
                graph = Tile(pos, t, zoom=self.zoom)
                graph.draw(surface)

        # draw line
        p0 = self.position + Vector2(390-900/2, 60-600/2)*2/3 * self.zoom
        w = (85*2/3) * self.zoom
        for iy, trow in enumerate(self.content['line']):
            for ix, t in enumerate(trow):
                # if t != 0:
                pos = p0 + Vector2(-ix, iy) * w
                graph = Tile(pos, t, zoom=self.zoom)
                graph.draw(surface)

        # draw floor
        p0 = self.position + Vector2(50-900/2, 550-600/2)*2/3 * self.zoom
        w = (90*2/3) * self.zoom
        for i, t in enumerate(self.content['floor']):
            pos = p0 + Vector2(i, 0) * w
            graph = Tile(pos, t, zoom=self.zoom)
            graph.draw(surface)

        # draw label
        font = pygame.font.SysFont(None, 50)
        img = font.render(
            f"{self.name}: {self.content['score']}", True, BLACK, BEIGE)
        img = scale(img, Vector2(img.get_size()) * self.zoom)
        pos = self.position + Vector2(10-900/2, 10-600/2)*2/3 * self.zoom
        surface.blit(img, pos)

    def set_content(self, content):
        self.content = content

    def set_score(self, score):
        self.content['score'] = score

    def mouse_callback(self, event):
        # assume is active player, zoom=1
        # return mark and numMark. If empty, return 0, 0

        # click floor
        if self.rect_floor.collidepoint(event.pos):
            return 'row', NUM_ROWS

        # check for click on lines
        for i, rect in enumerate(self.rect_line):
            if rect.collidepoint(event.pos):
                return 'row', i

        for i in range(len(self.rect_grid)):
            if self.rect_grid[i].collidepoint(event.pos):
                row = i // NUM_ROWS
                col = i % NUM_ROWS
                return 'colIdx', (row, col)
        return None, None


class Tile(GameObject):
    idx_to_path = {
        EMPTY_TILE: "",
        0: "tile_blue.png",
        1: "tile_yellow.png",
        2: "tile_red.png",
        3: "tile_black.png",
        4: "tile_white.png",
        5: "tile_first.png",
    }

    def __init__(self, position, mark=EMPTY_TILE, size=Vector2(50, 50), zoom=1.0):
        super().__init__(
            position,
            scale(load_sprite(self.idx_to_path[mark]), size),
            size,
            zoom
        )
        self.rect = pygame.Rect(position - size/2, size)


class Factory(GameObject):
    def __init__(self, content, position, size=(150, 150), zoom=1.0):
        super().__init__(
            position,
            scale(load_sprite("factory.png"), size),
            size,
            zoom
        )
        self.content = content

        if len(content) == 0:
            self.tiles = []
        else:
            w = size[0] * 30/130 * self.zoom
            offset = [
                Vector2(-w, -w),
                Vector2(w, -w),
                Vector2(-w, w),
                Vector2(w, w)
            ]
            self.tiles = [
                Tile(position + r, content[i], zoom=self.zoom) for i, r in enumerate(offset)]
        self.rect = pygame.Rect(position - size/2, size)

    def draw(self, surface):
        super().draw(surface)

        for tile in self.tiles:
            tile.draw(surface)

    def mouse_callback(self, event):
        # return mark and numMark. If empty, return None, None
        # if move.isRegularPhase == False:
        #     return move

        if len(self.tiles) == 0:
            return None
        else:
            pos = event.pos - self.position
            tileIdx = (pos[0] > 0)*1 + (pos[1] > 0)*2
            mark = self.content[tileIdx]
            numMark = self.content.count(mark)

        return mark

        # if move.color == mark and move.factoryIdx == self.idx and move.factoryIdxOn:
        #     move.colorOn = move.factoryIdxOn = False

        # else:
        #     move.colorOn = move.factoryIdxOn = True
        #     move.color = mark
        #     move.factoryIdx = self.idx

        # print(f"Factory: {mark=}, {numMark=}")
        # return move


class Center(GameObject):
    # TODO Catch case of >20 pieces. Could reach up to 28
    def __init__(self, content, position, size=(150, 150)):
        super().__init__(
            position,
            None,
            size
        )
        self.content = content

        # position center tiles
        w = size.x / 10
        s0 = position - Vector2(w*(10-1)/2, w*(2-1)/2)
        self.tiles = [
            Tile(s0 + Vector2(i, 0)*w if i < 10
                 else s0 + Vector2(i-10, 1)*w, m)
            for i, m in enumerate(content)]

        self.rect = pygame.Rect(position - size/2, size)

    def draw(self, surface):

        for tile in self.tiles:
            tile.draw(surface)

    def mouse_callback(self, event):
        # return factoryIdx and color. If empty, return None, None
        isFirst = self.content.count(-1) > 0

        if len(self.tiles) == 0:
            return None
        else:
            pos = (event.pos - (self.position - self.size/2))
            ixy = np.floor(pos * 10 / self.size.x)

            tileIdx = int(ixy[0] + ixy[1] * 10)
            if tileIdx >= len(self.content) or self.content[tileIdx] == -1:
                mark, numMark = None, None
            else:
                mark = self.content[tileIdx]
                numMark = self.content.count(mark)

        return mark

        # if move.color == mark and move.factoryIdx == -1 and move.factoryIdxOn:  # toggle
        #     move.colorOn = move.factoryIdxOn = False
        # else:
        #     move.colorOn = move.factoryIdxOn = True
        #     move.color = mark
        #     move.factoryIdx = -1

        # # print(f"Center: {mark=}, {numMark=}, {isFirst=}")
        # return move
