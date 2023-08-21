from pygame.math import Vector2
from utils import load_sprite
import pygame
from pygame.transform import scale
import numpy as np

SIZE_SCREEN = Vector2([1280, 720])
SIZE_BOARD = Vector2([600, 400])
SIZE_FACTORY = Vector2([130, 130])
SIZE_TILE = Vector2([50, 50])


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
    def __init__(self, name, content, position, size=(600, 400), zoom=1.0):  # 900x600
        super().__init__(
            position,
            scale(load_sprite("board2.png"), size),
            Vector2(size),
            zoom
        )
        self.content = content
        self.name = name
        self.rect = pygame.Rect(position - size/2, size)
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

    def draw(self, surface):
        super().draw(surface)

        # draw grid
        p0 = self.position + Vector2(510-900/2, 60-600/2)*2/3 * self.zoom
        w = (85*2/3) * self.zoom
        for iy, trow in enumerate(self.content['grid']):
            for ix, t in enumerate(trow):
                if t != 0:
                    pos = p0 + Vector2(ix, iy) * w
                    graph = Tile(pos, t, zoom=self.zoom)
                    graph.draw(surface)

        # draw line
        p0 = self.position + Vector2(390-900/2, 60-600/2)*2/3 * self.zoom
        w = (85*2/3) * self.zoom
        for iy, trow in enumerate(self.content['line']):
            for ix, t in enumerate(trow):
                if t != 0:
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
        font = pygame.font.SysFont(None, 30)
        img = font.render(
            f"{self.name}: {self.content['score']}", True, (0, 0, 0))
        pos = self.position + Vector2(10-900/2, 10-600/2)*2/3 * self.zoom
        surface.blit(scale(img, Vector2(img.get_size()) * self.zoom), pos)

    def mouse_callback(self, event):
        # assume is active player, zoom=1
        # return mark and numMark. If empty, return 0, 0

        lineIdx = None
        if self.rect_floor.collidepoint(event.pos):
            lineIdx = -1
        else:
            # check for click on lines
            for i, rect in enumerate(self.rect_line):
                if rect.collidepoint(event.pos):
                    lineIdx = i
                    break

        print(f"Board: {lineIdx=}")
        return lineIdx


class Tile(GameObject):
    idx_to_path = {
        -1: "tile_first.png",
        1: "tile_blue.png",
        2: "tile_yellow.png",
        3: "tile_red.png",
        4: "tile_black.png",
        5: "tile_white.png"
    }

    def __init__(self, position, mark=1, size=(50, 50), zoom=1.0):
        super().__init__(
            position,
            scale(load_sprite(self.idx_to_path[mark]), size),
            size,
            zoom
        )


class Factory(GameObject):
    def __init__(self, content, position, size=(150, 150), zoom=1.0):
        super().__init__(
            position,
            scale(load_sprite("factory.png"), size),
            size,
            zoom
        )
        self.content = content

        w = size[0] * 30/130
        self.offset = [
            Vector2(-w, -w),
            Vector2(w, -w),
            Vector2(-w, w),
            Vector2(w, w)
        ]
        self.tiles = [
            Tile(position, content[i]) for i in range(len(content))]
        self.rect = pygame.Rect(position - size/2, size)

    def draw(self, surface):
        super().draw(surface)

        for tile, offset in zip(self.tiles, self.offset):
            tile.position = self.position + offset * self.zoom
            tile.zoom = self.zoom
            tile.draw(surface)

    def mouse_callback(self, event):
        # return mark and numMark. If empty, return 0, 0
        if len(self.tiles) == 0:
            mark, numMark = 0, 0
        else:
            pos = event.pos - self.position
            tileIdx = (pos[0] > 0)*1 + (pos[1] > 0)*2
            mark = self.content[tileIdx]
            numMark = self.content.count(mark)

        print(f"Factory: {mark=}, {numMark=}")
        return mark, numMark


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
        # return mark and numMark. If empty, return 0, 0
        isFirst = self.content.count(-1) > 0

        if len(self.tiles) == 0:
            mark, numMark = 0, 0
        else:
            pos = (event.pos - (self.position - self.size/2))
            ixy = np.floor(pos * 10 / self.size.x)

            tileIdx = int(ixy[0] + ixy[1] * 10)
            if tileIdx >= len(self.content):
                mark, numMark = 0, 0
            else:
                mark = self.content[tileIdx]
                numMark = self.content.count(mark)

        print(f"Center: {mark=}, {numMark=}, {isFirst=}")
        return mark, numMark, isFirst
