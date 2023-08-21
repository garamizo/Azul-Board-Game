from pygame.math import Vector2
from utils import load_sprite
import pygame
from pygame.transform import scale
import numpy as np


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
    def __init__(self, content, position, size=(600, 400), zoom=1.0):  # 900x600
        super().__init__(
            position,
            scale(load_sprite("board2.png"), size),
            Vector2(size),
            zoom
        )
        self.content = content
        # {'grid': np.array 5x5,
        #  'line': [[]]*5,
        #  'score': int,
        #  'floor': []}

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
    def __init__(self, tiles, position, size=(150, 150), zoom=1.0):
        super().__init__(
            position,
            scale(load_sprite("factory.png"), size),
            size,
            zoom
        )

        w = size[0] * 30/130
        self.offset = [
            Vector2(-w, -w),
            Vector2(w, -w),
            Vector2(-w, w),
            Vector2(w, w)
        ]
        self.tiles = [
            Tile(position, tiles[i]) for i in range(len(tiles))]

    def draw(self, surface):
        super().draw(surface)

        for tile, offset in zip(self.tiles, self.offset):
            tile.position = self.position + offset * self.zoom
            tile.zoom = self.zoom
            tile.draw(surface)
