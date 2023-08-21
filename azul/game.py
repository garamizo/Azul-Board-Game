import pygame
from utils import load_sprite
from models import GameObject, Board, Factory, Tile, Center
from models import SIZE_SCREEN, SIZE_BOARD, SIZE_FACTORY, SIZE_TILE
from pygame.math import Vector2
from logic import Table


class Azul:
    def __init__(self):
        self._init_pygame()

        self.logic = Table()
        # self.logic.reset()
        obs = self.logic.get_observation()

        self.screen = pygame.display.set_mode(SIZE_SCREEN)
        self.clock = pygame.time.Clock()

        numPlayers, numFactories = len(obs['player']), len(obs['factory'])
        sb, sf, s = SIZE_BOARD, SIZE_FACTORY, SIZE_SCREEN
        st = SIZE_TILE[0]
        ssb = SIZE_BOARD * 0.6
        g, x = 10, SIZE_BOARD[0] * 0.6
        w = s[0] - numPlayers*g - (numPlayers-1)*x
        y = s[1] - ssb[1]/2 - g
        p1 = obs['activePlayer']

        # position boards
        self.board = [
            Board(f'Player {i+1}', b, (g + (g+x)*i + x/2, y), size=SIZE_BOARD, zoom=0.6) if i < p1
            else (Board(f'Player {i+1}', b, (g + (g+x)*(i-1) + x/2 + w, y), size=SIZE_BOARD, zoom=0.6) if i > p1
                  else Board(f'Player {i+1}', b, SIZE_BOARD / 2 + Vector2(g, g), size=SIZE_BOARD, zoom=1.0))
            for i, b in enumerate(obs['player'])
        ]
        self.activeBoard = self.board[p1]

        # position factories
        x1, x2 = sb.x + sf.x/2 + 1.5*g, s.x - sf.y/2 - 0.5*g
        y1, y2 = (sb.y + 2*g)/2 - (sf.y+2*st+g)/2 - g, \
            (sb.y + 2*g)/2 + (sf.y+2*st+g)/2 + g
        self.factory = [
            Factory(f, (x1 + (x2-x1)*i/4, y1), size=SIZE_FACTORY) if i < 5
            else Factory(f, (x1 + (x2-x1)*(i-5)/4, y2), size=SIZE_FACTORY)
            for i, f in enumerate(obs['factory'])]

        # position center tiles
        sz = Vector2(s.x - sb.x - 3*g, 2*st + 2*g)
        s0 = Vector2(sb.x + 2*g + sz.x/2, sb.y/2 + g)
        self.center = Center(obs['center'], s0, sz)

    def main_loop(self):
        while True:
            self._handle_input()
            self._process_game_logic()
            self._draw()

    def _init_pygame(self):
        pygame.init()
        pygame.display.set_caption("Azul")

    def _handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:

                if self.activeBoard.rect.collidepoint(event.pos):
                    self.activeBoard.mouse_callback(event)
                    break

                if self.center.rect.collidepoint(event.pos):
                    self.center.mouse_callback(event)
                    break

                for factory in self.factory:
                    if factory.rect.collidepoint(event.pos):
                        factory.mouse_callback(event)

    def _process_game_logic(self):
        pass

    def _draw(self):
        # self.screen.fill((220, 220, 100))
        self.screen.fill((0, 0, 0))

        for board in self.board:
            board.draw(self.screen)

        # self.screen.blit(self.factory, (800, 50))
        for factory in self.factory:
            factory.draw(self.screen)

        # self.factory.draw(self.screen)
        self.center.draw(self.screen)
        # for tile in self.center:
        #     tile.draw(self.screen)

        pygame.display.flip()
        self.clock.tick(20)
